#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Paralelismo multicore com OpenMP. Sem -fopenmp o programa continua sequencial.
#ifdef _OPENMP
#include <omp.h>
// Com uma thread só, a barreira é dispensada (evita o custo da chamada da biblioteca).
#define BARENN_BARREIRA if (T > 1) { _Pragma("omp barrier") }
#else
#define BARENN_BARREIRA
#endif

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Espaço de trabalho de uma camada para UMA thread: ativações, delta, gradiente da entrada
// e os acumuladores de gradiente do lote (dW, db).
struct CamadaWS {
    vector<float> z, out, delta, grad_in, dW, db;
    const float* in = nullptr;
    CamadaWS(int entradas, int saidas, size_t n_pesos)
        : z(saidas), out(saidas), delta(saidas), grad_in(entradas),
          dW(n_pesos, 0.0f), db(saidas, 0.0f) {}
};

// Camada densa: guarda só os pesos e o viés, que são somente leitura durante o cálculo
// de um lote. Todo o estado que muda por amostra fica no CamadaWS de cada thread.
class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // weights[i * in_features + j] liga a entrada j ao neurônio de saída i.
    // Matriz achatada: um único bloco contíguo, linha a linha.
    vector<float> weights;
    vector<float> bias;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(output_size) * input_size), bias(output_size, 0) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
    }

    int entradas() const { return in_features; }
    int saidas() const { return out_features; }
    size_t n_pesos() const { return weights.size(); }
    float* pesos() { return weights.data(); }
    float* vies() { return bias.data(); }

    // Calcula os neurônios [n0, n1). Cada neurônio soma o viés aos produtos entre
    // entradas e pesos em ordem crescente de j, quatro neurônios por vez, cada um
    // com seu acumulador (as cadeias de soma ficam independentes).
    void ForwardRange(const float* input, CamadaWS& ws, int n0, int n1) const {
        int i = n0;
        for (; i + 4 <= n1; i += 4) {
            float s0 = bias[i], s1 = bias[i + 1], s2 = bias[i + 2], s3 = bias[i + 3];
            const float* w0 = &weights[i * in_features];
            const float* w1 = w0 + in_features;
            const float* w2 = w1 + in_features;
            const float* w3 = w2 + in_features;
            for (int j = 0; j < in_features; ++j) {
                const float x = input[j];
                s0 += x * w0[j];
                s1 += x * w1[j];
                s2 += x * w2[j];
                s3 += x * w3[j];
            }
            const float sums[4] = {s0, s1, s2, s3};
            for (int k = 0; k < 4; ++k) {
                ws.z[i + k] = sums[k];
                // As camadas ocultas usam ReLU. A última devolve escores lineares.
                ws.out[i + k] = final ? sums[k] : max(0.0f, sums[k]);
            }
        }
        for (; i < n1; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i * in_features + j];
            }
            ws.z[i] = sum;
            ws.out[i] = final ? sum : max(0.0f, sum);
        }
    }

    // Forward sequencial (usado na avaliação).
    vector<float> Forward(const vector<float>& input) const {
        CamadaWS ws(in_features, out_features, 0);
        ForwardRange(input.data(), ws, 0, out_features);
        return ws.out;
    }

    // Forward de treino: guarda a entrada, que o backward usa para o gradiente dos pesos.
    void ForwardTreino(const float* input, CamadaWS& ws) const {
        ws.in = input;
        ForwardRange(input, ws, 0, out_features);
    }

    // Backward de uma amostra. Não altera os pesos: só acumula o gradiente do lote
    // (dW, db) no espaço da thread e calcula o gradiente da entrada.
    void Backward(const float* grad_out, CamadaWS& ws, bool need_grad_in) const {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            ws.delta[i] = final ? grad_out[i]
                                : grad_out[i] * (ws.z[i] > 0.0f ? 1.0f : 0.0f);
        }

        if (need_grad_in) {
            fill(ws.grad_in.begin(), ws.grad_in.end(), 0.0f);
            for (int j = 0; j < out_features; ++j) {
                const float d = ws.delta[j];
                if (d == 0.0f) continue;
                const float* w = &weights[j * in_features];
                for (int i = 0; i < in_features; ++i) {
                    ws.grad_in[i] += d * w[i];
                }
            }
        }

        // O gradiente do peso é delta vezes a entrada; o do viés é o próprio delta.
        for (int i = 0; i < out_features; ++i) {
            const float d = ws.delta[i];
            if (d == 0.0f) continue;
            float* g = &ws.dW[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                g[j] += d * ws.in[j];
            }
            ws.db[i] += d;
        }
    }
};

// Espaço de trabalho completo de uma thread (as três camadas). Alinhado a 64 bytes para
// que os cabeçalhos de threads diferentes não dividam linhas de cache.
struct alignas(64) Workspace {
    CamadaWS c[3];
    float gradiente[10];
    Workspace(const DenseLayer& l1, const DenseLayer& l2, const DenseLayer& l3)
        : c{CamadaWS(l1.entradas(), l1.saidas(), l1.n_pesos()),
            CamadaWS(l2.entradas(), l2.saidas(), l2.n_pesos()),
            CamadaWS(l3.entradas(), l3.saidas(), l3.n_pesos())},
          gradiente{} {}
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

float perda(const vector<float>& escores, int rotulo) {
    return perda(escores.data(), rotulo);
}

// Atualiza os pesos de uma camada com o gradiente médio do lote. Cada elemento soma os
// acumuladores das T threads na ordem fixa de thread 0 a T-1. Com `paralela`, cada thread
// atualiza uma faixa dos pesos; senão, só a thread 0 atualiza todos.
void atualizar_camada(DenseLayer& camada, int l, vector<unique_ptr<Workspace>>& wss,
                      float k, int tid, int T, bool paralela) {
    const size_t n = camada.n_pesos();
    const size_t nb = camada.saidas();
    size_t p0 = 0, p1 = n, b0 = 0, b1 = nb;
    if (paralela) {
        p0 = n * tid / T;     p1 = n * (tid + 1) / T;
        b0 = nb * tid / T;    b1 = nb * (tid + 1) / T;
    } else if (tid != 0) {
        return;
    }
    float* W = camada.pesos();
    float* B = camada.vies();
    for (size_t p = p0; p < p1; ++p) {
        float soma = wss[0]->c[l].dW[p];
        for (int t = 1; t < T; ++t) soma += wss[t]->c[l].dW[p];
        W[p] -= k * soma;
    }
    for (size_t i = b0; i < b1; ++i) {
        float soma = wss[0]->c[l].db[i];
        for (int t = 1; t < T; ++t) soma += wss[t]->c[l].db[i];
        B[i] -= k * soma;
    }
}

// Treino por mini-batch, executado por todas as threads de uma região OpenMP. Dentro de
// um lote os pesos não mudam: cada thread processa uma faixa contígua das amostras do lote
// e acumula gradientes no seu espaço; depois da barreira, os acumuladores são somados e os
// pesos atualizados uma única vez; uma segunda barreira libera o lote seguinte.
void treinar(int tid, int T, DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             vector<unique_ptr<Workspace>>& wss, double* parciais,
             const vector<Amostra>& treino, float taxa, size_t tamanho_lote, int epocas,
             double* perdas, bool atualizacao_paralela) {
    // Cada thread cria o próprio espaço de trabalho (alocação local à thread).
    wss[tid] = make_unique<Workspace>(l1, l2, l3);
    BARENN_BARREIRA
    Workspace& ws = *wss[tid];
    DenseLayer* camadas[3] = {&l1, &l2, &l3};

    for (int epoca = 0; epoca < epocas; ++epoca) {
        double perda_parcial = 0;
        // Uma época percorre todo o conjunto na ordem do arquivo.
        for (size_t inicio = 0; inicio < treino.size(); inicio += tamanho_lote) {
            const size_t fim = min(treino.size(), inicio + tamanho_lote);
            const size_t nlote = fim - inicio;
            // Zera os acumuladores da própria thread (a atualização só os lê).
            for (auto& c : ws.c) {
                fill(c.dW.begin(), c.dW.end(), 0.0f);
                fill(c.db.begin(), c.db.end(), 0.0f);
            }
            const size_t s0 = inicio + nlote * tid / T;
            const size_t s1 = inicio + nlote * (tid + 1) / T;
            for (size_t s = s0; s < s1; ++s) {
                const auto& amostra = treino[s];
                l1.ForwardTreino(amostra.imagem.data(), ws.c[0]);
                l2.ForwardTreino(ws.c[0].out.data(), ws.c[1]);
                l3.ForwardTreino(ws.c[1].out.data(), ws.c[2]);
                perda_parcial += perda(ws.c[2].out.data(), amostra.rotulo, ws.gradiente);

                // Propaga o gradiente da última camada até a primeira.
                l3.Backward(ws.gradiente, ws.c[2], true);
                l2.Backward(ws.c[2].grad_in.data(), ws.c[1], true);
                l1.Backward(ws.c[1].grad_in.data(), ws.c[0], false);
            }
            BARENN_BARREIRA  // os gradientes do lote estão prontos
            const float k = taxa / static_cast<float>(nlote);
            for (int l = 0; l < 3; ++l) {
                atualizar_camada(*camadas[l], l, wss, k, tid, T, atualizacao_paralela);
            }
            BARENN_BARREIRA  // os pesos novos estão prontos para o lote seguinte
        }
        parciais[tid] = perda_parcial;
        BARENN_BARREIRA
        if (tid == 0) {
            double total = 0;
            for (int t = 0; t < T; ++t) total += parciais[t];
            perdas[epoca] = total / treino.size();
        }
    }
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        auto escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "data/optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "data/optdigits.tes";

    try {
        auto treino_total = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        // Mini-batch: tamanho do lote e taxa para o gradiente médio. As variáveis
        // BARENN_LOTE e BARENN_TAXA servem só para a calibração (scripts/calibrar_minibatch.sh).
        size_t tamanho_lote = 64;
        float taxa = 0.3f;
        if (const char* env = getenv("BARENN_LOTE")) tamanho_lote = max(1, atoi(env));
        if (const char* env = getenv("BARENN_TAXA")) taxa = strtof(env, nullptr);

        // Com BARENN_VALIDACAO treina em 4/5 do conjunto de treino e avalia no 1/5 restante,
        // sem tocar no conjunto de teste (usado para escolher a taxa).
        const bool modo_validacao = getenv("BARENN_VALIDACAO") != nullptr;
        vector<Barenn::Amostra> treino, validacao;
        if (modo_validacao) {
            for (size_t i = 0; i < treino_total.size(); ++i) {
                (i % 5 == 0 ? validacao : treino).push_back(move(treino_total[i]));
            }
        } else {
            treino = move(treino_total);
        }

        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // Número de threads: OMP_NUM_THREADS (padrão 1). O posicionamento das threads fica
        // por conta de OMP_PLACES e OMP_PROC_BIND. BARENN_ATUALIZACAO_SEQUENCIAL=1 faz só a
        // thread 0 somar os gradientes e atualizar os pesos (para medir cada passo).
        int threads = 1;
        if (const char* env = getenv("OMP_NUM_THREADS")) threads = max(1, atoi(env));
        const bool atualizacao_paralela = getenv("BARENN_ATUALIZACAO_SEQUENCIAL") == nullptr;
        vector<unique_ptr<Barenn::Workspace>> wss(threads);
        vector<double> parciais(threads, 0.0);

        // A medição inclui forward, perda, backward e atualização de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
#pragma omp parallel num_threads(threads)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
            const int T = omp_get_num_threads();
#else
            const int tid = 0;
            const int T = 1;
#endif
            Barenn::treinar(tid, T, l1, l2, l3, wss, parciais.data(), treino, taxa,
                            tamanho_lote, epocas, perdas.data(), atualizacao_paralela);
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, modo_validacao ? validacao : teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
