#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

class DenseLayer {
    static constexpr int max_saidas = 32;
    int in_features;
    int out_features;
    bool final;
    // weights[i * in_features + j] liga a entrada j ao neurônio de saída i.
    // Matriz achatada: um único bloco contíguo, linha a linha.
    vector<float> weights;
    vector<float> bias;
    // Buffers fixos: soma antes da ativação, saída da camada e gradiente da entrada.
    vector<float> last_z;
    vector<float> out_;
    vector<float> grad_in_;
    // Entrada do último forward (aponta para o buffer da camada anterior ou da amostra).
    const float* last_in_ = nullptr;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(output_size) * input_size), bias(output_size, 0),
          last_z(output_size), out_(output_size), grad_in_(input_size) {
        if (output_size > max_saidas) {
            throw runtime_error("DenseLayer: no máximo 32 neurônios por camada");
        }
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
    }

    const float* saida() const { return out_.data(); }
    const float* grad_in() const { return grad_in_.data(); }

    // Calcula os neurônios [n0, n1). Cada neurônio soma o viés aos produtos entre
    // entradas e pesos em ordem crescente de j, quatro neurônios por vez, cada um
    // com seu acumulador (as cadeias de soma ficam independentes).
    void ForwardRange(const float* input, int n0, int n1) {
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
                last_z[i + k] = sums[k];
                // As camadas ocultas usam ReLU. A última devolve escores lineares.
                out_[i + k] = final ? sums[k] : max(0.0f, sums[k]);
            }
        }
        for (; i < n1; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i * in_features + j];
            }
            last_z[i] = sum;
            out_[i] = final ? sum : max(0.0f, sum);
        }
    }

    // Forward sequencial (usado na avaliação).
    vector<float> Forward(const vector<float>& input) {
        ForwardRange(input.data(), 0, out_features);
        return out_;
    }

    // Forward de treino: a thread tid de T calcula uma faixa de blocos de 4 neurônios.
    // Depois dele é preciso uma barreira antes de outra thread ler saida().
    void ForwardPar(const float* input, int tid, int T) {
        last_in_ = input;
        const int blocos = (out_features + 3) / 4;
        const int n0 = min(out_features, 4 * (blocos * tid / T));
        const int n1 = min(out_features, 4 * (blocos * (tid + 1) / T));
        ForwardRange(input, n0, n1);
    }

    // Backward de treino, dividido entre T threads. Cada elemento calculado (um
    // grad_in[i], uma linha de pesos) pertence a uma única thread e é somado na
    // mesma ordem do código sequencial. Com need_grad_in há uma barreira interna
    // entre o cálculo de grad_in (que usa os pesos antigos) e a atualização dos pesos.
    void BackwardPar(const float* grad_out, float learning_rate, bool need_grad_in,
                     int tid, int T) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        // Cada thread calcula o vetor inteiro (é pequeno), em memória própria.
        float delta[max_saidas];
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        if (need_grad_in) {
            // A thread calcula uma faixa de entradas i; para cada uma, j crescente.
            const int i0 = in_features * tid / T;
            const int i1 = in_features * (tid + 1) / T;
            for (int i = i0; i < i1; ++i) grad_in_[i] = 0.0f;
            for (int j = 0; j < out_features; ++j) {
                const float d = delta[j];
                if (d == 0.0f) continue;
                const float* w = &weights[j * in_features];
                for (int i = i0; i < i1; ++i) {
                    grad_in_[i] += d * w[i];
                }
            }
            BARENN_BARREIRA
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta. Cada thread atualiza suas linhas.
        const int r0 = out_features * tid / T;
        const int r1 = out_features * (tid + 1) / T;
        for (int i = r0; i < r1; ++i) {
            if (delta[i] == 0.0f) continue;
            float* w = &weights[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                w[j] -= learning_rate * (delta[i] * last_in_[j]);
            }
            bias[i] -= learning_rate * delta[i];
        }
    }
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

// Laço de treino executado por todas as threads de uma região OpenMP (a thread 0 é
// a principal). Todas percorrem as mesmas amostras na mesma ordem; as barreiras
// separam as etapas e nenhuma soma é repartida entre threads.
void treinar(int tid, int T, DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& treino, float taxa, int epocas,
             double* perdas, float* gradiente) {
    for (int epoca = 0; epoca < epocas; ++epoca) {
        double perda_total = 0;
        // Uma época percorre todo o conjunto na ordem do arquivo.
        for (const auto& amostra : treino) {
            l1.ForwardPar(amostra.imagem.data(), tid, T);
            BARENN_BARREIRA
            l2.ForwardPar(l1.saida(), tid, T);
            BARENN_BARREIRA
            l3.ForwardPar(l2.saida(), tid, T);
            BARENN_BARREIRA
            // A perda e o gradiente dos escores são calculados só pela thread 0.
            if (tid == 0) {
                perda_total += perda(l3.saida(), amostra.rotulo, gradiente);
            }
            BARENN_BARREIRA

            // Propaga o gradiente da última camada até a primeira.
            // Cada Backward também atualiza os parâmetros da sua camada.
            l3.BackwardPar(gradiente, taxa, true, tid, T);
            l2.BackwardPar(l3.grad_in(), taxa, true, tid, T);
            l1.BackwardPar(l2.grad_in(), taxa, false, tid, T);
            BARENN_BARREIRA
        }
        if (tid == 0) perdas[epoca] = perda_total / treino.size();
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
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};

        // Número de threads: OMP_NUM_THREADS (padrão 1). O posicionamento das threads
        // fica por conta de OMP_PLACES e OMP_PROC_BIND.
        int threads = 1;
        if (const char* env = getenv("OMP_NUM_THREADS")) threads = max(1, atoi(env));

        // A medição inclui forward, perda e backward de todas as épocas.
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
            Barenn::treinar(tid, T, l1, l2, l3, treino, taxa, epocas,
                            perdas.data(), gradiente.data());
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
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
