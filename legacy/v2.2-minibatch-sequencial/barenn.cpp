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
    // Gradientes acumulados ao longo do lote (os pesos só mudam no fim dele).
    vector<float> dW;
    vector<float> db;
    // Buffers fixos: soma antes da ativação, saída, delta e gradiente da entrada.
    vector<float> last_z;
    vector<float> out_;
    vector<float> delta_;
    vector<float> grad_in_;
    // Entrada do último forward (aponta para o buffer da camada anterior ou da amostra).
    const float* last_in_ = nullptr;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(output_size) * input_size), bias(output_size, 0),
          dW(static_cast<size_t>(output_size) * input_size, 0.0f), db(output_size, 0.0f),
          last_z(output_size), out_(output_size), delta_(output_size),
          grad_in_(input_size) {
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

    // Forward de treino: guarda a entrada, que o backward usa para o gradiente dos pesos.
    void ForwardTreino(const float* input) {
        last_in_ = input;
        ForwardRange(input, 0, out_features);
    }

    // Backward de uma amostra. Não altera os pesos: só acumula o gradiente do lote
    // (dW, db) e calcula o gradiente da entrada, que usa os pesos do início do lote.
    void Backward(const float* grad_out, bool need_grad_in) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta_[i] = final ? grad_out[i]
                              : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        if (need_grad_in) {
            fill(grad_in_.begin(), grad_in_.end(), 0.0f);
            for (int j = 0; j < out_features; ++j) {
                const float d = delta_[j];
                if (d == 0.0f) continue;
                const float* w = &weights[j * in_features];
                for (int i = 0; i < in_features; ++i) {
                    grad_in_[i] += d * w[i];
                }
            }
        }

        // O gradiente do peso é delta vezes a entrada; o do viés é o próprio delta.
        for (int i = 0; i < out_features; ++i) {
            const float d = delta_[i];
            if (d == 0.0f) continue;
            float* g = &dW[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                g[j] += d * last_in_[j];
            }
            db[i] += d;
        }
    }

    // Fim do lote: uma única atualização com o gradiente acumulado (k = taxa / amostras).
    void AplicarLote(float k) {
        const size_t n = weights.size();
        for (size_t p = 0; p < n; ++p) {
            weights[p] -= k * dW[p];
            dW[p] = 0.0f;
        }
        for (int i = 0; i < out_features; ++i) {
            bias[i] -= k * db[i];
            db[i] = 0.0f;
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

// Treino por mini-batch: dentro de um lote os pesos não mudam e cada amostra só acumula
// gradientes; ao fim do lote há uma única atualização, com o gradiente médio.
void treinar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3, const vector<Amostra>& treino,
             float taxa, size_t tamanho_lote, int epocas, double* perdas, float* gradiente) {
    for (int epoca = 0; epoca < epocas; ++epoca) {
        double perda_total = 0;
        // Uma época percorre todo o conjunto na ordem do arquivo.
        for (size_t inicio = 0; inicio < treino.size(); inicio += tamanho_lote) {
            const size_t fim = min(treino.size(), inicio + tamanho_lote);
            for (size_t s = inicio; s < fim; ++s) {
                const auto& amostra = treino[s];
                l1.ForwardTreino(amostra.imagem.data());
                l2.ForwardTreino(l1.saida());
                l3.ForwardTreino(l2.saida());
                perda_total += perda(l3.saida(), amostra.rotulo, gradiente);

                // Propaga o gradiente da última camada até a primeira.
                l3.Backward(gradiente, true);
                l2.Backward(l3.grad_in(), true);
                l1.Backward(l2.grad_in(), false);
            }
            const float k = taxa / static_cast<float>(fim - inicio);
            l1.AplicarLote(k);
            l2.AplicarLote(k);
            l3.AplicarLote(k);
        }
        perdas[epoca] = perda_total / treino.size();
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
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};

        // A medição inclui forward, perda, backward e atualização de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        Barenn::treinar(l1, l2, l3, treino, taxa, tamanho_lote, epocas,
                        perdas.data(), gradiente.data());
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
