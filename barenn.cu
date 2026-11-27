// Porte do treino por mini-batch (v2.2/v2.3) para CUDA: versão funcional, sem otimização.
// Um kernel por etapa; dados e pesos ficam na GPU durante todo o treino. Todo o código de
// CPU (leitura dos dados, avaliação final, inicialização dos pesos) vem de barenn.cpp.
//
// Compilação: nvcc -O2 -std=c++17 --fmad=false -arch=sm_70 barenn.cu -o barenn_cuda
// (--fmad=false evita multiplicações-somas fundidas, que arredondam diferente da CPU.)

#define main main_cpu
#include "barenn.cpp"
#undef main

#include <cuda_runtime.h>

#define CUDA_CHECK(chamada)                                                              \
    do {                                                                                 \
        cudaError_t erro_ = (chamada);                                                   \
        if (erro_ != cudaSuccess) {                                                      \
            throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro_) + " em " + \
                                #chamada);                                               \
        }                                                                                \
    } while (0)

namespace BarennCuda {

constexpr int bloco = 128;

inline int grade(int total) { return (total + bloco - 1) / bloco; }

// Forward de uma camada: uma thread por par (amostra, neurônio). Cada soma segue a ordem
// da CPU: parte do viés e soma entrada * peso para j crescente.
__global__ void forward(const float* in, int in_stride, const float* W, const float* b,
                        float* z, float* a, int IN, int OUT, int relu, int nb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * OUT) return;
    const int s = idx / OUT;
    const int i = idx % OUT;
    float soma = b[i];
    for (int j = 0; j < IN; ++j) {
        soma += in[s * in_stride + j] * W[i * IN + j];
    }
    z[idx] = soma;
    a[idx] = relu ? fmaxf(0.0f, soma) : soma;
}

// Perda e gradiente dos escores: uma thread por amostra. A perda de cada amostra vai para
// um vetor (somado na CPU em ordem, ao fim da época).
__global__ void perda_gradiente(const int* rotulos, const float* z3, float* d3, float* perdas,
                                int nb) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nb) return;
    float total = 0.0f;
    for (int i = 0; i < Barenn::classes; ++i) {
        const float erro = z3[s * Barenn::classes + i] - (i == rotulos[s] ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        d3[s * Barenn::classes + i] = erro;
    }
    perdas[s] = total;
}

// Delta da camada anterior: grad_in = soma_i delta_next[i] * W[i][j] (i crescente, pulando
// delta nulo, como na CPU) vezes a derivada da ReLU. Uma thread por par (amostra, entrada j).
__global__ void delta_anterior(const float* d_next, const float* W, const float* z_prev,
                               float* d_prev, int IN, int OUT, int nb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * IN) return;
    const int s = idx / IN;
    const int j = idx % IN;
    float gin = 0.0f;
    for (int i = 0; i < OUT; ++i) {
        const float d = d_next[s * OUT + i];
        if (d == 0.0f) continue;
        gin += d * W[i * IN + j];
    }
    d_prev[idx] = gin * (z_prev[idx] > 0.0f ? 1.0f : 0.0f);
}

// Gradiente dos pesos e do viés do lote: uma thread por parâmetro, somando as amostras em
// ordem crescente (sem atomics: o resultado é determinístico). G tem o mesmo layout de
// P = [pesos (OUT*IN) | viés (OUT)].
__global__ void gradiente_pesos(const float* delta, const float* in, int in_stride, float* G,
                                int IN, int OUT, int nb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OUT * IN + OUT) return;
    float g = 0.0f;
    if (idx < OUT * IN) {
        const int i = idx / IN;
        const int j = idx % IN;
        for (int s = 0; s < nb; ++s) {
            const float d = delta[s * OUT + i];
            if (d == 0.0f) continue;
            g += d * in[s * in_stride + j];
        }
    } else {
        const int i = idx - OUT * IN;
        for (int s = 0; s < nb; ++s) {
            const float d = delta[s * OUT + i];
            if (d == 0.0f) continue;
            g += d;
        }
    }
    G[idx] = g;
}

// Atualização: P -= k * G, com k = taxa / amostras do lote.
__global__ void atualizar(float* P, const float* G, float k, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) P[idx] -= k * G[idx];
}

// Buffers de uma camada na GPU: parâmetros, gradientes, ativações e deltas por amostra.
struct CamadaGPU {
    int IN = 0, OUT = 0;
    float *P = nullptr, *G = nullptr;  // [pesos | viés]
    float *z = nullptr, *a = nullptr, *delta = nullptr;
    size_t n_params() const { return static_cast<size_t>(OUT) * IN + OUT; }
    float* pesos() const { return P; }
    float* vies() const { return P + static_cast<size_t>(OUT) * IN; }

    void alocar(int entradas, int saidas, int lote) {
        IN = entradas;
        OUT = saidas;
        CUDA_CHECK(cudaMalloc(&P, n_params() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&G, n_params() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z, static_cast<size_t>(lote) * OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a, static_cast<size_t>(lote) * OUT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&delta, static_cast<size_t>(lote) * OUT * sizeof(float)));
    }
    void liberar() {
        cudaFree(P); cudaFree(G); cudaFree(z); cudaFree(a); cudaFree(delta);
    }
    // Copia pesos e viés da camada de CPU (H2D) ou de volta para ela (D2H).
    void enviar(Barenn::DenseLayer& l) {
        CUDA_CHECK(cudaMemcpy(pesos(), l.pesos(), static_cast<size_t>(OUT) * IN * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vies(), l.vies(), OUT * sizeof(float), cudaMemcpyHostToDevice));
    }
    void receber(Barenn::DenseLayer& l) {
        CUDA_CHECK(cudaMemcpy(l.pesos(), pesos(), static_cast<size_t>(OUT) * IN * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(l.vies(), vies(), OUT * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

// Treina na GPU e devolve o tempo em segundos. O tempo inclui cudaMalloc e as transferências
// H2D e D2H, e exclui a criação do contexto CUDA (feita antes de iniciar o cronômetro).
double treinar(Barenn::DenseLayer& l1, Barenn::DenseLayer& l2, Barenn::DenseLayer& l3,
               const vector<Barenn::Amostra>& treino, float taxa, int tamanho_lote, int epocas,
               double* perdas) {
    const int N = static_cast<int>(treino.size());
    CUDA_CHECK(cudaFree(0));  // cria o contexto CUDA fora da medição

    // Dados de treino em formato contíguo (a leitura continua sendo a da CPU).
    vector<float> X(static_cast<size_t>(N) * Barenn::entradas);
    vector<int> y(N);
    for (int s = 0; s < N; ++s) {
        copy(treino[s].imagem.begin(), treino[s].imagem.end(),
             X.begin() + static_cast<size_t>(s) * Barenn::entradas);
        y[s] = treino[s].rotulo;
    }

    const auto inicio_t = chrono::steady_clock::now();

    float* d_X = nullptr;
    int* d_y = nullptr;
    float* d_perdas = nullptr;
    float* d3 = nullptr;
    CamadaGPU c1, c2, c3;
    CUDA_CHECK(cudaMalloc(&d_X, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_perdas, N * sizeof(float)));
    c1.alocar(Barenn::entradas, 32, tamanho_lote);
    c2.alocar(32, 16, tamanho_lote);
    c3.alocar(16, Barenn::classes, tamanho_lote);
    d3 = c3.delta;
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    c1.enviar(l1);
    c2.enviar(l2);
    c3.enviar(l3);

    vector<float> perdas_amostra(N);
    for (int epoca = 0; epoca < epocas; ++epoca) {
        // Uma época percorre todo o conjunto na ordem do arquivo.
        for (int inicio = 0; inicio < N; inicio += tamanho_lote) {
            const int nb = min(tamanho_lote, N - inicio);
            const float* x = d_X + static_cast<size_t>(inicio) * Barenn::entradas;

            // Forward das três camadas (a última é linear: a = z).
            forward<<<grade(nb * c1.OUT), bloco>>>(x, Barenn::entradas, c1.pesos(), c1.vies(),
                                                   c1.z, c1.a, c1.IN, c1.OUT, 1, nb);
            forward<<<grade(nb * c2.OUT), bloco>>>(c1.a, c1.OUT, c2.pesos(), c2.vies(), c2.z,
                                                   c2.a, c2.IN, c2.OUT, 1, nb);
            forward<<<grade(nb * c3.OUT), bloco>>>(c2.a, c2.OUT, c3.pesos(), c3.vies(), c3.z,
                                                   c3.a, c3.IN, c3.OUT, 0, nb);
            // Perda e gradiente dos escores (delta da camada 3).
            perda_gradiente<<<grade(nb), bloco>>>(d_y + inicio, c3.z, d3, d_perdas + inicio, nb);
            // Deltas das camadas 2 e 1.
            delta_anterior<<<grade(nb * c3.IN), bloco>>>(c3.delta, c3.pesos(), c2.z, c2.delta,
                                                         c3.IN, c3.OUT, nb);
            delta_anterior<<<grade(nb * c2.IN), bloco>>>(c2.delta, c2.pesos(), c1.z, c1.delta,
                                                         c2.IN, c2.OUT, nb);
            // Gradiente dos pesos de cada camada (usa os pesos do início do lote).
            gradiente_pesos<<<grade(c3.n_params()), bloco>>>(c3.delta, c2.a, c2.OUT, c3.G, c3.IN,
                                                             c3.OUT, nb);
            gradiente_pesos<<<grade(c2.n_params()), bloco>>>(c2.delta, c1.a, c1.OUT, c2.G, c2.IN,
                                                             c2.OUT, nb);
            gradiente_pesos<<<grade(c1.n_params()), bloco>>>(c1.delta, x, Barenn::entradas,
                                                             c1.G, c1.IN, c1.OUT, nb);
            // Uma única atualização por lote, com o gradiente médio.
            const float k = taxa / static_cast<float>(nb);
            atualizar<<<grade(c3.n_params()), bloco>>>(c3.P, c3.G, k, c3.n_params());
            atualizar<<<grade(c2.n_params()), bloco>>>(c2.P, c2.G, k, c2.n_params());
            atualizar<<<grade(c1.n_params()), bloco>>>(c1.P, c1.G, k, c1.n_params());
        }
        CUDA_CHECK(cudaGetLastError());
        // Perda da época: soma das perdas por amostra, em ordem (como a CPU).
        CUDA_CHECK(cudaMemcpy(perdas_amostra.data(), d_perdas, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double total = 0;
        for (int s = 0; s < N; ++s) total += perdas_amostra[s];
        perdas[epoca] = total / N;
    }

    // Os pesos voltam à CPU para a avaliação final.
    c1.receber(l1);
    c2.receber(l2);
    c3.receber(l3);
    CUDA_CHECK(cudaDeviceSynchronize());
    const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio_t;

    c1.liberar(); c2.liberar(); c3.liberar();
    cudaFree(d_X); cudaFree(d_y); cudaFree(d_perdas);
    return tempo.count();
}

}  // namespace BarennCuda

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
        constexpr int tamanho_lote = 64;
        constexpr float taxa = 0.3f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        const double tempo = BarennCuda::treinar(l1, l2, l3, treino, taxa, tamanho_lote, epocas,
                                                 perdas.data());

        // Imprime as métricas e roda a avaliação (na CPU) após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
