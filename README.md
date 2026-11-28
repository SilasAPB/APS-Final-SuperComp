# Projeto Barenn: relatório de otimização (CPU e GPU)

Relatório executivo da APS Final de SuperComputação: otimização de uma rede neural feedforward legada (64 → 32 → 16 → 10, ReLU, dígitos `optdigits`), primeiro na CPU e depois em CUDA. O enunciado original está em [enunciado.md](enunciado.md). Cada intervenção é um **bloco lógico**: uma Issue com diagnóstico, hipótese e métrica base; uma branch e um PR; uma tag anotada; e um resultado fechado na própria Issue, com os dados de profiling.

## Tabela consolidada

Tempo de **treino** (3.823 amostras × 51 épocas), no Santos Dumont (SD): CPU Xeon Gold 6252, GPU Tesla V100. `-O2`, g++ 15.2.0; CUDA 12.6.

| Versão | Tag | O que é | Tempo de treino (SD) | Speedup sobre o baseline | Acurácia de teste |
|:--|:--|:--|--:|--:|:-:|
| **Baseline Legado** | [`v0.1-baseline`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v0.1-baseline) | SGD por amostra, 1 thread | 1,2418 s | 1,0× | 95,5 % |
| **CPU Otimizada** (sequencial) | [`v1.6-forward-unroll-jam`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.6-forward-unroll-jam) | mesmo algoritmo, 1 thread, v1.1 a v1.6 | 0,5031 s | **2,47×** ¹ | 95,5 % (saída idêntica) |
| **CPU Otimizada** (paralela) | [`v2.3-minibatch-paralelo`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v2.3-minibatch-paralelo) | mini-batch (B = 64), OpenMP, 16 threads | 0,1380 s | ≈ **9,0×** ² | 96,2 % |
| **CUDA Inicial** | [`v3.1-cuda-inicial`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v3.1-cuda-inicial) | mini-batch na V100, 12 kernels por lote, sem otimização | 0,1494 s | ≈ **8,3×** ² | 96,2 % (saída idêntica à da CPU) |
| **CUDA Otimizado** | — | em andamento | — | — | — |

¹ Medido no mesmo job, execuções intercaladas contra o baseline (comparação da Issue #11). ² Razão entre tempos medidos em **jobs diferentes** (baseline 1,2418 s); é uma indicação, não uma medida intercalada. A partir do mini-batch (v2.2) o algoritmo muda e o critério de correção passa a ser a acurácia de teste, que não pode ser pior que a do baseline (95,5 %).

## Índice de Issues, Tags e resultados

| Bloco | Mudança | Resultado (SD, salvo indicação) | Issue | PR | Tag |
|:--|:--|:--|:-:|:-:|:--|
| v0.1 | baseline legado, medido e perfilado | 1,2418 s; `Backward` ≈ 75 % das instruções | — | — | [`v0.1-baseline`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v0.1-baseline) |
| v1.1 | vetores por `const&` em `Forward`/`Backward` | −26,5 % de `malloc`; tempo ≈ ±2 % (efeito de alinhamento de código; PC +1,6 %, SD +0,8 % com alinhamento fixo) | [#1](https://github.com/SilasAPB/APS-Final-SuperComp/issues/1) | [#2](https://github.com/SilasAPB/APS-Final-SuperComp/pull/2) | [`v1.1-vetores-por-referencia`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.1-vetores-por-referencia) |
| v1.2 | inversão da ordem dos loops do `grad_in` | **1,157×** (quebra a cadeia de dependência; IPC 2,67 → 2,91) | [#3](https://github.com/SilasAPB/APS-Final-SuperComp/issues/3) | [#4](https://github.com/SilasAPB/APS-Final-SuperComp/pull/4) | [`v1.2-inversao-loops`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.2-inversao-loops) |
| v1.3 | `weights` em bloco contíguo | **1,124×** no SD; 0,863× no PC (depende da microarquitetura) | [#5](https://github.com/SilasAPB/APS-Final-SuperComp/issues/5) | [#6](https://github.com/SilasAPB/APS-Final-SuperComp/pull/6) | [`v1.3-weights-bloco-continuo`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.3-weights-bloco-continuo) |
| v1.4 | esparsidade do ReLU (pula `delta == 0`) | **1,381×**; −44 % das instruções | [#7](https://github.com/SilasAPB/APS-Final-SuperComp/issues/7) | [#8](https://github.com/SilasAPB/APS-Final-SuperComp/pull/8) | [`v1.4-esparsidade-relu`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.4-esparsidade-relu) |
| v1.5 | não calcula o `grad_in` da primeira camada (seu retorno era descartado) | **1,146×** | [#9](https://github.com/SilasAPB/APS-Final-SuperComp/issues/9) | [#10](https://github.com/SilasAPB/APS-Final-SuperComp/pull/10) | [`v1.5-grad-in-primeira-camada`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.5-grad-in-primeira-camada) |
| v1.6 | `Forward` com 4 neurônios por vez (unroll-and-jam) | **1,219×**; acumulado v0.1 → v1.6: **2,468×** | [#11](https://github.com/SilasAPB/APS-Final-SuperComp/issues/11) | [#12](https://github.com/SilasAPB/APS-Final-SuperComp/pull/12), [#13](https://github.com/SilasAPB/APS-Final-SuperComp/pull/13) | [`v1.6-forward-unroll-jam`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v1.6-forward-unroll-jam) |
| v2.1 | OpenMP: neurônios de cada camada entre threads | **hipótese refutada**: eficiência 28 % (2 threads) → 0,8 % (16); sem tag | [#14](https://github.com/SilasAPB/APS-Final-SuperComp/issues/14) | [#15](https://github.com/SilasAPB/APS-Final-SuperComp/pull/15) | — |
| v2.2 | mini-batch sequencial (B = 64, taxa 0,3) | acurácia 96,2 % (base: 95,5 %); tempo ≈ v1.6 (1,038×) | [#16](https://github.com/SilasAPB/APS-Final-SuperComp/issues/16) | [#17](https://github.com/SilasAPB/APS-Final-SuperComp/pull/17) | [`v2.2-minibatch-sequencial`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v2.2-minibatch-sequencial) |
| v2.3 | mini-batch paralelo com OpenMP | **3,4×** (8 threads), **3,6×** (16) sobre a v2.2; satura por custo fixo por lote | [#18](https://github.com/SilasAPB/APS-Final-SuperComp/issues/18) | [#19](https://github.com/SilasAPB/APS-Final-SuperComp/pull/19) | [`v2.3-minibatch-paralelo`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v2.3-minibatch-paralelo) |
| v3.1 | porte funcional para CUDA | 0,149 s; saída idêntica à da CPU; kernels minúsculos (36.720) dominam o tempo | [#20](https://github.com/SilasAPB/APS-Final-SuperComp/issues/20) | [#21](https://github.com/SilasAPB/APS-Final-SuperComp/pull/21) | [`v3.1-cuda-inicial`](https://github.com/SilasAPB/APS-Final-SuperComp/tree/v3.1-cuda-inicial) |

## Principais achados

- **A CPU sequencial ganhou 2,47×** com seis mudanças pequenas e independentes, cada uma com profiling, hipótese e medição. O maior ganho individual veio de **eliminar trabalho** (esparsidade do ReLU, 1,38×); dois outros vieram de **quebrar cadeias de dependência** (inversão de loops, `Forward` com 4 neurônios), que o contador de instruções subestima: o `perf stat` (ciclos e IPC) mostrou o efeito.
- **Efeitos de alguns por cento não são confiáveis sem controle:** na v1.1, a diferença de tempo vinha do posicionamento do código (visível em `branch-misses`), e um experimento de controle com alinhamento fixo mostrou isso.
- **O mesmo código tem efeito diferente por microarquitetura:** a v1.3 deu 1,124× no Xeon e 0,863× no i5.
- **Paralelismo por neurônio dentro de cada amostra não compensa** (v2.1): cada etapa dura ≈ 0,3 µs e uma barreira custa ≥ 0,4 µs. Foi preciso **mudar o algoritmo** (mini-batch, v2.2) para criar paralelismo entre amostras (v2.3), com a acurácia de teste como critério.
- **A GPU, na versão funcional, empata com a melhor CPU:** o treino tem pouco trabalho por lote, e 36.720 kernels de ≈ 3,5 µs dominam o tempo. As transferências somam 0,2 ms.
- **Limitação:** o Nsight Compute não está disponível no SD (`ERR_NVGPUCTRPERM`), então faltam ocupação real e métricas de memória por kernel; o Nsight Systems funciona.

## Metodologia

- **Ambientes:** SD (nó exclusivo na fila `sequana_cpu_dev`; fila `sequana_gpu_dev` para CUDA), Xeon Gold 6252 @ 2,10 GHz (2 × 24 núcleos, sem hyperthreading) e Tesla V100-SXM2-32GB (`sm_70`); PC de desenvolvimento com i5-13450HX e RTX 3050. Compiladores: g++ 15.2.0 (`-O2 -std=c++17`) e `nvcc` 12.6.77 (`--fmad=false`). Ferramentas: `valgrind` (memcheck e callgrind), `perf stat`, `compute-sanitizer`, Nsight Systems.
- **Medição:** o tempo é o do laço de treino. Comparações entre versões usam **execuções intercaladas (A, B, A, B, …) no mesmo job**, 20 pares, com mediana e desvio, para que oscilações de frequência e carga afetem as duas versões igualmente. Cada Issue registra ambiente, flags, `md5` do fonte e commit.
- **Corretude:** nas versões sequenciais, a saída (`saida.txt`) é idêntica à do baseline, bit a bit. A partir do mini-batch, o critério é **acurácia de teste ≥ 95,5 %**, com a taxa escolhida só com um conjunto de validação (1/5 do treino), e resultado determinístico.
- **Reprodutibilidade:** cada tag compila e reproduz a medição do seu bloco; as versões concluídas ficam em [`legacy/`](legacy/) para comparação, e os dados em [`profiling/<tag>/<ambiente>/`](profiling/).

## Como reproduzir

```bash
git checkout <tag>
scripts/profile_cpu.sh barenn.cpp <rotulo> 10                 # tempo, memcheck, callgrind, perf stat
sbatch slurm/compare_cpu.sbatch <rotuloA> <rotuloB> 20 "1 2 4 8 16"   # comparação intercalada no SD
scripts/calibrar_minibatch.sh <rotulo> 64 0.1 0.3 0.5         # taxa do mini-batch, só com validação
sbatch slurm/profile_gpu.sbatch barenn.cu <rotulo> 10         # CUDA: tempo, compute-sanitizer, nsys, ncu
```

## Estrutura do repositório

| Caminho | Conteúdo |
|:--|:--|
| `barenn.cpp`, `barenn.cu` | versão atual da CPU (OpenMP) e da GPU (CUDA) |
| `legacy/<tag>/` | cópia congelada de cada versão concluída |
| `profiling/<tag>/<ambiente>/` | evidências de cada bloco: tempos, `callgrind`, `perf stat`, `nsys`, comparações |
| `scripts/`, `slurm/` | medição, comparação e calibração (CPU e GPU) |
| `data/` | conjunto `optdigits` |
| [`enunciado.md`](enunciado.md) | enunciado e rubrica da atividade |

## Estado atual

Concluídos: CPU sequencial otimizada (v1.1 a v1.6), CPU paralela (v2.1 a v2.3) e CUDA inicial (v3.1). Em andamento: CUDA otimizado (v3.2, com a Issue já desenhada); `gradiente_pesos` responde por 45 % do tempo dos kernels da v3.1.

## To-dos (GPU): o que não deu tempo de implementar

O trabalho pendente está todo na GPU. Nenhum item abaixo está implementado.

### v3.2: fusão de kernels (próximo bloco, Issue já desenhada)

- [ ] **Reduzir de 12 para 2 kernels por lote.** Na v3.1, cada lote lança 12 kernels minúsculos (36.720 no total, média de 3,5 µs, soma de 127,7 ms ≈ 86 % do treino), e cada um paga o piso de ≈ 1,2 µs mais a ida e a volta dos intermediários pela memória global.
  - **Kernel 1, um bloco por amostra:** forward das três camadas, perda, gradiente dos escores e deltas das camadas 2 e 1, com somas, ativações e deltas em **memória compartilhada** (`__syncthreads()` entre as camadas); só as ativações, os deltas e a perda por amostra vão para a memória global.
  - **Kernel 2, uma thread por parâmetro de todas as camadas (2.778):** soma as amostras em ordem e já atualiza o parâmetro, o que é seguro porque o gradiente não usa pesos.
- [ ] **Previsão:** ≈ 19 µs por lote (contra ≈ 42 µs de kernels hoje), ou ≈ **2,5×** (faixa de 2× a 3×); refutada se o tempo não cair. É uma estimativa feita a partir dos tempos da v3.1, e a cadeia de sincronizações dentro do bloco pode deixar o primeiro kernel mais lento que o previsto.
- [ ] **Critério:** saída idêntica à da v3.1 e da CPU (cada valor continua sendo calculado por uma thread, na mesma ordem, com `--fmad=false`), determinismo, `compute-sanitizer` sem erros, e Nsight Systems com a contagem de kernels (36.720 → 6.120) e a soma dos tempos.
- [ ] **Fora do escopo desse bloco, de propósito:** CUDA Graphs, streams, tamanho de bloco e outra forma de somar as amostras, para atribuir o ganho a uma causa só.

### Depois do v3.2

- [ ] **Somar as amostras em paralelo no gradiente:** hoje cada thread percorre as 64 amostras em série (`gradiente_pesos`, 45 % do tempo dos kernels da v3.1); uma redução por bloco ou por warp encurtaria essa cadeia, mudando a ordem das somas (o critério seria a acurácia de teste).
- [ ] **CUDA Graphs e streams:** esconder o custo de lançamento (3,3 µs por kernel) e sobrepor trabalho; ajuste do tamanho de bloco.
- [ ] **Métricas do Nsight Compute:** ocupação real e acesso à memória por kernel. No SD os contadores estão bloqueados (`ERR_NVGPUCTRPERM`); alternativas: `sudo ncu` na RTX 3050 local (sinalizando que não é a V100), ocupação teórica via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, ou pedir a liberação ao LNCC.
- [ ] **Comparar a GPU com a CPU em nó exclusivo:** a CPU com 16 threads mediu 0,198 s no nó de GPU e 0,138 s no nó exclusivo, então a comparação justa precisa do mesmo tipo de nó.
