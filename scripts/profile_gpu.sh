#!/usr/bin/env bash
# Compila a versao CUDA, mede o tempo (N repeticoes), roda o compute-sanitizer e perfila com
# Nsight Systems (nsys) e Nsight Compute (ncu).
# Uso: scripts/profile_gpu.sh [FONTE] [ROTULO] [REPS]
#   ex.: scripts/profile_gpu.sh barenn.cu v3.1-cuda-inicial 10
# Saidas em profiling/<ROTULO>/<ambiente>/ (mesma regra de nome de profile_cpu.sh)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SRC="${1:-barenn.cu}"
LABEL="${2:-v3.1-cuda-inicial}"
REPS="${3:-10}"
NODE="$(hostname -s)"
case "$NODE" in sdumont*) ENV_NAME="sdumont" ;; *) ENV_NAME="$NODE" ;; esac
OUT="profiling/$LABEL/$ENV_NAME"
mkdir -p "$OUT"
BIN="$OUT/barenn_cuda"

# arquitetura da GPU detectada (ex.: 7.0 -> sm_70); GPU_ARCH=sm_XX sobrescreve
ARCH="${GPU_ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
# gcc < 9 (ex.: padrao do SD) precisa de -lstdc++fs para std::filesystem
LIBS=""
[ "$(g++ -dumpversion | cut -d. -f1)" -lt 9 ] && LIBS="-lstdc++fs"
FLAGS="-O2 -std=c++17 --fmad=false -arch=$ARCH"

{ date -Is; echo "no: $NODE"
  echo "md5 fonte: $(md5sum < "$SRC" | cut -c1-12)"
  nvcc --version | tail -2; echo "flags: nvcc $FLAGS $LIBS"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  echo "nsys: $(nsys --version 2>&1 | head -1)"; echo "ncu: $(ncu --version 2>&1 | sed -n 3p)"
  lscpu | grep -E 'Model name|^CPU\(s\)'; } > "$OUT/ambiente.txt" 2>&1

nvcc $FLAGS "$SRC" -o "$BIN" $LIBS 2> "$OUT/compilacao.txt" || { cat "$OUT/compilacao.txt"; exit 1; }

# saida de referencia (sem a linha de tempo) e tempos; ultima linha = resumo
"$BIN" | grep -v '^Tempo de treino' > "$OUT/saida.txt"
for _ in $(seq "$REPS"); do "$BIN" | awk '/^Tempo de treino/ {print $4}'; done > "$OUT/tempos.txt"
awk '{s+=$1; ss+=$1*$1} END{m=s/NR; printf "media=%.4f desvio=%.4f n=%d\n", m, sqrt(ss/NR-m*m), NR}' \
  "$OUT/tempos.txt" >> "$OUT/tempos.txt"
cat "$OUT/tempos.txt"

# memcheck da GPU
compute-sanitizer --tool memcheck --leak-check full "$BIN" 2>&1 | grep -E "ERROR SUMMARY|LEAK SUMMARY" > "$OUT/memcheck.txt" || true
cat "$OUT/memcheck.txt"

# Nsight Systems: linha do tempo (kernels, chamadas da API, transferencias)
nsys profile -o "$OUT/nsys_report" --force-overwrite=true "$BIN" > /dev/null 2> "$OUT/nsys_stderr.txt" || true
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  "$OUT/nsys_report.nsys-rep" > "$OUT/nsys_stats.txt" 2>&1 || true
head -n 40 "$OUT/nsys_stats.txt" | cut -c1-150

# Nsight Compute: metricas de alguns kernels representativos (ocupacao, memoria, SOL)
ncu --section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis \
    -k regex:"forward|gradiente_pesos|delta_anterior|atualizar|perda_gradiente" --launch-count 10 \
    "$BIN" > "$OUT/ncu_detalhes.txt" 2>&1 || true
tail -n 3 "$OUT/ncu_detalhes.txt"
echo "arquivos em $OUT:"; ls -1 "$OUT"
