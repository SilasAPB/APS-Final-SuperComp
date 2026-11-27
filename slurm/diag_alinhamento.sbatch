#!/bin/bash -l
#SBATCH --job-name=barenn-diag
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm/outputs/%x-%j.out

# DIAGNOSTICO (nao e uma versao oficial): testa se a diferenca de branch-misses entre
# baseline e v1.1 vem do posicionamento do codigo. Compila as duas fontes com as
# MESMAS flags de alinhamento e mede com perf stat.
# Nao toca em profiling/<rotulo>/: tudo vai para profiling/diag-alinhamento/sdumont/.
module load gcc/15.2.0_sequana

cd "$SLURM_SUBMIT_DIR"
OUT=profiling/diag-alinhamento/sdumont
FLAGS="-O2 -std=c++17 -falign-loops=32 -falign-functions=64"
EV=cycles,instructions,branch-misses,L1-dcache-load-misses
mkdir -p "$OUT"

{ echo "no=$(hostname -s)  branch=$(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)"
  echo "g++=$(g++ --version | head -1)"; echo "flags=$FLAGS"
  echo "md5 baseline=$(md5sum < legacy/v0.1-baseline/barenn.cpp | cut -c1-12)  md5 v1.1=$(md5sum < barenn.cpp | cut -c1-12)"
} > "$OUT/ambiente.txt"

g++ $FLAGS legacy/v0.1-baseline/barenn.cpp -o "$OUT/barenn_baseline"
g++ $FLAGS barenn.cpp                      -o "$OUT/barenn_v1.1"

for v in baseline v1.1; do
  { echo "== $v ($FLAGS)"
    perf stat -r 5 -e $EV "$OUT/barenn_$v" 2>&1 >/dev/null
  } | tee "$OUT/perf_stat_$v.txt"
done
