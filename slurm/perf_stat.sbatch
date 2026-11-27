#!/bin/bash -l
#SBATCH --job-name=barenn-perf
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm/outputs/%x-%j.out

# Contadores de hardware (ciclos, instrucoes, IPC) de dois binarios ja compilados por profile_cpu.sh.
# Uso: sbatch slurm/perf_stat.sbatch <rotuloA> <rotuloB> [reps]
module load gcc/15.2.0_sequana   # necessario: o binario usa a libstdc++ do gcc 15

cd "$SLURM_SUBMIT_DIR"
A="$1"; B="$2"; REPS="${3:-5}"
EV=cycles,instructions,branch-misses,L1-dcache-load-misses
for r in "$A" "$B"; do
  out="profiling/$r/sdumont/perf_stat.txt"
  { echo "no=$(hostname -s) reps=$REPS"; echo "bin=profiling/$r/sdumont/barenn"
    perf stat -r "$REPS" -e $EV "profiling/$r/sdumont/barenn" 2>&1 >/dev/null
  } | tee "$out"
done
