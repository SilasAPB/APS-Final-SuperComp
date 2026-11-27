#!/bin/bash -l
#SBATCH --job-name=barenn-cmp
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm/outputs/%x-%j.out

# Compara dois binarios JA compilados por profile_cpu.sh, com execucoes intercaladas.
# Uso: sbatch slurm/compare_cpu.sbatch <rotuloA> <rotuloB> [reps]
#   ex.: sbatch slurm/compare_cpu.sbatch v0.1-baseline v1.1-vetores-por-referencia 20
module load gcc/15.2.0_sequana

cd "$SLURM_SUBMIT_DIR"
A="$1"; B="$2"; REPS="${3:-20}"
scripts/compare_cpu.sh "profiling/$A/sdumont/barenn" "profiling/$B/sdumont/barenn" "$REPS" \
  "profiling/$B/sdumont/comparacao_$A.txt"
