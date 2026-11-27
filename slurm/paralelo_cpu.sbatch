#!/bin/bash -l
#SBATCH --job-name=barenn-omp
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24         # AJUSTAR: precisa cobrir o maior T da lista
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm/outputs/%x-%j.out

# Curva tempo x numero de threads (OpenMP): compara o binario sequencial (A) com o
# binario paralelo (B) para cada T, com execucoes intercaladas. Os dois binarios ja
# devem ter sido compilados por profile_cpu.sh (a versao paralela com CXXFLAGS_EXTRA=-fopenmp).
# Uso: sbatch slurm/paralelo_cpu.sbatch <rotuloA> <rotuloB> [lista de T]
#   ex.: sbatch slurm/paralelo_cpu.sbatch v1.6-forward-unroll-jam v2.1-paralelismo-cpu "1 2 4 8 16"
module load gcc/15.2.0_sequana    # a libgomp vem do gcc

cd "$SLURM_SUBMIT_DIR"
A="$1"; B="$2"; LISTA="${3:-1 2 4 8 16}"
export OMP_PLACES=cores           # uma thread por nucleo fisico
export OMP_PROC_BIND=close        # threads proximas (mesmo socket, enquanto couber)
export OMP_WAIT_POLICY=active     # espera ativa nas barreiras

for T in $LISTA; do
  OMP_NUM_THREADS=$T scripts/compare_cpu.sh "profiling/$A/sdumont/barenn" "profiling/$B/sdumont/barenn" 20 \
    "profiling/$B/sdumont/comparacao_${A}_T${T}.txt"
done
