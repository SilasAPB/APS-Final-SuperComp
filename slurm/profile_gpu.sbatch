#!/bin/bash -l
#SBATCH --job-name=barenn-gpu
#SBATCH --partition=sequana_gpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00            # limite da fila sequana_gpu_dev
#SBATCH --output=slurm/outputs/%x-%j.out

# Mede a versao CUDA (scripts/profile_gpu.sh) e, se pedido, compara com uma versao de CPU
# no mesmo job (execucoes intercaladas; a CPU usa OMP_NUM_THREADS = threads).
# Uso: sbatch slurm/profile_gpu.sbatch <fonte> <rotulo> [reps] [rotulo_cpu] [threads_cpu]
#   ex.: sbatch slurm/profile_gpu.sbatch barenn.cu v3.1-cuda-inicial 10 v2.3-minibatch-paralelo 16
module load cuda/12.6_sequana   # o nvcc usa o g++ padrao do no (o CUDA 12.6 nao aceita o gcc 15)

cd "$SLURM_SUBMIT_DIR"
FONTE="$1"; ROT="$2"; REPS="${3:-10}"; CPU="${4:-}"; TCPU="${5:-1}"
scripts/profile_gpu.sh "$FONTE" "$ROT" "$REPS"

if [ -n "$CPU" ]; then
  module load gcc/15.2.0_sequana   # libgomp/libstdc++ do binario de CPU (carregado so depois do nvcc)
  export OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active
  OMP_NUM_THREADS=$TCPU scripts/compare_cpu.sh "profiling/$CPU/sdumont/barenn" "profiling/$ROT/sdumont/barenn_cuda" 20 \
    "profiling/$ROT/sdumont/comparacao_${CPU}_T${TCPU}.txt"
  if diff -q <(OMP_NUM_THREADS=1 "profiling/$CPU/sdumont/barenn" | grep -v '^Tempo de treino') "profiling/$ROT/sdumont/saida.txt" > /dev/null; then
    echo "saida da GPU IDENTICA a da CPU ($CPU)" | tee "profiling/$ROT/sdumont/identidade_${CPU}.txt"
  else
    echo "saida da GPU DIFERENTE da CPU ($CPU)" | tee "profiling/$ROT/sdumont/identidade_${CPU}.txt"
  fi
fi
