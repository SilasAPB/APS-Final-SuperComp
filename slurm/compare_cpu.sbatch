#!/bin/bash -l
#SBATCH --job-name=barenn-cmp
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24         # precisa cobrir o maior T da lista
#SBATCH --exclusive
#SBATCH --time=00:20:00            # limite da fila sequana_cpu_dev
#SBATCH --output=slurm/outputs/%x-%j.out

# Compara duas versoes ja compiladas por profile_cpu.sh, com execucoes intercaladas, e confere a
# saida de B: identica a de A? deterministica (duas execucoes iguais)? qual a acuracia de teste?
# Para versoes OpenMP, a lista de T gera uma comparacao por numero de threads.
# Uso: sbatch slurm/compare_cpu.sbatch <rotuloA> <rotuloB> [reps] [lista de T] [sufixo]
#   ex.: sbatch slurm/compare_cpu.sbatch v1.5-grad-in-primeira-camada v1.6-forward-unroll-jam
#        sbatch slurm/compare_cpu.sbatch v2.2-minibatch-sequencial v2.3-minibatch-paralelo 20 "1 2 4 8 16"
#        BARENN_ATUALIZACAO_SEQUENCIAL=1 sbatch slurm/compare_cpu.sbatch v2.2-... v2.3-... 20 "1 2 4 8 16" _passo1
module load gcc/15.2.0_sequana    # a libgomp vem do gcc

cd "$SLURM_SUBMIT_DIR"
A="$1"; B="$2"; REPS="${3:-20}"; LISTA="${4:-1}"; ROT="${5:-}"   # ROT: sufixo opcional dos arquivos de saida
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active   # ignorados por binarios sequenciais

for T in $LISTA; do
  SUF=""; [ "$LISTA" != "1" ] && SUF="_T$T"
  OUT="profiling/$B/sdumont/comparacao_${A}${SUF}${ROT}.txt"
  OMP_NUM_THREADS=$T scripts/compare_cpu.sh "profiling/$A/sdumont/barenn" "profiling/$B/sdumont/barenn" "$REPS" "$OUT"
  s1=$(OMP_NUM_THREADS=$T "profiling/$B/sdumont/barenn" | grep -v '^Tempo de treino')
  s2=$(OMP_NUM_THREADS=$T "profiling/$B/sdumont/barenn" | grep -v '^Tempo de treino')
  [ "$s1" = "$(cat profiling/$A/sdumont/saida.txt)" ] && ig=SIM || ig=NAO
  [ "$s1" = "$s2" ] && det=SIM || det=NAO
  acc=$(echo "$s1" | awk '/^Acuracia/ {print $2" ("$4" acertos)"}')
  echo "T=$T identica_a_A=$ig deterministica=$det acuracia=$acc" | tee -a "profiling/$B/sdumont/identidade_${A}${ROT}.txt"
done
