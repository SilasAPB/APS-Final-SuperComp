#!/bin/bash
#SBATCH --job-name=barenn-cpu
#SBATCH --partition=sequana_cpu_dev        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24         # AJUSTAR ao no escolhido
#SBATCH --exclusive                # evita ruido de outros jobs nas medidas
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR"            # submeter sempre da raiz do repo
scripts/profile_cpu.sh "${1:-legacy/v0.1-baseline/barenn.cpp}" "${2:-v0.1-baseline}" "${3:-5}"
