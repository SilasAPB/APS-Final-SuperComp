#!/usr/bin/env bash
# Escolhe a taxa do mini-batch usando so uma parte do treino como validacao (1/5 das
# amostras, indices multiplos de 5); o conjunto de teste nao e usado.
# Uso: scripts/calibrar_minibatch.sh [ROTULO] [LOTE] [TAXAS...]
#   ex.: scripts/calibrar_minibatch.sh v2.2-minibatch-sequencial 64 0.1 0.2 0.3 0.5 0.8 1.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LABEL="${1:-v2.2-minibatch-sequencial}"; LOTE="${2:-64}"; shift 2 2>/dev/null || shift $# || true
TAXAS="${*:-0.1 0.2 0.3 0.5 0.8 1.0 1.5}"
OUT="profiling/$LABEL"; mkdir -p "$OUT"
BIN="$(mktemp)"; trap 'rm -f "$BIN"' EXIT
g++ -O2 -std=c++17 barenn.cpp -o "$BIN"

{
  echo "# calibracao em validacao (4/5 do treino para treinar, 1/5 para validar); lote=$LOTE; 51 epocas"
  echo "# md5 fonte: $(md5sum < barenn.cpp | cut -c1-12)"
  echo "taxa acuracia_val(%) acertos perda_val"
  for t in $TAXAS; do
    saida="$(BARENN_VALIDACAO=1 BARENN_LOTE=$LOTE BARENN_TAXA=$t "$BIN")"
    acc=$(echo "$saida" | awk '/^Acuracia/ {gsub("%","",$2); print $2}')
    hits=$(echo "$saida" | awk '/^Acuracia/ {print $4}')
    perda=$(echo "$saida" | awk '/^Perda media/ {print $3}')
    echo "$t $acc $hits $perda"
  done
} | tee "$OUT/calibracao.txt"

melhor=$(grep -v '^#' "$OUT/calibracao.txt" | tail -n +2 | sort -k2,2nr -k4,4n | head -1)
echo "melhor (maior acuracia de validacao; empate: menor perda): $melhor" | tee -a "$OUT/calibracao.txt"
