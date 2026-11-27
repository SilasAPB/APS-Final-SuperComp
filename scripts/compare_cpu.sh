#!/usr/bin/env bash
# Compara dois binarios com execucoes INTERCALADAS (A,B,A,B...), para que
# oscilacoes de frequencia/carga afetem os dois igualmente.
# Uso: [PIN=<cpu>] scripts/compare_cpu.sh BIN_A BIN_B [REPS] [ARQ_SAIDA]
#   PIN=4  fixa a execucao no CPU 4 (taskset); sem PIN, nao fixa (ex.: no exclusivo do SD)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A="$1"; B="$2"; REPS="${3:-20}"; OUTF="${4:-/dev/stdout}"
for b in "$A" "$B"; do [ -x "$b" ] || { echo "ERRO: binario nao encontrado ou sem permissao de execucao: $b" >&2; exit 1; }; done
RUN=(); [ -n "${PIN:-}" ] && RUN=(taskset -c "$PIN")

TMP="$(mktemp)"
for _ in $(seq "$REPS"); do
  for x in A B; do
    bin="${!x}"
    echo "$x $("${RUN[@]}" "$bin" | awk '/^Tempo de treino/ {print $4}')" >> "$TMP"
  done
done

{
  echo "A=$A"; echo "B=$B"; echo "reps=$REPS pin=${PIN:-nenhum} data=$(date -Is) no=$(hostname -s)"
  python3 -I - "$TMP" <<'PY'
import sys, statistics as st
d = {"A": [], "B": []}
for l in open(sys.argv[1]):
    k, v = l.split(); d[k].append(float(v))
for k in "AB":
    v = d[k]
    print(f"{k}: media={st.mean(v):.4f} desvio={st.pstdev(v):.4f} mediana={st.median(v):.4f} min={min(v):.4f} max={max(v):.4f} n={len(v)}")
print(f"speedup (mediana A/B) = {st.median(d['A'])/st.median(d['B']):.3f}x   (minimo A/B) = {min(d['A'])/min(d['B']):.3f}x")
PY
} > "$OUTF"
rm -f "$TMP"
[ "$OUTF" = /dev/stdout ] || cat "$OUTF"
