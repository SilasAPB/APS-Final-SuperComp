#!/usr/bin/env bash
# Uso: scripts/profile_cpu.sh [FONTE] [ROTULO] [REPS]
# Saidas em profiling/<ROTULO>/<host>/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # o programa usa caminhos relativos (data/...)

SRC="${1:-legacy/v0.1-baseline/barenn.cpp}"
LABEL="${2:-v0.1-baseline}"
REPS="${3:-10}"
NODE="$(hostname -s)"
case "$NODE" in sdumont*) ENV_NAME="sdumont" ;; *) ENV_NAME="$NODE" ;; esac  # no do SD muda a cada job
OUT="profiling/$LABEL/$ENV_NAME"
mkdir -p "$OUT"

# gcc < 9 (ex.: padrao do SD) precisa de -lstdc++fs para std::filesystem
LIBS=""
[ "$(g++ -dumpversion | cut -d. -f1)" -lt 9 ] && LIBS="-lstdc++fs"

# ambiente
{ date -Is; echo "no: $NODE"; git describe --tags --always --dirty; g++ --version | head -1
  valgrind --version; echo "flags: -O2 -std=c++17 $LIBS"; echo "fonte: $SRC"
  lscpu | grep -E 'Model name|^CPU\(s\)|L1d|L2|L3'; } > "$OUT/ambiente.txt" 2>&1

# compilacao (-g so para dar nomes ao callgrind)
g++ -O2 -std=c++17    "$SRC" -o "$OUT/barenn"   $LIBS
g++ -O2 -std=c++17 -g "$SRC" -o "$OUT/barenn_g" $LIBS

# saida de referencia (para diff nas proximas versoes) + tempos; ultima linha = resumo
"$OUT/barenn" | grep -v '^Tempo de treino' > "$OUT/saida.txt"
for _ in $(seq "$REPS"); do "$OUT/barenn" | awk '/^Tempo de treino/ {print $4}'; done > "$OUT/tempos.txt"
awk '{s+=$1; ss+=$1*$1} END{m=s/NR; printf "media=%.4f desvio=%.4f n=%d\n", m, sqrt(ss/NR-m*m), NR}' \
  "$OUT/tempos.txt" >> "$OUT/tempos.txt"
cat "$OUT/tempos.txt"

# valgrind e callgrind
valgrind --leak-check=full "$OUT/barenn_g" > /dev/null 2> "$OUT/memcheck.txt" || true
valgrind --tool=callgrind --cache-sim=yes --callgrind-out-file="$OUT/callgrind.out" \
  "$OUT/barenn_g" > /dev/null 2> /dev/null
# legivel: so instrucoes (Ir) + codigo-fonte anotado linha a linha
callgrind_annotate --show=Ir "$OUT/callgrind.out" "$SRC" > "$OUT/callgrind_annotate.txt"
# completo: Ir + acessos e misses de cache (D1/LL)
callgrind_annotate "$OUT/callgrind.out" "$SRC" > "$OUT/callgrind_annotate_cache.txt"

tail -n 1 "$OUT/memcheck.txt"
cat "$OUT/callgrind_annotate.txt"
