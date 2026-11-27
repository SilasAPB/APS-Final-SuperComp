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

# flags extras opcionais (CXXFLAGS_EXTRA)
EXTRA="${CXXFLAGS_EXTRA:-}"
# fontes com OpenMP sao compiladas com -fopenmp automaticamente (sem ele rodariam sequenciais)
grep -q '#pragma omp' "$SRC" && EXTRA="$EXTRA -fopenmp"

# gcc < 9 (ex.: padrao do SD) precisa de -lstdc++fs para std::filesystem
LIBS=""
[ "$(g++ -dumpversion | cut -d. -f1)" -lt 9 ] && LIBS="-lstdc++fs"

# ambiente (git pode nao estar no PATH do job: nesse caso o md5 da fonte continua provando o que rodou)
if command -v git >/dev/null 2>&1; then
  GIT_INFO="branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)
$(git describe --tags --always --dirty)"
else
  GIT_INFO="git: indisponivel neste no"
fi
{ date -Is; echo "no: $NODE"; echo "$GIT_INFO"; echo "md5 fonte: $(md5sum < "$SRC" | cut -c1-12)"
  g++ --version | head -1; valgrind --version; echo "flags: -O2 -std=c++17 $EXTRA $LIBS"; (env | grep -E "^OMP_" || true) | sed "s/^/env: /"; echo "fonte: $SRC"
  lscpu | grep -E 'Model name|^CPU\(s\)|L1d|L2|L3'; } > "$OUT/ambiente.txt" 2>&1

# compilacao (-g so para dar nomes ao callgrind)
g++ -O2 -std=c++17    $EXTRA "$SRC" -o "$OUT/barenn"   $LIBS
g++ -O2 -std=c++17 -g $EXTRA "$SRC" -o "$OUT/barenn_g" $LIBS

# saida de referencia (para diff nas proximas versoes) + tempos; ultima linha = resumo
"$OUT/barenn" | grep -v '^Tempo de treino' > "$OUT/saida.txt"
for _ in $(seq "$REPS"); do "$OUT/barenn" | awk '/^Tempo de treino/ {print $4}'; done > "$OUT/tempos.txt"
awk '{s+=$1; ss+=$1*$1} END{m=s/NR; printf "media=%.4f desvio=%.4f n=%d\n", m, sqrt(ss/NR-m*m), NR}' \
  "$OUT/tempos.txt" >> "$OUT/tempos.txt"
cat "$OUT/tempos.txt"

# contadores de hardware (ciclos, IPC, branch-misses): so se houver perf no no
if command -v perf >/dev/null 2>&1; then
  { echo "no=$NODE reps=5"
    perf stat -r 5 -e cycles,instructions,branch-misses,L1-dcache-load-misses "$OUT/barenn" 2>&1 >/dev/null
  } > "$OUT/perf_stat.txt" || true
  cat "$OUT/perf_stat.txt"
fi

# valgrind e callgrind
OMP_NUM_THREADS=1 valgrind --leak-check=full "$OUT/barenn_g" > /dev/null 2> "$OUT/memcheck.txt" || true
OMP_NUM_THREADS=1 valgrind --tool=callgrind --cache-sim=yes --callgrind-out-file="$OUT/callgrind.out" \
  "$OUT/barenn_g" > /dev/null 2> /dev/null
# legivel: so instrucoes (Ir) + codigo-fonte anotado linha a linha
callgrind_annotate --show=Ir "$OUT/callgrind.out" "$SRC" > "$OUT/callgrind_annotate.txt"
# completo: Ir + acessos e misses de cache (D1/LL)
callgrind_annotate "$OUT/callgrind.out" "$SRC" > "$OUT/callgrind_annotate_cache.txt"

tail -n 1 "$OUT/memcheck.txt"
# no log do job so as 12 funcoes mais caras (o arquivo callgrind_annotate.txt continua completo)
sed -n '/file:function/,+13p' "$OUT/callgrind_annotate.txt" | cut -c1-130
