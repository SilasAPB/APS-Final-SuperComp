# Diagnóstico de alinhamento (registro)

**Não é uma versão oficial e não entra nas tabelas de comparação.** Fica guardado apenas como registro do experimento de controle da Issue da v1.1.

**Pergunta:** a v1.1 ficou 1,7 % mais lenta no SD com as flags oficiais (`-O2`). O `perf stat` mostrou o dobro de `branch-misses` (11,1 M contra 5,5 M). A causa é o `const&` ou o posicionamento do código no binário?

**Experimento:** compilar as duas fontes (`legacy/v0.1-baseline/barenn.cpp` e `barenn.cpp` da v1.1) com as **mesmas** flags de alinhamento e medir de novo.

```bash
g++ -O2 -std=c++17 -falign-loops=32 -falign-functions=64 <fonte> -o <bin>
perf stat -r 5 -e cycles,instructions,branch-misses,L1-dcache-load-misses <bin>
```

**Resultado:** as `branch-misses` das duas versões ficam no mesmo patamar (≈ 5,2–5,4 M) e a v1.1 passa a ser 0,7 % mais rápida. A lentidão com `-O2` era artefato de posicionamento do código. Detalhes no item 5 da análise em `resultados.md` da Issue.

**Por que está aqui e não em uma pasta própria:** pertence à v1.1 (é evidência da conclusão dela). **Por que o job Slurm foi apagado:** foi um experimento de uso único. O comando acima, mais o `ambiente.txt` desta pasta (nó, flags e md5 das fontes), reproduzem tudo, e manter um `.sbatch` a mais só aumentaria a superfície do repositório.

Arquivos: `ambiente.txt`, `perf_stat_baseline.txt`, `perf_stat_v1.1.txt`. Os binários não são versionados.
