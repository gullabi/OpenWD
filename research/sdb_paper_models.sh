#!/bin/bash
# Models for the sdB paper figure, at fixed published parameters (nothing fitted).
#   research/sdb_paper_models.sh [OUTPUT_DIR] [TARGET ...]   (default results/sdb/paper, all four stars)
# One model at a time, single-threaded: metal-blanketed LTE structure (sdb_lte_hhe.py
# --abundance, every metal below) -> hybrid NLTE H/He (MALI, H16, dominant He
# conservation row) -> metals (NLTE C N O Si S; other measured elements as LTE
# line opacity; upper limits omitted).  Each stage records the git commit.
#
# HD 4539, Feige 38: Teff, log g, He/H from Schneider et al. (2018, hybrid
#   LTE/NLTE); metals from Geier (2013), log N/N(H), mean of the ions measured.
# LS IV-14 116: 35500 K, 5.85, -0.60 (Dorsch et al. 2020 refit of FORS2);
# Feige 46: 36100 K, 5.93, -0.32 (Latour et al. 2019b); metals from Dorsch et
#   al. (2020) Tables 6-7, log N/N(H).
set -u
OUT=${1:-results/sdb/paper}
shift $(( $# > 0 ? 1 : 0 ))
TARGETS=${*:-lsiv14116 feige46 hd4539 feige38}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src:research
D=src/wd_spectra/data/runtime
COMMIT=$(git rev-parse HEAD)$(git diff --quiet HEAD -- src research || echo -dirty)

star() {  # TAG TEFF LOGG LOG_HE_H "NLTE X=V ..." "LTE X=V ..." ["EXTRA metal-stage options"]
  # The structure gets the NLTE and LTE elements (heavy elements beyond Zn are not in Stout).
  local tag=$1 teff=$2 logg=$3 he=$4 nlte=$5 lte=$6 extra=${7:-} structure=() trace=() item
  [[ " $TARGETS " == *" $tag "* ]] || return 0
  for item in $nlte $lte; do structure+=(--abundance "$item"); done
  for item in $nlte; do trace+=(--abundance "$item"); done
  for item in $lte; do trace+=(--lte-abundance "$item"); done
  for item in $extra; do trace+=("$item"); done
  echo "=== $tag start $(date +%H:%M:%S) commit $COMMIT"
  [ -f "$OUT/$tag-lte/run-summary.json" ] || { rm -rf "$OUT/$tag-lte"; python research/sdb_lte_hhe.py --teff "$teff" \
    --logg "$logg" --log-he-h "$he" "${structure[@]}" --data-root $D --output "$OUT/$tag-lte" > "$OUT/$tag-lte.log" 2>&1
    echo "=== $tag lte exit $? $(date +%H:%M:%S)"; }
  [ -f "$OUT/$tag-hybrid/run-summary.json" ] || { rm -rf "$OUT/$tag-hybrid"; python research/sdb_hybrid_nlte.py \
    "$OUT/$tag-lte" --data-root $D --hydrogen-levels 16 --accelerated-lambda --population-iterations 200 200 200 \
    --output "$OUT/$tag-hybrid" > "$OUT/$tag-hybrid.log" 2>&1; echo "=== $tag hybrid exit $? $(date +%H:%M:%S)"; }
  [ -f "$OUT/$tag-metals/run-summary.json" ] || { rm -rf "$OUT/$tag-metals"; python research/sdb_trace_metals.py \
    "$OUT/$tag-lte" "$OUT/$tag-hybrid" --data-root $D "${trace[@]}" --output "$OUT/$tag-metals" \
    > "$OUT/$tag-metals.log" 2>&1; echo "=== $tag metals exit $? $(date +%H:%M:%S)"; }
  echo "$COMMIT" > "$OUT/$tag-commit.txt"
}

# LS IV-14 116: every element Dorsch et al. (2020) measured that has data (upper limits omitted); N III with
# 72 levels (N III 4200/4379); Ge Sr Y Zr Sn and the N II 3d-4f lines as heavy LTE lines (sdb_heavy_lines.py).
star lsiv14116 35500 5.85 -0.60 "C=-3.70 N=-3.82 O=-4.48 Si=-6.03" \
  "Ne=-4.50 Mg=-5.40 P=-6.76 Ar=-5.55 Ti=-5.69 Ni=-4.62 Zn=-4.92" \
  "--heavy-abundance Ge=-5.05 --heavy-abundance Sr=-4.44 --heavy-abundance Y=-5.13 --heavy-abundance Zr=-4.76
   --heavy-abundance Sn=-5.56 --heavy-abundance N=-3.82 --levels N:2=72"
star feige46 36100 5.93 -0.32 "C=-3.19 N=-3.57 O=-4.21 Si=-5.51" "Ne=-4.31 Mg=-5.05 Ar=-5.75 Fe=-4.64 Ni=-4.53"
star hd4539 23200 5.20 -2.27 "C=-4.06 N=-4.10 O=-4.97 Si=-5.30 S=-5.17" "Mg=-5.40 Al=-6.40 Ar=-4.83 Fe=-4.62" \
  "--heavy-abundance N=-4.10"
star feige38 28200 5.61 -2.70 "C=-4.23 N=-3.98 O=-4.35 Si=-4.52 S=-4.87" "Mg=-4.90 Al=-6.20 Ar=-4.90 Fe=-4.84"
# Not in the default targets: HD 4539 at 25,200 K, the Teff that fits its Si II/Si III balance in OpenWD
# (sdB guide, "Silicon ionization"); otherwise identical to hd4539.
star hd4539-hot 25200 5.20 -2.27 "C=-4.06 N=-4.10 O=-4.97 Si=-5.30 S=-5.17" "Mg=-5.40 Al=-6.40 Ar=-4.83 Fe=-4.62" \
  "--heavy-abundance N=-4.10"
echo "=== paper models done $(date +%H:%M:%S)"
