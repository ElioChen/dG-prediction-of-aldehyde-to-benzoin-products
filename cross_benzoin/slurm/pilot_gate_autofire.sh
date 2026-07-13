#!/bin/bash
# Overnight gate + auto-fire (2026-06-23). Waits for the cb pilot to drain, runs strict
# acceptance gates, and ONLY if ALL pass submits the full 220k array 0-2207%128.
# Any failed gate => DO NOT fire; leave a clear report. Authorized by user (asleep).
set -uo pipefail
REPO="/scratch-shared/schen3/benzoin-dg"
PILOT_JOB=24127523
PILOT_OUT="$REPO/data/cross_benzoin/homo_v6_pilot"
FULL_IN="$REPO/data/cross_benzoin/homo_v6/homo_pairs.csv"
FULL_OUT="$REPO/data/cross_benzoin/homo_v6"
PY=/home/schen3/venv/nhc-workflow/bin/python
LOG="$REPO/cross_benzoin/slurm/AUTOFIRE_STATUS.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== autofire armed: waiting for pilot $PILOT_JOB to drain ==="
# wait up to ~4h (480 * 30s)
for i in $(seq 1 480); do
  r=$(squeue -j "$PILOT_JOB" -h 2>/dev/null | wc -l)
  [ "$r" -eq 0 ] && break
  sleep 30
done
say "pilot drained (remaining=$r)"

FAIL=0
# Gate 1: all pilot tasks COMPLETED
STATES=$(sacct -j "$PILOT_JOB" -X -n --format=State 2>/dev/null | grep -vE '^\s*$' | sort -u | tr '\n' ',')
say "gate1 states: $STATES"
echo "$STATES" | grep -qiE 'FAILED|TIMEOUT|CANCELLED|OUT_OF' && { say "GATE1 FAIL: bad state"; FAIL=1; }
echo "$STATES" | grep -qi 'COMPLETED' || { say "GATE1 FAIL: none completed"; FAIL=1; }

# Gate 2: zero quota errors
QERR=$(grep -h 'Errno 122\|Disk quota' "$PILOT_OUT"/logs/cb_*.out 2>/dev/null | wc -l)
say "gate2 quota errors: $QERR"
[ "$QERR" -ne 0 ] && { say "GATE2 FAIL: quota errors"; FAIL=1; }

# Gate 3: max chunk elapsed < 4h (sanity vs 12h walltime)
MAXSEC=$(sacct -j "$PILOT_JOB" -X -n --format=ElapsedRaw 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)
say "gate3 max elapsed sec: ${MAXSEC:-NA}"
[ -n "${MAXSEC:-}" ] && [ "$MAXSEC" -ge 14400 ] && { say "GATE3 FAIL: a chunk >=4h"; FAIL=1; }

# Gate 4+5: dual-baseline coverage + SMILES validity (python)
$PY - "$PILOT_OUT" "$REPO" <<'PYEOF'
import sys, glob, pandas as pd
pout, repo = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{pout}/chunk_*/products.csv"))
if not fs:
    print("PYGATE FAIL: no products.csv"); sys.exit(3)
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
n = len(df)
fx = df['dG_xtb_kcal'].notna().mean() if 'dG_xtb_kcal' in df else 0
fg = df['dG_gxtb_kcal'].notna().mean() if 'dG_gxtb_kcal' in df else 0
gg = df['G_gxtb'].notna().mean() if 'G_gxtb' in df else 0
print(f"PYGATE rows={n} dG_xtb_frac={fx:.3f} dG_gxtb_frac={fg:.3f} G_gxtb_frac={gg:.3f}")
ok = (n >= 250) and (fx >= 0.95) and (fg >= 0.95) and (gg >= 0.95)
# SMILES validity (homo benzoin = no atom loss => product heavy atoms == 2x aldehyde;
# AND product must contain NO free aldehyde R-CHO — a leftover formyl = benzoin-generator bug).
try:
    from rdkit import Chem
    ALD = Chem.MolFromSmarts('[CX3H1]=O')   # free aldehyde
    bad_atoms = bad_formyl = checked = 0
    for _, r in df.iterrows():
        a, b = r.get('donor_smiles'), r.get('smiles')
        ma = Chem.MolFromSmiles(str(a)) if pd.notna(a) else None
        mb = Chem.MolFromSmiles(str(b)) if pd.notna(b) else None
        if ma is None or mb is None:
            continue
        checked += 1
        if mb.GetNumAtoms() != 2*ma.GetNumAtoms():
            bad_atoms += 1
        if mb.HasSubstructMatch(ALD):       # product still has a free CHO -> wrong product
            bad_formyl += 1
    fa_bad = bad_atoms/checked if checked else 1.0
    ff_bad = bad_formyl/checked if checked else 1.0
    print(f"PYGATE smiles checked={checked} bad_atomcount={bad_atoms}({fa_bad:.4f}) "
          f"free_aldehyde_in_product={bad_formyl}({ff_bad:.4f})")
    ok = ok and (fa_bad < 0.02) and (ff_bad < 0.02)
except Exception as e:
    print(f"PYGATE smiles check error: {e}")
    ok = False
sys.exit(0 if ok else 4)
PYEOF
PYRC=$?
say "gate4/5 python rc=$PYRC (0=pass)"
[ "$PYRC" -ne 0 ] && { say "GATE4/5 FAIL"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  say "=== HOLD: one or more gates FAILED — NOT firing the full run. Review pilot. ==="
  exit 1
fi

# All gates passed -> fire the full 220k array
N=$(($(wc -l < "$FULL_IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK ))
say "=== ALL GATES PASS — submitting full array 0-$((NCH-1))%128 (N=$N pairs) ==="
mkdir -p "$FULL_OUT/logs"
JID=$(sbatch --parsable --array=0-$((NCH-1))%128 --output="$FULL_OUT/logs/cb_%A_%a.out" \
  --export=ALL,INPUT="$FULL_IN",OUTDIR="$FULL_OUT",CHUNK=$CHUNK,MULTIWFN=0,EMIT_ALD=1,CONFORMER=funnel_v3,N_CONFS=10 \
  "$REPO/cross_benzoin/slurm/submit_cb_featurize_array.sh" 2>>"$LOG")
say "=== FIRED full library: SLURM job $JID ==="
# ensure the orphan-cleanup cron chain is alive for the multi-day run
squeue -u "$USER" --name=orphan_cron -h 2>/dev/null | grep -q . || {
  sbatch "$REPO/pipeline/slurm/orphan_cleanup_cron.sh" >>"$LOG" 2>&1; say "restarted orphan_cron chain"; }
say "DONE."
