#!/bin/bash
#SBATCH --job-name=r89_assemble
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch1/shared/schen3/benzoin-dg-restored/slurm_logs/r89_assemble_%j.out
#
# Auto-runs after the aldehyde SP array (26354346) drains, so the r8/9 recovery
# advances even if the interactive session is gone. Submit with:
#   sbatch --dependency=afterany:26354346 \
#     cross_benzoin/slurm/submit_r89_assemble_after_ald_sp.sh
#
set -o pipefail
REPO=/gpfs/scratch1/shared/schen3/benzoin-dg-restored
PY=/home/schen3/venv/nhc-workflow/bin/python
cd "$REPO"
echo "r89_assemble node=${SLURMD_NODENAME:-?} $(date)"

ALD=data/raw/dft_sp_cross/r89_aldehyde_sp
n_done=$(python3 -c "import glob,csv;print(sum(1 for f in glob.glob('$ALD/chunk_*.csv') for r in csv.DictReader(open(f)) if r.get('E_orca_Eh')))" 2>/dev/null)
n_empty=$(python3 -c "import glob,csv;print(sum(1 for f in glob.glob('$ALD/chunk_*.csv') for r in csv.DictReader(open(f)) if not r.get('E_orca_Eh')))" 2>/dev/null)
echo "aldehyde SP: $n_done done, $n_empty empty rows"

# 1. assemble the three-species recovered DFT-SP labels
"$PY" cross_benzoin/assemble_r89_dft_sp.py --rounds 8 9
echo "--- assembled ---"
for r in 8 9; do
  f=data/raw/dft_sp_cross/cross_round${r}/cross_round${r}_dft_sp.csv
  [[ -s "$f" ]] && echo "  r$r: $(( $(wc -l < "$f") - 1 )) labelled pairs -> $f" || echo "  r$r: MISSING $f"
done

# 2. refresh home backups (idempotent; captures aldehyde SP + regen geometry)
bash cross_benzoin/slurm/submit_backup_recovery_artifacts.sh || echo "backup script exited $?"

# 3. auto-chain the r1-9 retrain (steps 1-4 + GNN) iff both label files landed
ok=1
for r in 8 9; do [[ -s "data/raw/dft_sp_cross/cross_round${r}/cross_round${r}_dft_sp.csv" ]] || ok=0; done
if [[ $ok -eq 1 ]]; then
  echo "labels present -> submitting r1-9 retrain chain"
  sbatch cross_benzoin/slurm/submit_r19_retrain_chain.sh
else
  echo "r8/9 dft_sp labels missing -> NOT chaining retrain (run cross_benzoin/run_r19_retrain_chain.sh by hand)"
fi

echo "Done $(date)"
