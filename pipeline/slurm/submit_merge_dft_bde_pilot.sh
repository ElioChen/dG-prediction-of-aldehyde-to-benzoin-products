#!/bin/bash
#SBATCH --job-name=merge_dft_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/runs/logs/merge_dft_bde_pilot_%j.out
#
# Combines the dft_bde_pilot_n100 shards and reports g-xTB-vs-DFT / ALFABET-vs-DFT
# arbitration stats at scale. Submit with --dependency=afterany on the shard array's job id.
set -euo pipefail

REPO="/scratch-shared/schen3/benzoin-dg"
ENV="/gpfs/scratch1/shared/schen3/envs/bde_lite"
mkdir -p "$REPO/runs/logs"

echo "merging DFT BDE pilot n100 shards $(date)"
$ENV/bin/python -u "$REPO/pipeline/bde/merge_dft_bde_pilot_shards.py" \
    --glob "$REPO/data/cross_benzoin/homo_v6/dft_bde_pilot_n100_shard*.csv" \
    --out "$REPO/data/cross_benzoin/homo_v6/dft_bde_pilot_n100.csv" \
    --report-out "$REPO/runs/logs/dft_bde_pilot_n100_report.json"
echo "done $(date) exit=$?"
