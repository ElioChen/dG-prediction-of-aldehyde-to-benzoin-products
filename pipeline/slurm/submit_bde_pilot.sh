#!/bin/bash
#SBATCH --job-name=bde_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/logs/bde_pilot_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
cd "$REPO/pipeline/compute"
echo "BDE pilot $(date)"
$PY -u calc_bde.py --which aldehydes --n 20 --xtb-bin "$XTB_BIN" \
    --out /scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/bde_pilot_aldehydes.csv \
    --work-dir /scratch-shared/schen3/benzoin-dg/tmp/bde_pilot_ald
$PY -u calc_bde.py --which products --n 20 --xtb-bin "$XTB_BIN" \
    --out /scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/bde_pilot_products.csv \
    --work-dir /scratch-shared/schen3/benzoin-dg/tmp/bde_pilot_prod
echo "Done $(date) exit=$?"
