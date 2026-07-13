#!/bin/bash
#SBATCH --job-name=bdfe_pilot
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/logs/bdfe_pilot_%j.out
REPO="/scratch-shared/schen3/benzoin-dg"; PY="/gpfs/scratch1/shared/schen3/envs/gnn/bin/python"
XTB_BIN="/home/schen3/xtb/bin/xtb"; SHERMO_BIN="/home/schen3/.local/bin/Shermo"
source /etc/profile 2>/dev/null; module load 2023 2>/dev/null
export XTBPATH="/home/schen3/xtb/share/xtb"
cd "$REPO/pipeline/compute"
echo "BDFE pilot $(date)"
$PY -u calc_bde_free_energy.py --which aldehydes --n 10 --xtb-bin "$XTB_BIN" --shermo-bin "$SHERMO_BIN" \
    --out /scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/bdfe_pilot_aldehydes.csv \
    --work-dir /scratch-shared/schen3/benzoin-dg/tmp/bdfe_pilot_ald
$PY -u calc_bde_free_energy.py --which products --n 10 --xtb-bin "$XTB_BIN" --shermo-bin "$SHERMO_BIN" \
    --out /scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/bdfe_pilot_products.csv \
    --work-dir /scratch-shared/schen3/benzoin-dg/tmp/bdfe_pilot_prod
echo "Done $(date) exit=$?"
