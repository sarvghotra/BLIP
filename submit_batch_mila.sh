#!/bin/bash

#SBATCH --job-name=ft_repro_resume3_rtx4
#SBATCH --output=job_output_ft_repro_resume2_rtx4.txt
#SBATCH --error=job_error_ft_repro_resume2_rtx4.txt
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:rtx8000:8

module load miniconda/3
conda activate blip_py37
module load cuda/11.1

# python -m torch.distributed.run --nproc_per_node=1 train_vqa.py --evaluate --config ./configs/dry_run_vqa.yml --output_dir ~/scratch/models/blip/dry_run_sbatch/
cd /home/mila/s/sarvjeet-singh.ghotra/git/BLIP/util_scripts/
bash finetune_vqa_rtx8.sh

echo "====== Done Yo ====="

# # SBATCH --reservation=DGXA100
# # SBATCH --mem=192G
# # SBATCH -c 32
