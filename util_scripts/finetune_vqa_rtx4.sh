cd ..

python -m torch.distributed.run --master_port 29501 \
--nproc_per_node=4 \
train_vqa.py \
--config configs/repro_vqa_rtx4.yaml \
--output_dir /home/mila/s/sarvjeet-singh.ghotra/scratch/models/blip/ft_repro2_resume2_rtx4/ \

# --evaluate \ ft_repro

echo "Done"