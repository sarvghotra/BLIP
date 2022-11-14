cd ..

python -m torch.distributed.run --master_port 29501 \
--nproc_per_node=3 \
train_vqa.py \
--config configs/repro_vqa.yaml \
--output_dir /home/mila/s/sarvjeet-singh.ghotra/scratch/models/blip/ft_repro2_resume1/ \

# --evaluate \ ft_repro

echo "Done"