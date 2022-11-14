cd ..

python -m torch.distributed.run --master_port 29501 \
--nproc_per_node=1 \
train_vqa.py \
--config configs/repro_vqa.yaml \
--output_dir /home/mila/s/sarvjeet-singh.ghotra/scratch/models/blip/debug/ \

# --evaluate \ ft_repro

echo "Done"