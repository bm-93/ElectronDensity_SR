n_epochs=200
batch_size=4
img_height=64
img_width=64
img_depth=64
data_dir=data
gpu=cuda:0

python vox2vox.py \
    --n_epochs ${n_epochs} \
    --batch_size ${batch_size} \
    --img_height ${img_height} \
    --img_width ${img_width} \
    --img_depth ${img_depth} \
    --data_dir ${data_dir} \
    --gpu ${gpu}