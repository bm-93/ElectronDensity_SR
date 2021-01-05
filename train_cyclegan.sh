n_epochs=200
batch_size=4
img_height=64
img_width=64
img_depth=64
disc_update=5
xray_data_dir=data_aug
cryoem_data_dir=data_cryoem_aug
gpu=cuda:0

python cyclegan.py \
    --n_epochs ${n_epochs} \
    --batch_size ${batch_size} \
    --img_height ${img_height} \
    --img_width ${img_width} \
    --img_depth ${img_depth} \
    --disc_update ${disc_update} \
    --xray_data_dir ${xray_data_dir} \
    --cryoem_data_dir ${cryoem_data_dir} \
    --gpu ${gpu}