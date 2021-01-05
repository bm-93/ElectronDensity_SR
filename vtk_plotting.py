import os
import h5py
import argparse
import numpy as np
from utils import vtk_plot


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='07200224_3dunet')
opt = parser.parse_args()
print(opt)

data_dir = 'data/test'
input_files = [f for f in os.listdir('data/test') if f.endswith('_input.h5')]
gt_files = [f.replace('_input', '_gt') for f in input_files]
pred_files = [f.replace('_input', '') for f in input_files]

# model_id = '07192114_3dunet'
# model_id = '07200224_3dunet'
# model_id = '07192006'
model_id = opt.model_id
test_dir = os.path.join('./test', model_id)
fig_dir = os.path.join('figs', model_id)
os.makedirs(fig_dir, exist_ok=True)

for input_fn, gt_fn, pred_fn in zip(input_files, gt_files, pred_files):
    # print(input_fn)

    input_path = os.path.join('data/test', input_fn)
    gt_path = os.path.join('data/test', gt_fn)
    pred_path = os.path.join(test_dir, pred_fn)

    input = h5py.File(input_path, 'r').get('data')[()]
    gt = h5py.File(gt_path, 'r').get('data')[()]
    pred = h5py.File(pred_path, 'r').get('data')[()][0,0,:,:,:]

    print(np.sum(np.where(input > 1)), np.max(input))
    print(np.sum(np.where(pred > 1)), np.max(pred))

    threshold = 0.0
    save_input_path = os.path.join(fig_dir, input_fn.replace('.h5', '.png'))
    save_gt_path = os.path.join(fig_dir, gt_fn.replace('.h5', '.png'))
    save_pred_path = os.path.join(fig_dir, pred_fn.replace('.h5', '_pred.png'))
    vtk_plot(input, threshold, save_path=save_input_path)
    vtk_plot(gt, threshold, save_path=save_gt_path)
    vtk_plot(pred, threshold, save_path=save_pred_path)

