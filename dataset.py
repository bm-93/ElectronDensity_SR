import h5py
import numpy as np
import glob
import random
from torch.utils.data import Dataset, DataLoader


class EMDataset(Dataset):
    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = [x.replace('_gt.h5', '') for x in glob.glob(self.datapath + '/*_gt.h5')]

        self.min_val = -1.5
        self.max_val = 2.5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input = h5py.File(self.samples[idx] + '_input.h5', 'r').get('data')[()]
        gt = h5py.File(self.samples[idx] + '_gt.h5', 'r').get('data')[()]
        info = self.samples[idx]

        # data normalization 
        input = ((input - self.min_val) / (self.max_val - self.min_val) - 0.5) * 2.0
        gt = ((gt - self.min_val) / (self.max_val - self.min_val) - 0.5) * 2.0

        if self.transforms:
            input, gt = self.transforms(input), self.transforms(gt)
        
        return {"A": input, "B": gt, "info": info}


class UnalignEMDataset(Dataset):
    def __init__(self, datapath_xray, datapath_cryoem, transforms_):
        self.datapath_xray = datapath_xray
        self.datapath_cryoem = datapath_cryoem
        self.transforms = transforms_
        self.samples_xray = [x.replace('_gt.h5', '') for x in glob.glob(self.datapath_xray + '/*_gt.h5')]
        self.samples_cryoem = [x.replace('.h5', '') for x in glob.glob(self.datapath_cryoem + '/*.h5')]

        self.min_val = -1.5
        self.max_val = 2.5

    def __len__(self):
        return len(self.samples_cryoem)

    def __getitem__(self, idx):
        rand_idx = random.randint(0, len(self.samples_xray) - 1)
        cryoem = h5py.File(self.samples_cryoem[idx] + '.h5', 'r').get('data')[()]
        xray = h5py.File(self.samples_xray[rand_idx] + '_gt.h5', 'r').get('data')[()]
        info_A = self.samples_cryoem[idx]
        info_B = self.samples_xray[rand_idx]

        # data normalization 
        cryoem = ((cryoem - self.min_val) / (self.max_val - self.min_val) - 0.5) * 2.0
        xray = ((xray - self.min_val) / (self.max_val - self.min_val) - 0.5) * 2.0

        if self.transforms:
            xray, cryoem = self.transforms(xray), self.transforms(cryoem)
        
        return {"A": cryoem, "B": xray, "info_A": info_A, "info_B": info_B}


if __name__ == "__main__":
    
    import torchvision.transforms as transforms
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = UnalignEMDataset('data/train', 'data_cryoem/train', transforms_)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, batch["A"].shape)
    # print(dataset.samples_xray)
    # print(dataset.samples_cryoem)
    # print(dataset.__getitem__(0))