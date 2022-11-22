import os, sys, time, glob
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
from utils import to_tensor, normalize


class TrajectoryDataset(Dataset):
    def __init__(self, dataset=None, trs=None, view_range=None, window=3, config=None, aug=None):
        self.data_root = "/home/jishang/ftpv_dataset"
        self.config = config
        self.dataset, self.trs, self.view_range, self.window = dataset, trs, view_range, window
        self.aug = aug
        
    def load_data(self, tr):
        """
        return a multi-view trajectory of shape (V, T, C, H, W)
        """
        # tr = 1
        # sample_indices = [10, 25, 30, 35, 40, 45, 50]
        base_path = f"{self.data_root}/{self.dataset}/{tr}"
        imgs = sorted([ p  for p in glob.glob(os.path.join(base_path, "fpv") + "/*") if '.jpg' in p or '.png' in p])
        # print (imgs)
        n_frames = len(imgs)
        x_f, x_ts = [], [] # l, l*8 flattened
        for i in range(n_frames):
            f = Image.open(f"{base_path}/fpv/{i:06d}.jpg")
            f = f.resize((224, 224), Image.BICUBIC)
            if self.aug is not None:
                f = self.aug(f)
            x_f.append(to_tensor(f))
        for view in self.view_range:
            for i in range(n_frames):
                t = Image.open(f"{base_path}/tpv/view_{view:04d}/{i:06d}.jpg")
                t = t.resize((224, 224), Image.BICUBIC)
                if self.aug is not None:
                    t = self.aug(t)
                x_ts.append(to_tensor(t))
        x_f = torch.stack([normalize(x) for x in x_f])
        x_t = torch.stack([normalize(x) for x in x_ts]).reshape(len(self.view_range), n_frames, 3, 224, 224)
        # print (x_f.size(), x_t.size())
        # print (torch.sum(torch.abs(x_t[0,1]-x_t[0,2])), torch.sum(torch.abs(x_t[0,1]-x_t[1,1]))) # check resize correct
        x = torch.cat((x_f.unsqueeze(0), x_t))
        return x # V, T, C, H, W, on CPU
        
    
    def __getitem__(self, tr_index):
        return self.load_data(self.trs[tr_index])
    
    def __len__(self):
        return len(self.trs)
    
    
class TripletDataset(Dataset):
    def __init__(self, data=None, window=3, n_pos=1, n_neg=1, device='cpu'):
        self.data = data # a tensor from one view (V, T, C, H, W)
        self.n_views, self.n_frames = data.size(0), data.size(1)
        self.window = window
        self.n_pos, self.n_neg = n_pos, n_neg
        self.view_pairs = [(i, j) for i in range(self.n_views) for j in range(i+1, self.n_views)]
        self.max_len = len(self.view_pairs) * self.n_frames
    
    def __getitem__(self, idx):
        window, n_frames, n_views = self.window, self.n_frames, self.n_views
        
        anchor = idx % self.n_frames
        i, j = self.view_pairs[idx // self.n_frames]
        
        positives = np.random.choice(np.concatenate([np.arange(max(anchor-window,0), anchor), 
                                                                 np.arange(anchor+1, min(anchor+1+window, n_frames))]), self.n_pos, replace=False)
        negatives = np.random.choice(np.concatenate([np.arange(0, anchor-window), 
                                                     np.arange(anchor+1+window, n_frames)]), self.n_neg, replace=False)
        indices = [anchor] + positives.tolist() + negatives.tolist()
        
        return self.data[i][indices], self.data[j][indices]

        
    def __len__(self):
        return self.max_len