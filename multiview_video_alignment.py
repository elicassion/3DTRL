import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import os, time, random, sys, pickle
import torchvision.models as models
import argparse

from torchvision.utils import make_grid
from utils import seed_everything, distance, alignment_error, cycle_error, kendalls_tau
from utils import to_tensor, normalize
import matplotlib.pyplot as plt
import glob, timm
from data_pipeline.trajectory_dataset import TrajectoryDataset, TripletDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# data params
parser.add_argument('--data', type=str, default="panda")
parser.add_argument('--train_views', type=int, default=-1)
parser.add_argument('--train_videos', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=1)

# model structure params
parser.add_argument('--model', type=str, default="vitnpl_tiny_patch16_224")
parser.add_argument('--pred_depth', type=int, default=1)
parser.add_argument('--pred_campos_from', type=str, default="both-sep")
# parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--pretrained', action='store_true')

#
parser.add_argument('--test', action='store_true')


# misc
parser.add_argument('--note', type=str, default="")

# loss params
parser.add_argument('--lr', type=float, default=1e-6) # main lr


## for distributed
parser.add_argument('--distributed', type=int, default=0)
parser.add_argument('--address', type=str, default="localhost")
parser.add_argument('--port', type=str, default="64751")
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--xpar', action='store_true') # close progress bar

args = parser.parse_args()



seed_everything(args.seed)
torch.set_num_threads(8)



from PIL import Image
dataset = args.data

# dataset = "Pandav1_Train_Frame" # "MineRLReachGoalGroundStatic-v0_Frame" "can_mg" "can_mh" "lift_mh" "lift_ph"
if "panda" in dataset.lower():
    dataset = "Pandav1_Train_Frame"
    train_trs = [f"tr_{x:04d}" for x in range(10)]
    eval_trs = [f"tr_{x:04d}" for x in range(10, 15)]
    test_trs = [f"tr_{x:04d}" for x in range(15, 20)]
    max_views = 9
    window = 3
    
elif "mine" in dataset.lower():
    dataset = "MineRLReachGoalGroundStatic-v0_Frame"
    train_trs = [f"tr_{x:04d}" for x in range(4)]
    eval_trs = [f"tr_{x:04d}" for x in range(4, 6)]
    test_trs = [f"tr_{x:04d}" for x in range(6, 8)]
    max_views = 8
    window = 3
    
elif "lift_mh" in dataset.lower() or "can_mh" in dataset.lower():
    train_trs = [f"tr_{x:04d}" for x in range(200)]
    eval_trs = [f"tr_{x:04d}" for x in range(200, 250)]
    test_trs = [f"tr_{x:04d}" for x in range(250, 300)]
    max_views = 4
    window = 2
    
elif "pouring" in dataset.lower():
    data_dir = f"/home/jishang/ftpv_dataset/{dataset}"
    train_items = pickle.load(open(os.path.join(data_dir, "train.pkl"), "rb"))[:-10]
    val_items = train_items[-10:]
    test_items = pickle.load(open(os.path.join(data_dir, "val.pkl"), "rb"))
    def get_name(item):
        name = item["name"]
        name = name.split("_real_")[0]
        return name
    assert args.train_videos < len(train_items)
    if args.train_videos == -1:
        train_trs = [get_name(item) for item in train_items]
    else:
        train_trs = [get_name(item) for item in train_items][:args.train_videos]
    eval_trs = [get_name(item) for item in val_items]
    test_trs = [get_name(item) for item in test_items]
    max_views = 1
    window = 3
else:
    raise NotImplementedError()

if args.train_views < 0 or args.train_views > max_views:
    args.train_views = max_views
train_views = eval_views = range(args.train_views)

if args.train_views < max_views:
    test_views = range(args.train_views, max_views)
else:
    test_views = range(max_views)
    
args.note = f"views{len(train_views):02d}-{args.note}"
if args.train_videos > -1:
    args.note += f"videos{args.train_videos}"
device = "cuda:0"

# one item is a trajectory with multiple views
# this iterates over trajectory indices
train_dataset = TrajectoryDataset(dataset, train_trs, train_views, window, config=args)
train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=None, shuffle=True)


from model.vit import ViT
import model.vit_3dtrl as vit_3dtrl
import model.tnt_3dtrl as tnt_3dtrl
import model.swin_3dtrl as swin_3dtrl
from utils import *

    
def tc_loss(z1, z2, detach=False):
    T, D = z1.size(0), z1.size(1)
    dist_mat_ori = cos_distance(z1[0:1].expand_as(z2).reshape(-1, D), z2.reshape(-1, D))
    dist_mat = torch.exp(dist_mat_ori*10)
    # print("distmat", dist_mat.size(), dist_mat)
    return -(torch.log(dist_mat[0]) - torch.log(torch.sum(dist_mat))), dist_mat_ori

def simple_tc_loss(z1, z2):
    alpha = 0.3
    anc1, pos1, neg1 = z1[..., 0, :], z1[..., 1, :], z1[..., 2, :]
    anc2, pos2, neg2 = z2[..., 0, :], z1[..., 1, :], z1[..., 2, :]
    match = distance(anc1, anc2) + distance(pos1, pos2)
    attract =  distance(anc1, pos1) + distance(anc2, pos2) + distance(anc1, pos2) + distance(anc2, pos1)
    repulsion = - distance(anc1, neg2)  - distance(anc2, neg1) - distance(anc1, neg1)  - distance(anc2, neg2)
    return torch.clamp(attract + match + repulsion + alpha*100, 0) #distance(z1[0:1].expand_as(z2), z2)



# load multiple views from single trajectory
# this loads actual frames
def load_data(tr, sample_indices, view_range):
    base_path = f"/home/jishang/ftpv_dataset/{dataset}/{tr}"
    x_f, x_ts = [], [] # l, l*8 flattened
    for i in sample_indices:
        f = Image.open(f"{base_path}/fpv/{i:06d}.jpg")
        f = f.resize((224, 224), Image.BICUBIC)
        x_f.append(to_tensor(f))
    for view in view_range:
        for i in sample_indices:
            t = Image.open(f"{base_path}/tpv/view_{view:04d}/{i:06d}.jpg")
            t = t.resize((224, 224), Image.BICUBIC)
            x_ts.append(to_tensor(t))
    x_f = torch.stack([normalize(x) for x in x_f])
    x_t = torch.stack([normalize(x) for x in x_ts]).reshape(len(view_range), len(sample_indices), 3, 224, 224)
    # print (x_f.size(), x_t.size())
    # print (torch.sum(torch.abs(x_t[0,1]-x_t[0,2])), torch.sum(torch.abs(x_t[0,1]-x_t[1,1]))) # check resize correct
    x = torch.cat((x_f.unsqueeze(0), x_t))
    return x # view, sample (anchor/pos + neg), C, H, W


if not args.test:
    if "3dtrl_" in args.model:
        if "vit" in args.model:
            pkg = vit_3dtrl
        elif "swin" in args.model:
            pkg = swin_3dtrl
        elif "tnt" in args.model:
            pkg = tnt_3dtrl
        if "swin" in args.model:
            m3d = pkg.__dict__[args.model](
            global_pool="avg",
            npl_depth=args.n_layer,
            num_classes=1000)
        else:
            m3d = pkg.__dict__[args.model](
                global_pool=None,
                npl_depth=args.n_layer,
                num_classes=1000)
    else:
        m3d = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=1000,
            global_pool=None)
    setattr(m3d, "name", f"{args.model}-{dataset}-{args.note}")
    print (m3d.name)
    print ("-------------")
    print (args)
    from torch.optim import Adam
    optimizer = Adam(m3d.parameters(), lr=args.lr)


    num_negatives = 1
    num_positives = 1


    m3d.to(device)
    best = 0.999
    best_ep = -1
    from tqdm import tqdm
    for ep in range(100):
        # train
        m3d.train()
        losses = []
        dist_mats = []
        itr = tqdm(train_loader, leave=False) if not args.xpar else train_loader
        for tr in itr:
            tr = tr.squeeze(0)
            frame_dataset = TripletDataset(data=tr, window=window, n_pos=num_positives, n_neg=num_negatives)
            frame_dataloader = DataLoader(frame_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, drop_last=False, sampler=None, shuffle=True)
            for x1, x2 in frame_dataloader:
                B, T, C, H, W = x1.size() # T=3
                x1 = x1.view(B*T, C, H, W).to(device)
                x2 = x2.view(B*T, C, H, W).to(device)
                s1 = m3d.forward_features(x1)
                s2 = m3d.forward_features(x2)
                if "swin" in args.model:
                    s1, s2 = s1.mean(dim=1).view(B,T,-1), s2.mean(dim=1).view(B,T,-1)
                else:
                    s1, s2 = s1[:, 0].view(B, T, -1), s2[:, 0].view(B, T, -1)
                # print (s1.size(), s2.size())
                loss = simple_tc_loss(s1, s2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m3d.parameters(), max_norm=10.0)
                optimizer.step()

                losses.append(loss.item())


        # eval
        m3d.eval()
        eval_align_errors = []
        eval_cycle_errors = []
        eval_taus = []
        for tr in eval_trs:
            path = f"/home/jishang/ftpv_dataset/{dataset}/{tr}/fpv"
            imgs = sorted([ p  for p in glob.glob(path + "/*") if '.jpg' in p or '.png' in p])
            # print (imgs)
            n_frames = len(imgs)
            indices = list(range(0, n_frames, 1))
            x = load_data(tr, indices, eval_views)
            with torch.no_grad():
                for i in range(x.size(0)):
                    for j in range(i+1, x.size(0)):
                        x1, x2 = x[i].to(device), x[j].to(device)
                        s1, s2 = m3d.forward_features(x1), m3d.forward_features(x2)
                        if "swin" in args.model:
                            s1, s2 = s1.mean(dim=1).cpu(), s2.mean(dim=1).cpu()
                        else:
                            s1, s2 = s1[:, 0].cpu(), s2[:, 0].cpu()
                        eval_align_errors.append(alignment_error(s1, s2))
                        eval_cycle_errors.append(cycle_error(s1, s2))
                        eval_taus.append(kendalls_tau(s1, s2))
        a_error = np.mean(eval_align_errors)
        c_error = np.mean(eval_cycle_errors)
        tau = np.mean(eval_taus)
        print (f"Ep {ep}, Eval Align Error: {a_error:.4f}, Eval Cycle Error: {c_error:.4f}, Loss {np.mean(losses):.4f}, Eval Kendall's Tau: {tau:.4f}")
        if a_error < best:
            torch.save(m3d.state_dict(), f"trained_models/{m3d.name}_best.pth")
            best = a_error
            best_ep = ep
        else:
            if ep - best_ep >= (10 if dataset == "pouring" else 10):
                print (f"Best Ep: {best_ep}")
                break

    del m3d
    
# test
if "npl_" in args.model:
        if "vit" in args.model:
            if "mh" in args.__dict__ and args.mh is True:
                pkg = vitnpl_mh
            else:
                pkg = vitnpl
        elif "swin" in args.model:
            pkg = swinnpl
        elif "tnt" in args.model:
            pkg = tntnpl

        if "swin" in args.model:
            m3d = pkg.__dict__[args.model](
            global_pool="avg",
            npl_depth=args.n_layer,
            num_classes=1000)
        else:
            m3d = pkg.__dict__[args.model](
                global_pool=None,
                npl_depth=args.n_layer,
                num_classes=1000)
else:
    m3d = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        global_pool=None)
setattr(m3d, "name", f"{args.model}-{dataset}-{args.note}")

m3d.to(device)

m3d.load_state_dict(torch.load(f"trained_models/{m3d.name}_best.pth"))
m3d.eval()
test_align_errors = []
test_cycle_errors = []
test_taus = []
for tr in test_trs:
    path = f"/home/jishang/ftpv_dataset/{dataset}/{tr}/fpv"
    imgs = sorted([ p  for p in glob.glob(path + "/*") if '.jpg' in p or '.png' in p])
    # print (imgs)
    n_frames = len(imgs)
    indices = list(range(0, n_frames, 1))
    x = load_data(tr, indices, test_views)
    with torch.no_grad():
        for i in range(x.size(0)):
            for j in range(i+1, x.size(0)):
                x1, x2 = x[i].to(device), x[j].to(device)
                s1, s2 = m3d.forward_features(x1), m3d.forward_features(x2)
                if "swin" in args.model:
                    s1, s2 = s1.mean(dim=1).cpu(), s2.mean(dim=1).cpu()
                else:
                    s1, s2 = s1[:, 0].cpu(), s2[:, 0].cpu()
                test_align_errors.append(alignment_error(s1, s2))
                test_cycle_errors.append(cycle_error(s1, s2))
                test_taus.append(kendalls_tau(s1, s2))
a_error = np.mean(test_align_errors)
c_error = np.mean(test_cycle_errors)
tau = np.mean(test_taus)
print (f"Test Align Error: {a_error:.4f}, Test Cycle Error: {c_error:.4f}, Test Kendall's Tau: {tau:.4f}")



