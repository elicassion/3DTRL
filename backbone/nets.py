import torch
import torch.nn as nn
import torch.functional as F

def rotation_tensor(theta, phi, psi, b=1):
    """
    Takes theta, phi, and psi and generates the
    3x3 rotation matrix. Works for batched ops
    As well, returning a Bx3x3 matrix.
    """
    device = theta.device
    
    one = torch.ones(b, 1, 1).to(device)
    zero = torch.zeros(b, 1, 1).to(device)
    theta = theta.view(b, 1, 1)
    phi = phi.view(b, 1, 1)
    psi = psi.view(b, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1)), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1)), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))


def reverse_camera_props(rot, trans):
    return rot.T, rot.T @ trans.T


class CameraCoordEstimator(nn.Module):
    """
    Input:
        Feature tokens B N D
    Ouput:
        Estimated coordinates of each token B N 1(3) in camera view
        1 is depth, 3 is x y z
    """
    
    def __init__(self, config, dim=3):
        super(CameraCoordEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class CameraProps(nn.Module):
    """
    Generates the extrinsic rotation and translation matrix
    For the current camera. Takes CLS and pos_emb as input, then
    Returns the rotation matrix (3x3) and translation (3x1)
    """
    def __init__(self, config, pred_campos_from="both-sep"):
        super(CameraProps, self).__init__()
        self.pred_campos_from = pred_campos_from
        if pred_campos_from == "both-sep":
            self.token_proj_1 = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU()
            )
            self.token_proj_2 = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU()
            )
            self.cls_proj = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU()
            )
        elif pred_campos_from == "both-avg" or pred_campos_from == "tok":
            self.token_proj_1 = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU()
            )
            self.token_proj_2 = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU()
            )
        elif pred_campos_from == "cls":
            self.cls_proj = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU()
            )

            
        
        self.mix_proj = nn.Sequential(
            nn.Linear(config.embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.rot = nn.Linear(32, 3)
        self.trans = nn.Linear(32, 3)
        
        self.trans.weight.data.copy_(torch.ones(32, dtype=torch.float))
        self.trans.bias.data.zero_()
        # self.rot.weight.data.zero_()
        # self.rot.bias.data.zero_()

    def forward(self, cls_x, x):
        B = x.size(0)
        if self.pred_campos_from == "both-sep":
            cls_x = self.cls_proj(cls_x)
            x = self.token_proj_1(x)
            x = torch.mean(x, dim=1)
            x = self.token_proj_2(x)
            x = cls_x + x
 
        elif self.pred_campos_from == "both-avg":
            x = torch.cat((cls_x.unsqueeze(1), x), dim=1)
            x = self.token_proj_1(x)
            x = torch.mean(x, dim=1)
            x = self.token_proj_2(x)
 
        elif self.pred_campos_from == "cls":
            x = self.cls_proj(cls_x)

        elif self.pred_campos_from == "tok":
            x = self.token_proj_1(x)
            x = torch.mean(x, dim=1)
            x = self.token_proj_2(x)

        x = self.mix_proj(x)
        
        r = self.rot(x)
        return rotation_tensor(r[:, 0], r[:, 1], r[:, 2], B), self.trans(x).view(B, 1, 3)
    
def uvd_to_xyz(u, v, d):
    z = d
    return torch.cat((u*z, v*z, z), dim=-1)

    
class CameraProjection(nn.Module):
    def __init__(self, config=None):
        super(CameraProjection, self).__init__()
    
    def forward(self, p_camera, rot, trans):
        """
        p_camera: B N 3
        rot: 3x3
        trans: 3x3
        """
        # print (p_camera.size(), rot.size(), trans.size())
        return p_camera @ rot + trans
        
