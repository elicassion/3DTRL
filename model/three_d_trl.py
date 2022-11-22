import torch
import torch.nn as nn
import torch.functional as F

from backbone.nets import CameraCoordEstimator, CameraProps, CameraProjection, uvd_to_xyz, rotation_tensor

import copy

class Estimator(nn.Module):
    """
    x: 2d-representation tokens (B, N+1, C)
    output: estimated world coordinates (B, N, 3), depth (if has) (B, N, 1), 
            camera-centered coordinates (B, N, 3), rotation (B, 3, 3), 
            translation (B, 3, 3)
    """
    def __init__(self, config):
        super(Estimator, self).__init__()
        self.config = config
        
        """
        register 2d-coordinates in patch numbers
        u (N, 1), v (N, 1)
        """
        C, H, W = config.input_size
        n_pw, n_ph = H // config.patch_size, W // config.patch_size
        self.register_buffer("v", torch.stack([torch.tensor([i]*n_pw, dtype=torch.float) for i in range(n_ph)]).view(n_ph*n_pw, 1) - (n_ph-1)/2)
        self.register_buffer("u", torch.stack([torch.tensor(list(range(n_pw)), dtype=torch.float) for i in range(n_ph)]).view(n_ph*n_pw, 1) - (n_pw-1)/2)
        
        """
        Depth/3d coordinates estimator. Estimate depth/3d 
        in camera-centered coord-system, for each token.
        """
        if config.pred_depth:
            self.camera_coord = CameraCoordEstimator(config, dim=1) # literally a depth estimator
        else:
            self.camera_coord = CameraCoordEstimator(config, dim=3)
        
        """
        Estimate camera rotation and translation 
        matrices from global information. 
        """
        if config.has_cls:
            self.camera_props = CameraProps(config, pred_campos_from=config.pred_campos_from)
        else:
            self.camera_props = CameraProps(config, pred_campos_from='tok')
        
        """
        Project estimated camera-centered coordinates 
        to world coordinates by estimated camera params
        """
        self.camera_proje = CameraProjection()
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
    
    def forward(self, x):
        """
        x: tokens (B, N+1, C)
        """
        config = self.config
        if config.pred_depth:
            depth = self.camera_coord(x[:, 1:] if config.has_cls else x)
            coord = uvd_to_xyz(self.u.expand_as(depth), self.v.expand_as(depth), depth)
        else:
            depth = None
            coord = self.camera_coord(x[:, 1:] if config.has_cls else x)
        
        
        if config.has_cls:
            rot, trans = self.camera_props(x[:, 0], x[:, 1:])
        else:
            rot, trans = self.camera_props(None, x)
        world_coord = self.camera_proje(coord, rot, trans)
        
        return world_coord, depth, coord, rot, trans
        
        
class ThreeDTRL(nn.Module):
    """
    input: 2d-representation tokens from images (B, N+1, C)
    output: 3d-representation tokens (B, N+1, C)
    """
    def __init__(self, config):
        super(ThreeDTRL, self).__init__()
        self.estimator = Estimator(config)
        self.config = config
        
        if config.has_cls:
            """
            create a new pos embedding for [CLS] for 3d representation
            """
            self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, config.embed_dim)))

        """
        convert 3d coordinates to positional embeddings
        """
        self.pos_emb_world = nn.Sequential(
            nn.Linear(3, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

    
    def forward(self, x, pos_emb_2d=None, x2d=False, **kwargs):
        world_coord, depth, coord, rot, trans = self.estimator(x)
        B = x.size(0)
        if self.config.has_cls:
            pos_emb_world = torch.cat((self.pos_emb_world_cls.repeat(B, 1, 1), self.pos_emb_world(world_coord)), dim=1) 
        else:
            pos_emb_world = self.pos_emb_world(world_coord)
        
        """
        If x2d=True, skip fusing original 2D positional embedding.
        We use x2d=False by default
        """
        if x2d:
            pass
        else:
            if pos_emb_2d is not None:
                pos_emb_world += pos_emb_2d.repeat(B//pos_emb_2d.size(0), 1, 1)
            
        return x + pos_emb_world



    
class ThreeDTRLCat(nn.Module):
    """
    The concatination version of 3DTRL
    input: 2d-representation tokens from images (B, N+1, C)
    output: 3d-representation tokens (B, N+1, C)
    """
    def __init__(self, config):
        super(ThreeDTRLCat, self).__init__()
        self.estimator = Estimator(config)
        self.config = config
        
        if config.has_cls:
            """
            create a new pos embedding for [CLS] for 3d representation
            """
            self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, config.embed_dim)))

        """
        convert 3d coordinates to positional embeddings
        """
        self.proj_world = nn.Sequential(
            nn.Linear(config.embed_dim+3, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

    
    def forward(self, x, pos_emb_2d=None, **kwargs):
        world_coord, depth, coord, rot, trans = self.estimator(x)
        B = x.size(0)
        x_token = self.proj_world(torch.cat((x[:, 1:, :], world_coord), dim=-1))
        if self.config.has_cls:
            x_world = torch.cat((self.pos_emb_world_cls.repeat(B, 1, 1), x_token), dim=1) 
        else:
            x_world = x_tokens
            
        return x + x_world
        
        
        
if __name__ == '__main__':
    
    # verify 3DTRL!
    from utils import Config
    # config = Config(n_layer=4, n_3denc_layer=4,
    #             n_patches = (14, 14),
    #             pred_depth = True, 
    #             pred_campos_from="both-sep",
    #             n_embd=768, D=768, N_head=8, pretrained=False,
    #             backbone='vit')
    # model = ThreeDTRL(config)
    # x = torch.randn((6,196+1,768))
    # pos_2d = torch.randn((1,196+1,768))
    # y = model.forward(x, pos_2d)
    # print (y.size())
    
    config = Config(depth=12,
                patch_size = 16,
                pred_depth = True, 
                pred_campos_from="both-sep",
                embed_dim=192, num_heads=3, pretrained=False,
                backbone='timesformer', attention_type='divided')
    model = ThreeDTRL(config)
    # input features and 2d pos embeddings from enc
    x = torch.randn((6,196*4+1,768))
    pos_2d = torch.randn((1,196+1,768))
    y = model.forward(x, pos_2d)
    print (y.size())