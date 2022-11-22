import torch
import torch.nn as nn
import torch.functional as F

from timesformer.models.backbone.nets import CameraCoordEstimator, CameraProps, CameraProjection, uvd_to_xyz
from einops import rearrange, repeat
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
        self.register_buffer("v", torch.stack([torch.tensor([i] * n_pw, dtype=torch.float) for i in range(n_ph)]).view(
            n_ph * n_pw, 1) - (n_ph - 1) / 2)
        self.register_buffer("u", torch.stack(
            [torch.tensor(list(range(n_pw)), dtype=torch.float) for i in range(n_ph)]).view(n_ph * n_pw, 1) - (
                                         n_pw - 1) / 2)

        """
        Depth/3d coordinates estimator. Estimate depth/3d 
        in camera-centered coord-system, for each token.
        """
        if config.pred_depth:
            self.camera_coord = CameraCoordEstimator(config, dim=1)  # literally a depth estimator
        else:
            self.camera_coord = CameraCoordEstimator(config, dim=3)

        """
        Estimate camera rotation and translation 
        matrices from global information. 
        """
        self.camera_props = CameraProps(config, pred_campos_from=config.pred_campos_from)

        """
        Project estimated camera-centered coordinates 
        to world coordinates by estimated camera params
        """
        self.camera_proje = CameraProjection()

    def forward(self, x, B, T, W, H):
        """
        x: tokens (B, N+1, C)
        """
        config = self.config
        if config.pred_depth:
            depth = self.camera_coord(x[:, 1:])
            coord = uvd_to_xyz(self.u.expand_as(depth), self.v.expand_as(depth), depth)
        else:
            depth = None
            coord = self.camera_coord(x[:, 1:])

        # single camera parameters for a single video sample

        cls_token = x[:, 0, :]
        cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
        new_cls_token = torch.mean(cls_token, 1, True)  # for T frames
        x = x[:, 1:, :]
        x = rearrange(x, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
        x = torch.cat((new_cls_token, x), dim=1)

        rot, trans = self.camera_props(x[:, 0], x[:, 1:])
        rot = repeat(rot, 'b r_i r_j -> (b tile) r_i r_j', tile=T)
        trans = repeat(trans, 'b t_i t_j -> (b tile) t_i t_j', tile=T)
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


    def forward(self, x, pos_emb_2d, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W
        cls_token = x[:, 0, :]
        new_cls_token = cls_token.repeat(B*T, 1, 1)
        x = x[:, 1:, :]
        x = rearrange(x, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
        x = torch.cat((new_cls_token, x), dim=1)
        world_coord, depth, coord, rot, trans = self.estimator(x, B, T, W, H)
        pos_emb_world = torch.cat((self.pos_emb_world_cls.repeat(B*T, 1, 1), self.pos_emb_world(world_coord)),
                                  dim=1) + pos_emb_2d.repeat(B*T, 1, 1)
        x = x + pos_emb_world
        cls_token = x[:, 0, :]
        cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
        new_cls_token = torch.mean(cls_token, 1, True)  # for T frames
        x = x[:, 1:, :]
        x = rearrange(x, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
        x = torch.cat((new_cls_token, x), dim=1)
        return x


if __name__ == '__main__':
    from utils import Config

    config = Config(depth=12,
                    patch_size=16,
                    pred_depth=True,
                    pred_campos_from="both-sep",
                    embed_dim=192, num_heads=3, pretrained=False,
                    backbone='timesformer', attention_type='divided')
    model = ThreeDTRL(config)
    # input features and 2d pos embeddings from enc
    x = torch.randn((6, 196 * 4 + 1, 768))
    pos_2d = torch.randn((1, 196 + 1, 768))
    y = model.forward(x, pos_2d)
    print (y.size())
