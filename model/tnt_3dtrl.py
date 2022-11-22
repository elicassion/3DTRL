import torch
import timm
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from model.three_d_trl import ThreeDTRL
from utils import Config
class TNT_3DTRL(timm.models.tnt.TNT):
    """ TNT NPL """
    def __init__(self, global_pool=False, trl_depth=4, **kwargs):
        super(TNT_3DTRL, self).__init__(**kwargs)
        config = Config(
                has_cls = True,
                input_size = (3, 224, 224),
                pred_depth = True, 
                pred_campos_from="both-sep",
                **kwargs)

        self.three_d_trl = ThreeDTRL(config)
        
        self.trl_depth = trl_depth # where to insert 3DTRL
        
    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for i, blk in enumerate(self.blocks):
            if i == self.trl_depth:
                patch_embed = self.three_d_trl(patch_embed, self.patch_pos)
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return patch_embed


@register_model
def tnt_3dtrl_s_patch16_224(pretrained=False, **kwargs):
    model_cfg = dict(
        patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
        qkv_bias=False, **kwargs)
    model = TNT_3DTRL(**model_cfg)
    return model


@register_model
def tnt_3dtrl_b_patch16_224(pretrained=False, **kwargs):
    model_cfg = dict(
        patch_size=16, embed_dim=640, in_dim=40, depth=12, num_heads=10, in_num_head=4,
        qkv_bias=False, **kwargs)
    model = TNT_3DTRL(**model_cfg)
    return model
