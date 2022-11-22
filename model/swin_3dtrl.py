import torch
import timm
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from model.three_d_trl import ThreeDTRL
from utils import Config
class Swin_3DTRL(timm.models.swin_transformer.SwinTransformer):
    """ Swin 3DTRL """
    def __init__(self, trl_depth=1, **kwargs):
        super(Swin_3DTRL, self).__init__(**kwargs)
        config = Config(
                has_cls = False,
                input_size = (3, 224, 224),
                pred_depth = True, 
                pred_campos_from="tok",
                **kwargs)
        """ resolve Swin patch mergings """
        for i in range(trl_depth):
            config.patch_size *= 2
            config.embed_dim *= 2
        self.three_d_trl = ThreeDTRL(config)
        
        self.trl_depth = trl_depth # where to insert 3DTRL
            
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            if i == self.trl_depth:
                x = self.three_d_trl(x, None)
            x = layer(x)
        x = self.norm(x)  # B L C
        
        return x

@register_model
def swin_3dtrl_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-B @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model = Swin_3DTRL(**model_kwargs)
    return model


@register_model
def swin_3dtrl_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model = Swin_3DTRL(**model_kwargs)
    return model


@register_model
def swin_3dtrl_tiny_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model = Swin_3DTRL(**model_kwargs)
    return model