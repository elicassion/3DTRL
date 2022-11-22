import torch
import timm
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from model.three_d_trl import ThreeDTRL
from utils import Config
class ViT_3DTRL(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, trl_depth=4, **kwargs):
        super(ViT_3DTRL, self).__init__(**kwargs)
        config = Config(
                has_cls = True,
                input_size = (3, 224, 224),
                pred_depth = True,
                pred_campos_from="both-sep",
                **kwargs)
        self.trl_depth = trl_depth # where to insert 3DTRL
        self.three_d_trl = ThreeDTRL(config)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            # del self.norm  # remove the original norm

            
    def forward_features(self, x, out_emb_only=True):
        B = x.shape[0]

        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            # ----insert 3DTRL
            if i == self.trl_depth:
                x = self.three_d_trl(x, self.pos_embed, out_emb_only, x2d=False)
            # ----insert end
            x = blk(x)
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
            
        return x

@register_model
def vit_3dtrl_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model


@register_model
def vit_3dtrl_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model



@register_model
def vit_3dtrl_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model


@register_model
def vit_3dtrl_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model


@register_model
def vit_3dtrl_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model


@register_model
def vit_3dtrl_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model


@register_model
def vit_3dtrl_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = ViT_3DTRL(**model_kwargs)
    return model
