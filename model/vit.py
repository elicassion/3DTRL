import torch
import torch.nn as nn
import torch.functional as F
import timm
class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.config = config
        base_name = "vit_t16"

        self.name = f"{base_name}_l{config.n_layer+config.n_3denc_layer}-0-0-{config.dataset_name}"
        if config.note != "":
            self.name += f"-{config.note}"
        self.feature_extractor = timm.create_model('vit_tiny_patch16_224', pretrained=config.pretrained, num_classes=config.D)


    def forward(self, x):
        m = self.feature_extractor
        x = m.patch_embed(x)
        cls_token = m.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        for i in range(self.config.n_layer+self.config.n_3denc_layer):
            x = m.blocks[i](x)
        # x = m.blocks(x)
        x = m.norm(x)
        return (x, None, None, None, None)
