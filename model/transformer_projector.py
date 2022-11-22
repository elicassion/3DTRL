import torch
import torch.nn as nn
import torch.functional as F

from backbone.transformer import Transformer, SelfAttention
from backbone.nets import CameraCoordEstimator, CameraProps, CameraProjection, uvd_to_xyz

import timm, copy


class TransProjector(nn.Module):
    
    def __init__(self, config):
        super(TransProjector, self).__init__()
        self.config = config
        self.name = f"transprojector_b16_l{config.n_layer}-{config.n_3denc_layer}-0"
        # config.D is 768 if we use vit-b-16
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=config.pretrained, num_classes=config.D)
        self.pos_emb = self.feature_extractor._parameters["pos_embed"] # (1, 1+14*14, 768)
        self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, 768))) # pos emb for cls token in world coord
        
        self.camera_props = CameraProps(config)
        self.camera_coord = CameraCoordEstimator(config)
        self.camera_proje = CameraProjection()
        
        self.pos_emb_world = nn.Sequential(
            nn.Linear(3, config.D),
            nn.ReLU(),
            nn.Linear(config.D, config.D)
        )
        
        enc_config, dec_config = copy.deepcopy(config), copy.deepcopy(config)
        enc_config.n_layer, dec_config.n_layer = config.n_3denc_layer, config.n_3ddec_layer
        self.enc_3d = Transformer(enc_config)
        # self.cls_head = nn.Linear(config.D, 128)
        self.dec_3d = Transformer(dec_config)
        
    def _forward_features(self, x):
        m = self.feature_extractor
        x = m.patch_embed(x)
        cls_token = m.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if m.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, m.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        x = m.blocks(x)
        x = m.norm(x)
        return x
        
    def forward_features(self, x):
        return self._forward_features(x)
    
    def forward_3d(self, x):
        config = self.config
        B, C, H, W = x.size()
        x = self.forward_features(x)
        # print("feature tokens", x.size())
        coord = self.camera_coord(x[:, 1:])
        rot, trans = self.camera_props(x[:, 0], x[:, 1:])
        world_coord = self.camera_proje(coord, rot, trans)
        pos_emb_world = torch.cat((self.pos_emb_world_cls.repeat(B, 1, 1), self.pos_emb_world(world_coord)), dim=1) + self.pos_emb.repeat(B, 1, 1)
        
        x = self.enc_3d(x + pos_emb_world)
        
        return x, coord, rot, trans
    
    def forward_3d_decode(self, x, query):
        B, N, D = x.size()
        if len(query.size()) > 3:
            query = self.forward_feature(query)
        x = self.dec_3d(x, query=query)
        return x
        
    def forward(self, x, xq=None):
        config = self.config
        B, C, H, W = x.size()
        # x = self.forward_feature(x)

        x_3d, coord, rot, trans = self.forward_3d(x)
        if xq is None:
            return (x_3d, coord, rot, trans)
        else:
            pass

class TransProjectorWorldCoord(nn.Module):
    
    def __init__(self, config):
        super(TransProjector, self).__init__()
        self.config = config
        # config.D is 768 if we use vit-b-16
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=config.pretrained, num_classes=config.D)
        self.pos_emb = self.feature_extractor._parameters["pos_embed"] # (1, 1+14*14, 768)
        self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, 768))) # pos emb for cls token in world coord
        
        self.camera_props = CameraProps(config)
        self.camera_coord = CameraCoordEstimator(config)
        self.camera_proje = CameraProjection()
        
        self.pos_emb_world = nn.Sequential(
            nn.Linear(3, config.D),
            nn.ReLU(),
            nn.Linear(config.D, config.D)
        )
        
        enc_config, dec_config = copy.deepcopy(config), copy.deepcopy(config)
        enc_config.n_layer, dec_config.n_layer = 2, 2
        self.enc_3d = Transformer(enc_config)
        # self.cls_head = nn.Linear(config.D, 128)
        self.dec_3d = Transformer(dec_config)
        
    def _forward_features(self, x):
        m = self.feature_extractor
        x = m.patch_embed(x)
        cls_token = m.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if m.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, m.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        x = m.blocks(x)
        x = m.norm(x)
        return x
        
    def forward_features(self, x):
        return self._forward_features(x)
    
    def forward_3d(self, x):
        config = self.config
        B, C, H, W = x.size()
        x = self.forward_features(x)
        # print("feature tokens", x.size())
        coord = self.camera_coord(x[:, 1:])
        rot, trans = self.camera_props(x[:, 0], x[:, 1:])
        world_coord = self.camera_proje(coord, rot, trans)
        pos_emb_world = torch.cat((self.pos_emb_world_cls.repeat(B, 1, 1), self.pos_emb_world(world_coord)), dim=1) + self.pos_emb.repeat(B, 1, 1)
        
        x = self.enc_3d(x + pos_emb_world)
        
        return x, coord, rot, trans
    
    def forward_3d_decode(self, x, query):
        B, N, D = x.size()
        if len(query.size()) > 3:
            query = self.forward_feature(query)
        x = self.dec_3d(x, query=query)
        return x
        
    def forward(self, x, xq=None):
        config = self.config
        B, C, H, W = x.size()
        # x = self.forward_feature(x)

        x_3d, coord, rot, trans = self.forward_3d(x)
        if xq is None:
            return (x_3d, coord, rot, trans)
        else:
            pass

class TransProjector2(nn.Module):
    def __init__(self, config):
        super(TransProjector2, self).__init__()
        self.config = config
        base_name = "transprojector2"
        if config.pred_depth:
            base_name += "-depth"
        else:
            base_name += "-xyz"
            
        if config.pred_campos_from == "both-sep":
            base_name += "-cambothsep"
        elif config.pred_campos_from == "both-avg":
            base_name += "-cambothavg"
        elif config.pred_campos_from == "cls":
            base_name += "-camcls"
        elif config.pred_campos_from == "tok":
            base_name += "-camtok"
        
    
        self.name = f"{base_name}_b16_l{config.n_layer}-{config.n_3denc_layer}-0-{config.dataset_name}"
        if config.note != "":
            self.name += f"-{config.note}"
        # config.D is 768 if we use vit-b-16
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=config.pretrained, num_classes=config.D)
        self.register_buffer("v", torch.stack([torch.tensor([i]*14, dtype=torch.float) for i in range(14)]).view(14*14, 1) - 6.5)
        self.register_buffer("u", torch.stack([torch.tensor(list(range(14)), dtype=torch.float) for i in range(14)]).view(14*14, 1) - 6.5)
        self.pos_emb = self.feature_extractor._parameters["pos_embed"] # (1, 1+14*14, 768)
        self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, 768))) # pos emb for cls token in world coord
        
        self.camera_props = CameraProps(config, pred_campos_from=config.pred_campos_from)
        if config.pred_depth:
            self.camera_coord = CameraCoordEstimator(config, dim=1) # literally a depth estimator
        else:
            self.camera_coord = CameraCoordEstimator(config, dim=3)
        self.camera_proje = CameraProjection()
        
        self.pos_emb_world = nn.Sequential(
            nn.Linear(3, config.D),
            nn.ReLU(),
            nn.Linear(config.D, config.D)
        )
        
        enc_config, dec_config = copy.deepcopy(config), copy.deepcopy(config)
        enc_config.n_layer, dec_config.n_layer = config.n_3denc_layer, config.n_3ddec_layer
        self.enc_3d = Transformer(enc_config)
        # self.cls_head = nn.Linear(config.D, 128)
        self.dec_3d = Transformer(dec_config)
        
    def _forward_features(self, x):
        m = self.feature_extractor
        x = m.patch_embed(x)
        cls_token = m.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #if m.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        #else:
        #x = torch.cat((cls_token, m.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        for i in range(self.config.n_layer):
            x = m.blocks[i](x)
        # x = m.blocks(x)
        x = m.norm(x)
        return x
        
    def forward_features(self, x):
        return self._forward_features(x)
    
    def forward_3d(self, x):
        config = self.config
        B, C, H, W = x.size()
        x = self.forward_features(x)
        # print("feature tokens", x.size())
        
        if config.pred_depth:
            depth = self.camera_coord(x[:, 1:])
            coord = uvd_to_xyz(self.u.expand_as(depth), self.v.expand_as(depth), depth)
        else:
            depth = None
            coord = self.camera_coord(x[:, 1:])
            
        rot, trans = self.camera_props(x[:, 0], x[:, 1:])
        world_coord = self.camera_proje(coord, rot, trans)
        pos_emb_world = torch.cat((self.pos_emb_world_cls.repeat(B, 1, 1), self.pos_emb_world(world_coord)), dim=1) + self.pos_emb.repeat(B, 1, 1)
        
        x = self.enc_3d(x + pos_emb_world)
        
        return x, depth, coord, rot, trans
    
    def forward_3d_decode(self, x, query):
        B, N, D = x.size()
        if len(query.size()) > 3:
            query = self.forward_feature(query)
        x = self.dec_3d(x, query=query)
        return x
        
    def forward(self, x, xq=None):
        config = self.config
        B, C, H, W = x.size()
        # x = self.forward_feature(x)

        x_3d, depth, coord, rot, trans = self.forward_3d(x)
        if xq is None:
            return (x_3d, depth, coord, rot, trans)
        else:
            pass
        

class TransProjector2_MLP(nn.Module):
    def __init__(self, config):
        super(TransProjector2_MLP, self).__init__()
        self.config = config
        base_name = "transprojector2_mlp"
        if config.pred_depth:
            base_name += "-depth"
        else:
            base_name += "-xyz"
            
        if config.pred_campos_from == "both-sep":
            base_name += "-cambothsep"
        elif config.pred_campos_from == "both-avg":
            base_name += "-cambothavg"
        elif config.pred_campos_from == "cls":
            base_name += "-camcls"
        elif config.pred_campos_from == "tok":
            base_name += "-camtok"
        
    
        self.name = f"{base_name}_b16_l{config.n_layer}-{config.n_3denc_layer}-0-{config.dataset_name}"
        if config.note != "":
            self.name += f"-{config.note}"
        # config.D is 768 if we use vit-b-16
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=config.pretrained, num_classes=config.D)
        self.register_buffer("v", torch.stack([torch.tensor([i]*14, dtype=torch.float) for i in range(14)]).view(14*14, 1) - 6.5)
        self.register_buffer("u", torch.stack([torch.tensor(list(range(14)), dtype=torch.float) for i in range(14)]).view(14*14, 1) - 6.5)
        self.pos_emb = self.feature_extractor._parameters["pos_embed"] # (1, 1+14*14, 768)
        self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, 768))) # pos emb for cls token in world coord
        
        self.mlp = nn.Sequential(
            nn.Linear(config.D, config.D),
            nn.Linear(config.D, config.D),
            nn.Linear(config.D, config.D),
            nn.Linear(config.D, config.D),
            nn.Linear(config.D, config.D),
        )
        
        self.camera_props = CameraProps(config, pred_campos_from=config.pred_campos_from)
        if config.pred_depth:
            self.camera_coord = CameraCoordEstimator(config, dim=1) # literally a depth estimator
        else:
            self.camera_coord = CameraCoordEstimator(config, dim=3)
        self.camera_proje = CameraProjection()
        
        self.pos_emb_world = nn.Sequential(
            nn.Linear(3, config.D),
            nn.ReLU(),
            nn.Linear(config.D, config.D)
        )
        
        enc_config, dec_config = copy.deepcopy(config), copy.deepcopy(config)
        enc_config.n_layer, dec_config.n_layer = config.n_3denc_layer, config.n_3ddec_layer
        self.enc_3d = Transformer(enc_config)
        # self.cls_head = nn.Linear(config.D, 128)
        self.dec_3d = Transformer(dec_config)
        
    def _forward_features(self, x):
        m = self.feature_extractor
        x = m.patch_embed(x)
        cls_token = m.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #if m.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        #else:
        #x = torch.cat((cls_token, m.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        for i in range(self.config.n_layer):
            x = m.blocks[i](x)
        # x = m.blocks(x)
        x = m.norm(x)
        return x
        
    def forward_features(self, x):
        return self._forward_features(x)
    
    def forward_3d(self, x):
        config = self.config
        B, C, H, W = x.size()
        x = self.forward_features(x)
        # print("feature tokens", x.size())
        
        x = self.enc_3d(x + self.mlp(x))
        
        return x, None, None, None, None
    
    def forward_3d_decode(self, x, query):
        B, N, D = x.size()
        if len(query.size()) > 3:
            query = self.forward_feature(query)
        x = self.dec_3d(x, query=query)
        return x
        
    def forward(self, x, xq=None):
        config = self.config
        B, C, H, W = x.size()
        # x = self.forward_feature(x)

        x_3d, depth, coord, rot, trans = self.forward_3d(x)
        if xq is None:
            return (x_3d, depth, coord, rot, trans)
        else:
            pass

class TransProjector2_Cat(nn.Module):
    def __init__(self, config):
        super(TransProjector2_Cat, self).__init__()
        self.config = config
        base_name = "transprojector2_cat"
        if config.pred_depth:
            base_name += "-depth"
        else:
            base_name += "-xyz"
            
        if config.pred_campos_from == "both-sep":
            base_name += "-cambothsep"
        elif config.pred_campos_from == "both-avg":
            base_name += "-cambothavg"
        elif config.pred_campos_from == "cls":
            base_name += "-camcls"
        elif config.pred_campos_from == "tok":
            base_name += "-camtok"
        
    
        self.name = f"{base_name}_b16_l{config.n_layer}-{config.n_3denc_layer}-0-{config.dataset_name}"
        if config.note != "":
            self.name += f"-{config.note}"
        # config.D is 768 if we use vit-b-16
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=config.pretrained, num_classes=config.D)
        self.register_buffer("v", torch.stack([torch.tensor([i]*14, dtype=torch.float) for i in range(14)]).view(14*14, 1) - 6.5)
        self.register_buffer("u", torch.stack([torch.tensor(list(range(14)), dtype=torch.float) for i in range(14)]).view(14*14, 1) - 6.5)
        self.pos_emb = self.feature_extractor._parameters["pos_embed"] # (1, 1+14*14, 768)
        self.pos_emb_world_cls = nn.Parameter(torch.randn((1, 1, 768))) # pos emb for cls token in world coord
        
        self.camera_props = CameraProps(config, pred_campos_from=config.pred_campos_from)
        if config.pred_depth:
            self.camera_coord = CameraCoordEstimator(config, dim=1) # literally a depth estimator
        else:
            self.camera_coord = CameraCoordEstimator(config, dim=3)
        self.camera_proje = CameraProjection()
        
        # self.pos_emb_world = nn.Sequential(
        #     nn.Linear(3, config.D),
        #     nn.ReLU(),
        #     nn.Linear(config.D, config.D)
        # )
        
        self.proj_world = nn.Sequential(
            nn.Linear(config.embed_dim+3, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        enc_config, dec_config = copy.deepcopy(config), copy.deepcopy(config)
        enc_config.n_layer, dec_config.n_layer = config.n_3denc_layer, config.n_3ddec_layer
        self.enc_3d = Transformer(enc_config)
        # self.cls_head = nn.Linear(config.D, 128)
        self.dec_3d = Transformer(dec_config)
        
    def _forward_features(self, x):
        m = self.feature_extractor
        x = m.patch_embed(x)
        cls_token = m.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #if m.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        #else:
        #x = torch.cat((cls_token, m.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = m.pos_drop(x + m.pos_embed)
        for i in range(self.config.n_layer):
            x = m.blocks[i](x)
        # x = m.blocks(x)
        x = m.norm(x)
        return x
        
    def forward_features(self, x):
        return self._forward_features(x)
    
    def forward_3d(self, x):
        config = self.config
        B, C, H, W = x.size()
        x = self.forward_features(x)
        # print("feature tokens", x.size())
        
        if config.pred_depth:
            depth = self.camera_coord(x[:, 1:])
            coord = uvd_to_xyz(self.u.expand_as(depth), self.v.expand_as(depth), depth)
        else:
            depth = None
            coord = self.camera_coord(x[:, 1:])
            
        rot, trans = self.camera_props(x[:, 0], x[:, 1:])
        world_coord = self.camera_proje(coord, rot, trans)
        x_token = self.proj_world(torch.cat((x[:, 1:, :], world_coord), dim=-1))
        x_token = torch.cat((self.pos_emb_world_cls.repeat(B, 1, 1), x_token), dim=1) 
        
        x = self.enc_3d(x + x_token)
        
        return x, depth, world_coord, rot, trans
    
    def forward_3d_decode(self, x, query):
        B, N, D = x.size()
        if len(query.size()) > 3:
            query = self.forward_feature(query)
        x = self.dec_3d(x, query=query)
        return x
        
    def forward(self, x, xq=None):
        config = self.config
        B, C, H, W = x.size()
        # x = self.forward_feature(x)

        x_3d, depth, world_coord, rot, trans = self.forward_3d(x)
        if xq is None:
            return (x_3d, depth, world_coord, rot, trans)
        else:
            pass
        
        
if __name__ == '__main__':
    from utils import Config
    config = Config(D=768, img_size=(3, 224, 224))
    model = TransProjector(config)
    x = torch.randn((6,3,224,224))
    y = model.forward(x)
    print (y.size())
