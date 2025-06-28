import torch
import torch.nn as nn
from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, I2TLoss, InterMeanLoss


@ADAPTATION_REGISTRY.register()
class OURS(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)
        # setup loss function
        self.softmax_entropy = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

    
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        with torch.cuda.amp.autocast():
            logits, _, text_features, img_pre_features, text_pre_features = self.model(imgs_test, return_features=True)
        
        loss = self.softmax_entropy(logits).mean(0)
        i2t_loss = self.i2t_loss(logits, img_pre_features, text_features)
        inter_mean_loss = self.inter_mean_loss(logits, img_pre_features)
        loss -= i2t_loss
        loss -= inter_mean_loss

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return logits.detach()

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        # re-enable parameters
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names