import torch
import torch.nn as nn
import torchvision.models as models

class blip_resnet(nn.Module):
    def __init__(self, args):
        super(blip_resnet, self).__init__()
        model = getattr(models, 'resnet101')(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        map_size = int(args.image_size / 32)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=map_size, stride=1, padding=0)
    def forward(self, x):
        patch_feats = self.model(x)
        avg_feats = self.avg_fnt(patch_feats).flatten(1)
        batch_size, feat_size, _, _ = patch_feats.shape
        # NxLxD
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

