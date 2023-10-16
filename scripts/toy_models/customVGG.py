'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class customVGG(nn.Module):
    def __init__(self, vgg_name, nbr_feats=None, nbr_labels=None, layer_id=None, s=None, num_channels=3):
        super(customVGG, self).__init__()
        self.num_channels = num_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        # required for custom forward pass:
        self.nbr_feats = nbr_feats
        self.nbr_labels = nbr_labels
        self.layer_id = layer_id
        self.s = s

    def l1_norm(self, X_i, X_j):
        return (abs(X_i - X_j)).sum(-1)

    def l2_norm(self, X_i, X_j):
        return ((X_i - X_j) ** 2).sum(-1)

    def forward(self, x):
        x_feat = self.features[:(self.layer_id+1)](x)
        x_feat = torch.nn.functional.interpolate(x_feat,(8,8))
        x_feat_1d = torch.reshape(x_feat, (x_feat.shape[0],-1))
        nr = 0
        dr = 0
        pred_cls = int(self.nbr_labels.mode()[0]) # predicted class based, argmax of the KNN
        for nbr in self.nbr_feats[self.nbr_labels==pred_cls]:    
            nr = nr + torch.exp(-self.l1_norm(x_feat_1d,nbr)/self.s)
        for nbr in self.nbr_feats:    
            dr = dr + torch.exp(-self.l1_norm(x_feat_1d,nbr)/self.s)
        out = nr/dr

        return out, x_feat  # probability value
    
    def simpleForward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.num_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)