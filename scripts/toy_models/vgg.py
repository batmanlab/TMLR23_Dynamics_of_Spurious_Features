'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, num_channels=3):
        super(VGG, self).__init__()
        self.num_channels = num_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.gradients = None # required for custom_forwardpass1
        
    # ==================================GRADCAM on Intermediate Layers using KNN output==========================
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def l1_norm(self, X_i, X_j):
        return (abs(X_i - X_j)).sum(-1)

    def l2_norm(self, X_i, X_j):
        return ((X_i - X_j) ** 2).sum(-1)

    def gradcam_forward(self, x, nbr_feats, nbr_labels, layer_id, s):
        x_feat = self.features[:(layer_id+1)](x)
        _ = x_feat.register_hook(self.activations_hook)
        x_feat_1d = torch.reshape(x_feat, (x_feat.shape[0],-1))
        nr = 0
        dr = 0
        pred_cls = int(nbr_labels.mode()[0]) # predicted class based, argmax of the KNN
        for nbr in nbr_feats[nbr_labels==pred_cls]:    
            nr = nr + torch.exp(-self.l1_norm(x_feat_1d,nbr)/s)
        for nbr in nbr_feats:    
            dr = dr + torch.exp(-self.l1_norm(x_feat_1d,nbr)/s)
        out = nr/dr

        return out, x_feat  # probability value
    # ==================================end of GRADCAM code======================================

    def forward(self, x):
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



def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()