import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_model.SegFormer import mix_transformer


class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('D:/Workplace/pythonProject/coralReef/pretrained_weight/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )
            print('pretrained weight has been loaded!')

        self.classifier = nn.Linear(in_features=self.in_channels[-1], out_features=self.num_classes)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.squeeze(x)
        cls = self.classifier(x)
        return cls
