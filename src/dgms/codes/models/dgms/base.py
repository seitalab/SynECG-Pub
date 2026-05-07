import torch.nn as nn

class BaseDGM(nn.Module):

    def calc_loss(self, data):
        raise NotImplementedError()
    
    def forward(self, data):
        raise NotImplementedError()
    
    def generate(self, data):
        raise NotImplementedError()