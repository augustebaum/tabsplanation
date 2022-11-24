from torch import nn


# <https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py>
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.layers.forward(x)
