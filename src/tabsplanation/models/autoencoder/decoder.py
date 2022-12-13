from torch import nn, Tensor


class MultiplyBy(nn.Module):
    """Multiplies all number by a constant factor.

    Examples::

     >>> m = MultiplyBy(5)
     >>> input = torch.ones((2,2,2))
     tensor([[[1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.]]])
     >>> output = m(input)
     >>> print(output)
     tensor([[[5., 5.],
              [5., 5.]],

             [[5., 5.],
              [5., 5.]]])
    """

    __constants__ = ["factor"]
    factor: int

    def __init__(self, factor: int) -> None:
        super(MultiplyBy, self).__init__()
        self.factor = factor

    def forward(self, input: Tensor) -> Tensor:
        return input * self.factor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.factor})"


# <https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py>
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.layers.forward(x)


class DecoderSigmoid(Decoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(DecoderSigmoid, self).__init__(input_dim, hidden_dim, latent_dim)

        self.layers.append(nn.Sigmoid())
        self.layers.append(MultiplyBy(50))


class DecoderNormalized(Decoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(DecoderNormalized, self).__init__(input_dim, hidden_dim, latent_dim)

        self.layers.append(nn.Tanh())
        self.layers.append(MultiplyBy(5))
