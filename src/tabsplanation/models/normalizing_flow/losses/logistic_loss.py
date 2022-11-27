import torch
from torch import nn


def nice_logistic_log_likelihood(h, diag):
    return torch.sum(diag) - (
        torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)), dim=1)
    )


class LogisticPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogisticPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        if self.size_average:
            return torch.mean(-nice_logistic_log_likelihood(fx, diag))
        else:
            return torch.sum(-nice_logistic_log_likelihood(fx, diag))
