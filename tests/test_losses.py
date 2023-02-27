import torch

from src.tabsplanation.explanations.losses import GeneralValidityLoss
from src.tabsplanation.types import Tensor


def test_loss():
    loss = GeneralValidityLoss(kind="prb", classes="others")

    logits: Tensor[2, 3] = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    source = torch.tensor([0, 0])
    target = torch.tensor([2, 1])

    actual = loss(logits, source, target)
    expected = torch.tensor([[0.3305], [-0.5105]])
    assert actual.equal(expected)
