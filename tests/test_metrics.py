import torch

from tabsplanation.metrics import mmd


def original_mmd(x, y, kernel, device="cpu"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


def test_mmd_still_correct():
    """My modified MMD that works with unequal sample sizes should
    get the same result as the original when the samples do have equal size."""
    from torch.distributions.multivariate_normal import MultivariateNormal

    m = 5
    x_mean = torch.zeros(2) + 1
    y_mean = torch.zeros(2)
    x_cov = 2 * torch.eye(2)  # IMPORTANT: Covariance matrices must be positive definite
    y_cov = 3 * torch.eye(2) - 1
    px = MultivariateNormal(x_mean, x_cov)
    qy = MultivariateNormal(y_mean, y_cov)

    device = "cpu"
    x = px.sample([m]).to(device)
    y = qy.sample([m]).to(device)

    assert torch.isclose(mmd(x, y, "rbf"), original_mmd(x, y, "rbf"))
