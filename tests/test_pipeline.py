import torch

from models import CIFARResNet18
from unlearning import kd_loss


def test_model_output_shape():
    model = CIFARResNet18(10)
    x = torch.randn(4, 3, 32, 32)
    assert model(x).shape == (4, 10)


def test_kd_loss_is_finite():
    a = torch.randn(4, 10)
    b = torch.randn(4, 10)
    loss = kd_loss(a, b, 4.0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
