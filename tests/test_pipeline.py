import torch

from datasets import LocalMNIST, make_split
from models import MNISTCNN
from unlearning import kd_loss


def test_model_output_shape():
    model = MNISTCNN(10)
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)


def test_kd_loss_is_finite():
    a = torch.randn(4, 10)
    b = torch.randn(4, 10)
    loss = kd_loss(a, b, 4.0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_mnist_idx_loader_api():
    assert callable(LocalMNIST)
    assert callable(make_split)
