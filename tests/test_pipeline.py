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


def test_similarity_metrics_are_sane():
    from evaluation.similarity import compare_logits

    a = torch.tensor([[4.0, 1.0], [1.0, 4.0]])
    b = torch.tensor([[4.0, 1.0], [1.0, 4.0]])
    metrics = compare_logits(a, b)
    assert metrics["prediction_agreement"] == 1.0
    assert abs(metrics["js_divergence"]) < 1e-7
    assert abs(metrics["probability_mae"]) < 1e-7
    assert abs(metrics["logit_cosine_similarity"] - 1.0) < 1e-6


def test_conflicting_gradient_projection():
    from unlearning.combined import project_conflicting_gradient

    retain = [torch.tensor([1.0, 0.0])]
    forget = [torch.tensor([-1.0, 1.0])]
    projected, applied, dot = project_conflicting_gradient(forget, retain)

    assert applied is True
    assert dot < 0
    assert abs(torch.dot(projected[0], retain[0]).item()) < 1e-6


def test_non_conflicting_gradient_is_unchanged():
    from unlearning.combined import project_conflicting_gradient

    retain = [torch.tensor([1.0, 0.0])]
    forget = [torch.tensor([1.0, 1.0])]
    projected, applied, dot = project_conflicting_gradient(forget, retain)

    assert applied is False
    assert dot > 0
    assert torch.allclose(projected[0], forget[0])
