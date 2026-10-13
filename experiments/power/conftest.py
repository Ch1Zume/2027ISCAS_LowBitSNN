"""Power activity tests use the inference-only execution contract."""
import pytest
import torch


@pytest.fixture(autouse=True)
def inference_only():
    with torch.no_grad():
        yield
