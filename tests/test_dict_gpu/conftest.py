import pytest, torch

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA")

@pytest.fixture
def has_cuda():
    return torch.cuda.is_available()
