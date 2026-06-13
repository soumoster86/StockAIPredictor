"""Test setup. CI installs the real dependencies (torch, yfinance, etc.)
from requirements.txt. In minimal environments, lightweight stubs are
injected so the PURE-LOGIC tests still run; tests that genuinely need the
real packages detect STUB_MODE and skip themselves."""
import sys
import types

STUB_MODE = False


def _stub_torch():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    optim = types.ModuleType("torch.optim")

    class Module:
        pass

    class Tensor:
        pass

    for name in ["Sequential", "Linear", "ReLU", "BCEWithLogitsLoss", "LSTM", "GRU"]:
        setattr(nn, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    nn.Module = Module
    torch.nn = nn
    torch.optim = optim
    torch.Tensor = Tensor
    torch.manual_seed = lambda s: None
    torch.tensor = lambda *a, **k: None
    torch.is_tensor = lambda x: False
    torch.no_grad = lambda: None
    torch.sigmoid = lambda x: None
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.optim": optim})


try:
    import torch  # noqa: F401
except ImportError:
    _stub_torch()
    STUB_MODE = True

try:
    import yfinance  # noqa: F401
except ImportError:
    yf = types.ModuleType("yfinance")
    yf.download = lambda *a, **k: None
    sys.modules["yfinance"] = yf
    STUB_MODE = True

# Make the app modules importable when running from the repo root or tests/
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
