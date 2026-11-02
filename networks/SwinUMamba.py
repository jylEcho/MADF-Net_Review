import importlib.util
import os


_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Swin-UMamba",
    "model.py",
)

_SPEC = importlib.util.spec_from_file_location("swin_umamba_model", _MODEL_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

Network = _MODULE.Network
