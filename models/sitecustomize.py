"""Global safeguards to enforce the no-sic_triton policy for this run."""

from __future__ import annotations

import importlib.abc
import sys


class _SicTritionBlocker(importlib.abc.MetaPathFinder):
    """Meta-path hook that blocks any module whose fullname references sic_triton."""

    def find_spec(self, fullname, path, target=None):  # noqa: D401,W0613
        if "sic_triton" in fullname:
            raise RuntimeError("sic_triton is forbidden")
        return None


if not any(isinstance(hook, _SicTritionBlocker) for hook in sys.meta_path):
    sys.meta_path.insert(0, _SicTritionBlocker())


def _ensure_torchvision_stub() -> None:
    if "torchvision" in sys.modules:
        return

    import types
    import importlib.machinery

    def _make_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return mod

    tv = _make_module("torchvision")
    transforms = _make_module("torchvision.transforms")

    class _InterpolationMode(types.SimpleNamespace):
        NEAREST = 0
        NEAREST_EXACT = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 6

    transforms.InterpolationMode = _InterpolationMode
    transforms.functional = _make_module("torchvision.transforms.functional")

    class _IdentityTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    class _Compose:
        def __init__(self, transforms_list):
            self.transforms = list(transforms_list or [])

        def __call__(self, x):
            for transform in self.transforms:
                x = transform(x)
            return x

    transforms.Compose = _Compose
    for _name in [
        "ToTensor",
        "Normalize",
        "RandomResizedCrop",
        "CenterCrop",
        "Resize",
        "ColorJitter",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomErasing",
        "RandomApply",
        "RandomRotation",
        "GaussianBlur",
        "RandomAffine",
        "RandomPerspective",
        "RandomGrayscale",
        "AutoAugment",
        "RandAugment",
    ]:
        if not hasattr(transforms, _name):
            setattr(transforms, _name, _IdentityTransform)

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = transforms.functional
    for sub in ["datasets", "io", "models", "ops", "utils"]:
        sys.modules[f"torchvision.{sub}"] = _make_module(f"torchvision.{sub}")

    datasets = sys.modules["torchvision.datasets"]
    for _name in [
        "CIFAR100",
        "CIFAR10",
        "MNIST",
        "KMNIST",
        "FashionMNIST",
        "ImageFolder",
    ]:
        if not hasattr(datasets, _name):
            setattr(datasets, _name, type(_name, (), {}))


def _ensure_optimum_stub() -> None:
    if "optimum.bettertransformer" in sys.modules:
        return

    import types

    module = types.ModuleType("optimum.bettertransformer")

    class _BetterTransformer:  # noqa: WPS431 - simple shim
        @staticmethod
        def transform(model, **kwargs):  # noqa: WPS110
            return model

    module.BetterTransformer = _BetterTransformer
    sys.modules["optimum.bettertransformer"] = module


# Ensure HuggingFace transformers exposes LLaMA classes needed by third-party libs (e.g. AirLLM).
try:
    import torchvision as _tv  # import real torchvision first; _ensure_torchvision_stub() is a no-op if already in sys.modules
    _ensure_torchvision_stub()
    _ensure_optimum_stub()

    import transformers  # noqa: WPS433  # Heavy import happens once at interpreter start.

    from transformers.models.llama.configuration_llama import LlamaConfig  # type: ignore
    from transformers.models.llama.modeling_llama import LlamaForCausalLM  # type: ignore
    from transformers.models.llama.modeling_llama_can import CANLlamaForCausalLM  # type: ignore

    setattr(transformers, "LlamaConfig", LlamaConfig)
    setattr(transformers, "LlamaForCausalLM", LlamaForCausalLM)
    setattr(transformers, "CANLlamaForCausalLM", CANLlamaForCausalLM)
except Exception:  # pragma: no cover - best effort patching
    pass
