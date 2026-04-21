"""Independent video-only risk window detection package."""

__all__ = ["RiskWindowDetector"]


def __getattr__(name):
    if name == "RiskWindowDetector":
        from .model import RiskWindowDetector

        return RiskWindowDetector
    raise AttributeError(name)
