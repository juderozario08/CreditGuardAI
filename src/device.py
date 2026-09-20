import os

import torch


def get_compute_device(use_gpu: bool = False) -> tuple[torch.device, str]:
    if not use_gpu:
        torch.set_num_threads(os.cpu_count() or 4)
        return torch.device("cpu"), f"CPU ({torch.get_num_threads()} threads)"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"NVIDIA GPU ({name})"

    try:
        import torch_directml

        if torch_directml.is_available():
            dml = torch_directml.device()
            name = torch_directml.device_name(dml.index).strip()
            return dml, f"AMD GPU ({name})"
    except ImportError:
        pass

    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return torch.device("cuda"), "AMD ROCm GPU"

    torch.set_num_threads(os.cpu_count() or 4)
    return (
        torch.device("cpu"),
        f"CPU (GPU unavailable, falling back to {torch.get_num_threads()} threads)",
    )
