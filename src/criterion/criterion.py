def get_criterion(name: str):
    """
    Get criterion based on the name.

    Args:
        name: Name of the criterion.

    Returns:
        Criterion object.
    """
    import torch

    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif name == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Criterion {name} not found.")
