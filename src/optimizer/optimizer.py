def get_optimizer(name: str, *args, **kwargs):
    """
    Get optimizer based on the name.

    Args:
        name: Name of the optimizer.

    Returns:
        Optimizer object.
    """
    import torch

    if name == "adam":
        return torch.optim.Adam(*args, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(*args, **kwargs)
    else:
        raise ValueError(f"Optimizer {name} not found.")
