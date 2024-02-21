"""
This module holds the function that retrieves a model object based on the given name and kwargs.
"""
import torch
from models.custom_models.path_infomax import Path_Infomax
from models.custom_models.dgi import DGI

def get_model(model_name: str, **kwargs):
    """Retrieves a model based on the model name.

    Parameters
    ----------
    model_name (str) : The model name 
    kwargs (dict): The dictionary of kwargs used to initialize the model.

    Returns:
    model: An initialized PyTorch model
    """
    if model_name == "Path_Infomax":
        model = Path_Infomax(**kwargs)
    elif model_name == "dgi":
        model = DGI(**kwargs)
    else:
        raise NotImplementedError(f" Model '{model_name}' is not implemented.")
    return model