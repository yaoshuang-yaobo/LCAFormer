from .LCAFormer import *


Get_Models = {
    'LCAFormer': LCAFormer

}


def get_segmentation_Model(name, **data_kwargs):
    """Segmentation Datasets"""
    return Get_Models[name](**data_kwargs)
