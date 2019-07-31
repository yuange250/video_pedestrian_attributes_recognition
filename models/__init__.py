from __future__ import absolute_import

from models.AttrModels import AttrResNet50TP, AttrResNet503D, AttrResNet50TPBaseline

__factory = {
    'attr_resnet50tp':AttrResNet50TP,
    'attr_resnet503d':AttrResNet503D,
    'attr_resnet50tp_baseline': AttrResNet50TPBaseline
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
