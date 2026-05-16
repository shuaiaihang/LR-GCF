from torch.optim import SGD

from .dataloaders import get_acdc_loaders
from .logger import Logger
from .losses import CrossEntropyLoss_, DiceLoss_
from .metrics import Dice, Jaccard
from .models import UNetTF
from .schedulers import PolyLR

__all__ = ['build_model', 'build_dataloader', 'build_scheduler',
           'build_optimizer', 'build_logger', '_build_from_cfg']

COMPONENTS = {
    'UNetTF': UNetTF,
    'DiceLoss_': DiceLoss_,
    'CrossEntropyLoss_': CrossEntropyLoss_,
    'Dice': Dice,
    'Jaccard': Jaccard,
    'PolyLR': PolyLR,
    'SGD': SGD,
    'acdc': get_acdc_loaders,
}


def _build_from_cfg(cfg):
    if isinstance(cfg, dict):
        name = cfg['name']
        if 'kwargs' in cfg.keys() and cfg['kwargs'] is not None:
            return COMPONENTS[name](**cfg['kwargs'])
    else:
        name = cfg.name
        if hasattr(cfg, 'kwargs') and cfg.kwargs is not None:
            return COMPONENTS[name](**cfg.kwargs.__dict__)
    return COMPONENTS[name]()


def build_model(cfg):
    return _build_from_cfg(cfg)


def build_optimizer(model_parameters, cfg):
    if isinstance(cfg, dict):
        name = cfg['name']
        if 'kwargs' in cfg.keys() and cfg['kwargs'] is not None:
            kwargs = cfg['kwargs']
            kwargs['params'] = model_parameters
            return COMPONENTS[name](**kwargs)
    else:
        name = cfg.name
        if hasattr(cfg, 'kwargs') and cfg.kwargs is not None:
            kwargs = cfg.kwargs.__dict__
            kwargs['params'] = model_parameters
            return COMPONENTS[name](**kwargs)
    kwargs = {'params': model_parameters}
    return COMPONENTS[name](**kwargs)


def build_scheduler(optimizer_, cfg):
    if isinstance(cfg, dict):
        name = cfg['name']
        if 'kwargs' in cfg.keys() and cfg['kwargs'] is not None:
            kwargs = cfg['kwargs']
            kwargs['optimizer'] = optimizer_
            return COMPONENTS[name](**kwargs)
    else:
        name = cfg.name
        if hasattr(cfg, 'kwargs') and cfg.kwargs is not None:
            kwargs = cfg.kwargs.__dict__
            kwargs['optimizer'] = optimizer_
            return COMPONENTS[name](**kwargs)
    kwargs = {'optimizer': optimizer_}
    return COMPONENTS[name](**kwargs)


def build_dataloader(cfg, worker_init_fn):
    if not isinstance(cfg, dict):
        kwargs = cfg.kwargs.__dict__
    else:
        kwargs = cfg['kwargs']
    kwargs['worker_init_fn'] = worker_init_fn
    return COMPONENTS[cfg.name](**kwargs)

def build_logger(cfg):
    return Logger(**cfg.__dict__)
