# models/__init__.py
import logging


def build_model(args, mode='owdetr'):
    logging.info('Building model with mode: %s', mode)
    if mode == 'prob':
        from .prob_deformable_detr import build
    elif mode == 'uod':
        from .prob_deformable_detr_uod import build
    else:
        from .deformable_detr import build
    return build(args)
