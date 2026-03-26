# models/__init__.py
import logging


def build_model(args, mode='owdetr'):
    logging.info(f'Building model with mode: {mode}')
    if mode == 'prob':
        from .prob_deformable_detr import build
    elif mode == 'uod':
        from .prob_deformable_detr_uod import build
    elif mode == 'uod_paramless':
        from .prob_deformable_detr_uod_paramless import build
    else:
        from .deformable_detr import build
    return build(args)
