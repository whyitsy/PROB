# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import logging

def build_model(args, mode='owdetr'):
    logging.info(f'Building model with mode: {mode}')
    if mode == 'prob':
        from .prob_deformable_detr import build
    elif mode == 'innov_1':
        from .prob_deformable_detr_innov_1 import build
    elif mode == 'innov_2':
        from .prob_deformable_detr_innov_2 import build
    else:
        from .deformable_detr import build
    return build(args)