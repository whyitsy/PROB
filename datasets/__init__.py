# ------------------------------------------------------------------------
# Detection-only dataset helpers used by the current PROB/UOD workflow.
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    return None


def build_dataset(image_set, args):
    dataset_file = getattr(args, 'dataset_file', None)
    if dataset_file == 'coco':
        try:
            from .coco import build as build_coco  # legacy API, lazily imported
        except ImportError as exc:
            raise ImportError(
                'datasets.coco no longer exposes build(). In the current detection-only '
                'workflow, main_open_world.py should use OWDetection + make_coco_transforms '
                'directly. If you still need the legacy COCO builder, re-introduce a '
                'compatible build() function in datasets/coco.py.'
            ) from exc
        return build_coco(image_set, args)

    if dataset_file == 'coco_panoptic':
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    raise ValueError(f'dataset {dataset_file} not supported')