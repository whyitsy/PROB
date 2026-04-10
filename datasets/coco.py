"""Detection transforms used by PROB/UOD training, evaluation and inference.

This file no longer carries segmentation-specific dataset preparation logic.
It only exposes the image/box transform pipeline that is actually used by the
current open-world object detection workflow.
"""

import datasets.transforms as T


def build_detection_transforms(split_name: str):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    multi_scale_sizes = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    split_lower = split_name.lower()
    transform_prefix = []

    if 'train' in split_lower:
        transform_prefix.append(['train'])
        transform_prefix.append(T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(multi_scale_sizes, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(multi_scale_sizes, max_size=1333),
                ]),
            ),
            normalize,
        ]))
        return transform_prefix

    if 'ft' in split_lower:
        transform_prefix.append(['ft'])
        transform_prefix.append(T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(multi_scale_sizes, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(multi_scale_sizes, max_size=1333),
                ]),
            ),
            normalize,
        ]))
        return transform_prefix

    if 'val' in split_lower:
        transform_prefix.append(['val'])
        transform_prefix.append(T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ]))
        return transform_prefix

    if 'test' in split_lower or 'infer' in split_lower:
        transform_prefix.append(['test'])
        transform_prefix.append(T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ]))
        return transform_prefix

    raise ValueError(f'Unsupported split name: {split_name}')


# Backward-compatible alias used by the current training entry.
def make_coco_transforms(image_set):
    return build_detection_transforms(image_set)
