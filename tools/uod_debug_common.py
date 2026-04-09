import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, SequentialSampler


def _coerce_checkpoint_args(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, argparse.Namespace):
        return vars(raw)
    if isinstance(raw, dict):
        return raw
    return {}


def load_checkpoint_arg_defaults(resume_path: Optional[str]) -> Dict[str, Any]:
    if not resume_path:
        return {}
    p = Path(resume_path)
    if not p.exists():
        return {}
    try:
        checkpoint = torch.load(str(p), map_location='cpu')
    except Exception:
        return {}
    return _coerce_checkpoint_args(checkpoint.get('args'))


class _StageEpochAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(values))


def build_repo_parser(argv=None) -> argparse.ArgumentParser:
    main_mod = importlib.import_module('main_open_world')
    if not hasattr(main_mod, 'get_args_parser'):
        raise RuntimeError('main_open_world.py does not expose get_args_parser()')

    minimal = argparse.ArgumentParser(add_help=False)
    minimal.add_argument('--resume', default='')
    known, _ = minimal.parse_known_args(argv)
    ckpt_defaults = load_checkpoint_arg_defaults(known.resume)

    base = main_mod.get_args_parser()
    if ckpt_defaults:
        safe_defaults = {k: v for k, v in ckpt_defaults.items() if isinstance(k, str)}
        base.set_defaults(**safe_defaults)

    parser = argparse.ArgumentParser(
        'UOD mining debug tools',
        parents=[base],
        conflict_handler='resolve'
    )
    parser.add_argument('--resume', required=True, help='Checkpoint path')
    parser.add_argument('--split', required=True, help='Dataset split, e.g. t2_train / t2_test / t2_ft')
    parser.add_argument('--output_dir_debug', required=True, help='Directory to save raw debug outputs')
    parser.add_argument('--max_batches', type=int, default=-1, help='Optional limit for quick runs')
    parser.add_argument('--save_images_limit', type=int, default=-1, help='Optional limit on total saved images')
    parser.add_argument('--stage_epoch', default=999, action=_StageEpochAction, help='Stage-local epoch used to replay mining warmup logic')
    parser.add_argument('--stage_name', default='', help='Optional stage name tag for metadata only')
    parser.add_argument('--gt_overlap_mode', default='iou_iof', choices=['iou_iof', 'iou_only', 'none'], help='How to filter candidates against GT when replaying mining')
    parser.add_argument('--collect_query_jsonl', action='store_true', help='Also dump one JSONL row per query for ad-hoc analysis')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    return parser


def build_model_bundle(args):
    models_mod = importlib.import_module('models')
    built = models_mod.build_model(args, mode=getattr(args, 'model_type', 'uod'))
    if len(built) != 4:
        raise RuntimeError(f'Unexpected build_model(args) return size: {len(built)}')
    model, criterion, postprocessors, exemplar_selection = built
    return model, criterion, postprocessors, exemplar_selection


def build_owod_dataset(args, split: str):
    from datasets.coco import make_coco_transforms
    from datasets.torchvision_datasets.open_world import OWDetection

    return OWDetection(
        args,
        args.data_root,
        image_set=split,
        transforms=make_coco_transforms(split),
        dataset=args.dataset,
    )


def build_loader(dataset, batch_size: int, num_workers: int):
    misc = importlib.import_module('util.misc')
    sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=misc.collate_fn,
    )


def checkpoint_load_to_model(model, resume_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(resume_path, map_location='cpu')
    state = checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=False)
    return checkpoint


def to_device_targets(targets, device: torch.device):
    out = []
    for t in targets:
        item = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                item[k] = v.to(device)
            else:
                item[k] = v
        out.append(item)
    return out


def image_path_map_from_dataset(dataset):
    imgid_to_path = {}
    imgid_to_name = {}
    for img_id, path in zip(dataset.imgids, dataset.images):
        imgid_to_path[int(img_id)] = str(path)
        imgid_to_name[int(img_id)] = Path(path).name
    return imgid_to_path, imgid_to_name


def image_id_from_target(target: Dict[str, Any], fallback: int) -> int:
    v = target.get('image_id', None)
    if v is None:
        return fallback
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def scale_xyxy_to_orig(boxes_xyxy: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    return boxes_xyxy * scale


def tensor_to_list(x: torch.Tensor):
    return x.detach().cpu().tolist()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
