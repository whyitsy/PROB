import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _load_repo_parser() -> argparse.ArgumentParser:
    mod = importlib.import_module('main_open_world')
    if not hasattr(mod, 'get_args_parser'):
        raise RuntimeError('main_open_world.py does not expose get_args_parser()')
    base = mod.get_args_parser()
    parser = argparse.ArgumentParser(
        'PROB/UOD evaluation debug collector',
        parents=[base],
        conflict_handler='resolve'
    )
    parser.add_argument('--resume', required=True, help='Checkpoint path')
    parser.add_argument('--output_dir_debug', default='eval_debug_outputs', help='Directory to save score records and debug npz files')
    parser.add_argument('--split', default=None, help='Dataset split to evaluate, defaults to args.test_set')
    parser.add_argument('--max_batches', type=int, default=-1, help='Optional limit for quick debugging')
    parser.add_argument('--dump_debug_npz_every', type=int, default=50, help='Save one NPZ every N batches when tensors are available')
    parser.add_argument('--known_class_cutoff', type=int, default=None, help='Labels < cutoff are treated as known; defaults to num_classes-1')
    parser.add_argument('--unknown_label_id', type=int, default=None, help='Optional explicit unknown label id')
    parser.add_argument('--match_iou', type=float, default=0.5, help='IoU threshold for record matching')
    parser.add_argument('--score_thresh', type=float, default=0.0, help='Minimum postprocessed score to record')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    return parser


def _build_model(args: argparse.Namespace):
    models_mod = importlib.import_module('models')
    built = models_mod.build_model(args)
    if len(built) >= 3:
        return built[0], built[1], built[2]
    raise RuntimeError('Unexpected build_model(args) return value')


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _extract_query_debug(outputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    debug: Dict[str, np.ndarray] = {}
    def _prob(x: torch.Tensor) -> torch.Tensor:
        if float(x.min()) >= 0.0 and float(x.max()) <= 1.0:
            return x
        return x.sigmoid()

    if 'pred_obj' in outputs:
        debug['obj_prob'] = _prob(outputs['pred_obj'][0]).squeeze(-1).detach().cpu().numpy()
    if 'pred_known' in outputs:
        kp = _prob(outputs['pred_known'][0]).squeeze(-1)
        debug['known_prob'] = kp.detach().cpu().numpy()
        debug['unknown_prob'] = (1.0 - kp).detach().cpu().numpy()
    if 'pred_logits' in outputs:
        cls_prob = outputs['pred_logits'][0].sigmoid()
        debug['known_max'] = cls_prob.max(dim=-1).values.detach().cpu().numpy()
    if 'debug' in outputs and isinstance(outputs['debug'], dict):
        for k, v in outputs['debug'].items():
            if torch.is_tensor(v):
                debug[k] = v.detach().cpu().numpy()
    return debug


def _build_dataset_and_loader(args: argparse.Namespace):
    datasets_mod = importlib.import_module('datasets')
    misc = importlib.import_module('util.misc')
    split = args.split or getattr(args, 'test_set', None)
    if split is None:
        raise RuntimeError('No split provided and args.test_set is missing')
    dataset = datasets_mod.build_dataset(image_set=split, args=args)
    loader = DataLoader(dataset,
                        batch_size=getattr(args, 'batch_size', 1),
                        shuffle=False,
                        num_workers=getattr(args, 'num_workers', 2),
                        collate_fn=misc.collate_fn)
    return dataset, loader


def _label_type(label: int, cutoff: int, unknown_label_id: Optional[int]) -> str:
    if unknown_label_id is not None:
        return 'unknown' if label == unknown_label_id else 'known'
    return 'unknown' if label >= cutoff else 'known'


def _image_id_from_target(t: Dict[str, Any], fallback: int) -> int:
    if 'image_id' not in t:
        return fallback
    v = t['image_id']
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def main() -> None:
    parser = _load_repo_parser()
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    out_dir = Path(args.output_dir_debug)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, postprocessors = _build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()

    _, data_loader = _build_dataset_and_loader(args)
    cutoff = args.known_class_cutoff or (max(1, int(getattr(args, 'num_classes', 81)) - 1))

    records: List[Dict[str, Any]] = []
    image_counter = 0

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(data_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            samples = samples.to(device)
            outputs = model(samples)

            target_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
            results = postprocessors['bbox'](outputs, target_sizes)
            debug_np = _extract_query_debug(outputs)

            for i, (result, target) in enumerate(zip(results, targets)):
                image_id = _image_id_from_target(target, image_counter)
                image_counter += 1
                scores = result['scores'].detach().cpu()
                labels = result['labels'].detach().cpu()
                boxes = result['boxes'].detach().cpu()

                keep = scores >= args.score_thresh
                scores = scores[keep]
                labels = labels[keep]
                boxes = boxes[keep]

                gt_boxes = target['boxes'].detach().cpu()
                gt_labels = target['labels'].detach().cpu()
                orig_h = int(target['orig_size'][0].item())
                orig_w = int(target['orig_size'][1].item())
                gt_boxes = _cxcywh_to_xyxy(gt_boxes)
                gt_boxes = gt_boxes * torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
                ious = _box_iou(boxes, gt_boxes) if len(boxes) > 0 and len(gt_boxes) > 0 else torch.zeros((len(boxes), len(gt_boxes)))

                for det_idx, (score, label, box) in enumerate(zip(scores, labels, boxes)):
                    pred_type = _label_type(int(label), cutoff, args.unknown_label_id)
                    if ious.numel() > 0:
                        best_iou, best_gt_idx = ious[det_idx].max(dim=0)
                        best_iou_f = float(best_iou.item())
                        matched = best_iou_f >= args.match_iou
                        gt_label = int(gt_labels[best_gt_idx].item())
                        matched_gt_type = _label_type(gt_label, cutoff, args.unknown_label_id) if matched else 'none'
                    else:
                        best_iou_f = 0.0
                        matched = False
                        gt_label = -1
                        matched_gt_type = 'none'

                    records.append({
                        'image_id': image_id,
                        'det_index': det_idx,
                        'score': float(score.item()),
                        'label': int(label.item()),
                        'pred_type': pred_type,
                        'box_xyxy': [float(x) for x in box.tolist()],
                        'matched': matched,
                        'matched_gt_label': gt_label,
                        'matched_gt_type': matched_gt_type,
                        'iou': best_iou_f,
                    })

                if debug_np and args.dump_debug_npz_every > 0 and batch_idx % args.dump_debug_npz_every == 0 and i == 0:
                    np.savez_compressed(out_dir / f'debug_batch{batch_idx:05d}_img{image_id}.npz', **debug_np)

            if batch_idx % 10 == 0:
                print(f'[INFO] processed batch {batch_idx}, records={len(records)}')

    jsonl_path = out_dir / 'score_records.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    meta = {
        'num_records': len(records),
        'known_class_cutoff': cutoff,
        'unknown_label_id': args.unknown_label_id,
        'score_thresh': args.score_thresh,
        'match_iou': args.match_iou,
        'split': args.split or getattr(args, 'test_set', None),
    }
    (out_dir / 'score_records_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f'[OK] wrote {len(records)} records to {jsonl_path}')


if __name__ == '__main__':
    main()
