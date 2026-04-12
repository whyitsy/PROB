import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_CHAPTER_SPECS = {
    'chapter3': {
        'categories': ['unknown'],
        'modes': ['sampling', 'trajectory'],
        'title': 'Chapter 3 candidate galleries',
    },
    'chapter4': {
        'categories': ['known', 'unknown', 'odqe_salient'],
        'modes': ['gate', 'joint', 'trajectory'],
        'title': 'Chapter 4 candidate galleries',
    },
}


def parse_csv_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def infer_mode(path_str):
    name = Path(path_str).name.lower()
    if 'joint_mechanism' in name:
        return 'joint'
    if 'trajectory' in name:
        return 'trajectory'
    if 'sampling' in name:
        return 'sampling'
    if 'gate_curve' in name or 'gate_heatmap' in name:
        return 'gate'
    return 'other'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def build_representative_lookup(rep_manifest):
    lookup = {}
    categories = rep_manifest.get('categories', {}) if isinstance(rep_manifest, dict) else {}
    for category, entries in categories.items():
        for entry in entries:
            key = (category, int(entry['sample_index']), int(entry['query_index']))
            lookup[key] = entry
    return lookup


def enrich_cases(render_manifest, representative_lookup):
    enriched = []
    for case in render_manifest.get('cases', []):
        category = case.get('category', 'unknown')
        sample_index = int(case['sample_index'])
        query_index = int(case['query_index'])
        key = (category, sample_index, query_index)
        rep_entry = representative_lookup.get(key, {})
        rendered_by_mode = defaultdict(list)
        for path_str in case.get('rendered_files', []):
            rendered_by_mode[infer_mode(path_str)].append(path_str)
        enriched.append({
            'category': category,
            'sample_index': sample_index,
            'image_id': int(case.get('image_id', sample_index)),
            'query_index': query_index,
            'query_kind': case.get('query_kind', category),
            'case_dir': case.get('case_dir'),
            'rendered_files': case.get('rendered_files', []),
            'rendered_by_mode': dict(rendered_by_mode),
            'category_score': float(rep_entry.get('category_score', -1.0)),
            'obj_prob': rep_entry.get('obj_prob'),
            'unknown_prob': rep_entry.get('unknown_prob'),
            'max_known': rep_entry.get('max_known'),
            'known_score': rep_entry.get('known_score'),
            'unknown_score': rep_entry.get('unknown_score'),
            'gate_mean': rep_entry.get('gate_mean'),
            'gate_depth_delta': rep_entry.get('gate_depth_delta'),
        })
    enriched.sort(key=lambda item: (item['category'], -item['category_score'], item['sample_index'], item['query_index']))
    return enriched


def add_label_bar(tile, label_lines, tile_size):
    canvas = Image.new('RGB', (tile_size, tile_size), color=(20, 20, 20))
    tile = tile.resize((tile_size, tile_size - 58))
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, tile_size - 58, tile_size, tile_size], fill=(18, 18, 18))
    draw.multiline_text((8, tile_size - 54), '\n'.join(label_lines[:3]), fill=(255, 255, 255), spacing=2)
    return canvas


def make_tile(image_path, entry, mode, tile_size):
    with Image.open(image_path) as image:
        tile = image.convert('RGB')
    score = entry.get('category_score', -1.0)
    score_str = 'n/a' if score < 0 else f'{score:.3f}'
    label_lines = [
        f"{entry['category']} | {mode}",
        f"img {entry['image_id']} q{entry['query_index']} s={score_str}",
        f"obj={_fmt(entry.get('obj_prob'))} unk={_fmt(entry.get('unknown_prob'))} gate={_fmt(entry.get('gate_mean'))}",
    ]
    return add_label_bar(tile, label_lines, tile_size)


def _fmt(value):
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.3f}'
    except Exception:
        return 'n/a'


def build_sheet(items, output_path, title, tile_size=340, cols=3):
    if not items:
        return None
    rows = math.ceil(len(items) / cols)
    title_h = 56
    canvas = Image.new('RGB', (cols * tile_size, rows * tile_size + title_h), color=(12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.size[0], title_h], fill=(28, 28, 28))
    draw.text((12, 16), title, fill=(255, 255, 255))
    for idx, (image_path, entry, mode) in enumerate(items):
        tile = make_tile(image_path, entry, mode, tile_size)
        row = idx // cols
        col = idx % cols
        canvas.paste(tile, (col * tile_size, title_h + row * tile_size))
    canvas.save(output_path)
    return output_path


def select_board_items(entries, categories, modes, per_group_limit):
    items = []
    filtered = [entry for entry in entries if entry['category'] in categories]
    grouped = defaultdict(list)
    for entry in filtered:
        grouped[entry['category']].append(entry)
    for category in categories:
        ranked = sorted(grouped.get(category, []), key=lambda item: item['category_score'], reverse=True)[:per_group_limit]
        for entry in ranked:
            for mode in modes:
                paths = entry.get('rendered_by_mode', {}).get(mode, [])
                if paths:
                    items.append((paths[0], entry, mode))
    return items


def write_markdown_index(output_path, entries, board_paths, base_dir):
    lines = ['# Figure Atlas', '']
    if board_paths:
        lines += ['## Gallery Boards', '']
        for title, path in board_paths:
            rel = Path(path).relative_to(base_dir)
            lines.append(f'- [{title}]({rel.as_posix()})')
        lines.append('')

    by_category = defaultdict(list)
    for entry in entries:
        by_category[entry['category']].append(entry)

    lines += ['## Rendered Cases', '']
    for category, category_entries in by_category.items():
        lines.append(f'### {category}')
        for entry in sorted(category_entries, key=lambda item: item['category_score'], reverse=True):
            score = entry.get('category_score', -1.0)
            score_str = 'n/a' if score < 0 else f'{score:.3f}'
            lines.append(
                f"- image {entry['image_id']} / query {entry['query_index']} / score {score_str} / obj {_fmt(entry.get('obj_prob'))} / unk {_fmt(entry.get('unknown_prob'))} / gate {_fmt(entry.get('gate_mean'))}"
            )
            for mode in ['sampling', 'gate', 'joint', 'trajectory']:
                for path_str in entry.get('rendered_by_mode', {}).get(mode, []):
                    rel = Path(path_str).relative_to(base_dir)
                    lines.append(f'  - [{mode}]({rel.as_posix()})')
        lines.append('')

    output_path.write_text('\n'.join(lines), encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser('Automatic gallery organizer for rendered thesis figures')
    parser.add_argument('--render_manifest', required=True, type=str, help='path to render_manifest.json')
    parser.add_argument('--representative_manifest', default=None, type=str, help='optional path to representative_case_manifest.json')
    parser.add_argument('--output_dir', required=True, type=str, help='directory where organized atlas will be written')
    parser.add_argument('--tile_size', default=340, type=int)
    parser.add_argument('--cols', default=3, type=int)
    parser.add_argument('--per_group_limit', default=3, type=int)
    parser.add_argument('--chapter3_categories', default='unknown', type=str)
    parser.add_argument('--chapter3_modes', default='sampling,trajectory', type=str)
    parser.add_argument('--chapter4_categories', default='known,unknown,odqe_salient', type=str)
    parser.add_argument('--chapter4_modes', default='gate,joint,trajectory', type=str)
    return parser


def main(args):
    render_manifest = load_json(args.render_manifest)
    representative_lookup = {}
    if args.representative_manifest:
        representative_lookup = build_representative_lookup(load_json(args.representative_manifest))
    entries = enrich_cases(render_manifest, representative_lookup)
    if not entries:
        raise RuntimeError('No rendered cases found in render manifest.')

    output_dir = Path(args.output_dir)
    boards_dir = output_dir / 'boards'
    boards_dir.mkdir(parents=True, exist_ok=True)

    chapter_specs = {
        'chapter3': {
            'categories': parse_csv_list(args.chapter3_categories),
            'modes': parse_csv_list(args.chapter3_modes),
            'title': DEFAULT_CHAPTER_SPECS['chapter3']['title'],
        },
        'chapter4': {
            'categories': parse_csv_list(args.chapter4_categories),
            'modes': parse_csv_list(args.chapter4_modes),
            'title': DEFAULT_CHAPTER_SPECS['chapter4']['title'],
        },
    }

    board_paths = []

    all_categories = sorted({entry['category'] for entry in entries})
    for category in all_categories:
        for mode in ['sampling', 'gate', 'joint', 'trajectory']:
            items = select_board_items(entries, [category], [mode], args.per_group_limit)
            if not items:
                continue
            board_path = boards_dir / f'{category}_{mode}_sheet.png'
            title = f'{category} | {mode} gallery'
            build_sheet(items, board_path, title=title, tile_size=args.tile_size, cols=args.cols)
            board_paths.append((title, board_path))

    for chapter_name, spec in chapter_specs.items():
        for mode in spec['modes']:
            items = select_board_items(entries, spec['categories'], [mode], args.per_group_limit)
            if not items:
                continue
            board_path = boards_dir / f'{chapter_name}_{mode}_sheet.png'
            title = f"{spec['title']} | {mode}"
            build_sheet(items, board_path, title=title, tile_size=args.tile_size, cols=args.cols)
            board_paths.append((title, board_path))

    atlas_json = {
        'render_manifest': str(args.render_manifest),
        'representative_manifest': str(args.representative_manifest) if args.representative_manifest else None,
        'num_cases': len(entries),
        'boards': [
            {
                'title': title,
                'path': str(path),
            }
            for title, path in board_paths
        ],
        'categories': sorted({entry['category'] for entry in entries}),
    }
    with open(output_dir / 'atlas_manifest.json', 'w', encoding='utf-8') as file:
        json.dump(atlas_json, file, ensure_ascii=False, indent=2)

    write_markdown_index(output_dir / 'INDEX.md', entries, board_paths, base_dir=output_dir.parent)
    print(f'Saved organized figure atlas to: {output_dir}')


if __name__ == '__main__':
    main(build_parser().parse_args())
