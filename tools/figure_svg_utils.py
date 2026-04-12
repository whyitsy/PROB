import base64
import io
import math
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def save_svg_figure(fig, output_path, *, pad_inches: float = 0.06):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)


def save_svg_image(image_np, output_path, *, title: Optional[str] = None, max_width_in: float = 10.0):
    height, width = image_np.shape[:2]
    fig_width = min(max_width_in, max(4.5, width / 160.0))
    fig_height = fig_width * float(height) / max(float(width), 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(image_np)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    save_svg_figure(fig, output_path, pad_inches=0.01)


def _parse_svg_number(raw_value: Optional[str]) -> Optional[float]:
    if raw_value is None:
        return None
    value = ''.join(ch for ch in str(raw_value) if (ch.isdigit() or ch in '.-'))
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _svg_dimensions_from_bytes(raw_bytes: bytes):
    root = ET.fromstring(raw_bytes)
    width = _parse_svg_number(root.attrib.get('width'))
    height = _parse_svg_number(root.attrib.get('height'))
    if width is not None and height is not None and width > 0 and height > 0:
        return width, height
    view_box = root.attrib.get('viewBox')
    if view_box:
        parts = [float(part) for part in view_box.replace(',', ' ').split()]
        if len(parts) == 4:
            return max(parts[2], 1.0), max(parts[3], 1.0)
    return 1000.0, 1000.0


def _file_dims_and_data_uri(path: Path):
    suffix = path.suffix.lower()
    raw_bytes = path.read_bytes()
    if suffix == '.svg':
        width, height = _svg_dimensions_from_bytes(raw_bytes)
        mime = 'image/svg+xml'
    else:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            width, height = image.size
        mime = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
        }.get(suffix, 'application/octet-stream')
    encoded = base64.b64encode(raw_bytes).decode('ascii')
    return float(width), float(height), f'data:{mime};base64,{encoded}'


def _pil_dims_and_data_uri(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    width, height = image.size
    return float(width), float(height), f'data:image/png;base64,{encoded}'


def _tile_data(item: dict):
    if item.get('image_path'):
        return _file_dims_and_data_uri(Path(item['image_path']))
    if item.get('pil_image') is not None:
        return _pil_dims_and_data_uri(item['pil_image'])
    raise ValueError('Each gallery item must provide either image_path or pil_image.')


def _pick_cols(mode: str, item_count: int, explicit_cols: Optional[int] = None) -> int:
    if explicit_cols is not None and explicit_cols > 0:
        return explicit_cols
    if mode in {'sampling', 'joint'}:
        return 1
    if mode == 'trajectory':
        return 1 if item_count <= 3 else 2
    if mode == 'gate':
        return 2
    return 2


def _pick_tile_width(mode: str, explicit_tile_width: Optional[int] = None) -> int:
    if explicit_tile_width is not None and explicit_tile_width > 0:
        return explicit_tile_width
    if mode in {'sampling', 'joint'}:
        return 1280
    if mode == 'trajectory':
        return 1040
    if mode == 'gate':
        return 900
    return 1000


def write_gallery_svg(
    items: List[dict],
    output_path,
    *,
    title: str,
    mode: str,
    cols: Optional[int] = None,
    tile_width: Optional[int] = None,
    label_height: int = 86,
    title_height: int = 64,
    outer_padding: int = 20,
    hgap: int = 22,
    vgap: int = 28,
):
    if not items:
        return None

    cols = _pick_cols(mode, len(items), explicit_cols=cols)
    tile_width = _pick_tile_width(mode, explicit_tile_width=tile_width)

    prepared = []
    for item in items:
        src_width, src_height, data_uri = _tile_data(item)
        content_height = int(round(tile_width * src_height / max(src_width, 1.0)))
        prepared.append({
            'data_uri': data_uri,
            'src_width': src_width,
            'src_height': src_height,
            'content_height': content_height,
            'label_lines': item.get('label_lines', []),
        })

    rows = [prepared[idx:idx + cols] for idx in range(0, len(prepared), cols)]
    row_heights = [max(item['content_height'] + label_height for item in row) for row in rows]
    canvas_width = outer_padding * 2 + cols * tile_width + (cols - 1) * hgap
    canvas_height = outer_padding * 2 + title_height + sum(row_heights) + max(0, len(rows) - 1) * vgap

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="#111111"/>',
        f'<rect x="0" y="0" width="{canvas_width}" height="{title_height + outer_padding}" fill="#161616"/>',
        f'<text x="{outer_padding}" y="{outer_padding + 20}" fill="#FFFFFF" font-size="24" font-family="Arial, Helvetica, sans-serif">{escape(title)}</text>',
    ]

    current_y = outer_padding + title_height
    item_index = 0
    for row, row_height in zip(rows, row_heights):
        for col_index, item in enumerate(row):
            x = outer_padding + col_index * (tile_width + hgap)
            y = current_y
            content_height = item['content_height']
            label_y = y + row_height - label_height
            svg_lines.append(f'<rect x="{x}" y="{y}" width="{tile_width}" height="{row_height}" fill="#1A1A1A" rx="10" ry="10"/>')
            svg_lines.append(f'<image x="{x}" y="{y}" width="{tile_width}" height="{content_height}" preserveAspectRatio="xMidYMid meet" xlink:href="{item["data_uri"]}"/>')
            svg_lines.append(f'<rect x="{x}" y="{label_y}" width="{tile_width}" height="{label_height}" fill="#0F0F0F"/>')
            for line_idx, line in enumerate(item['label_lines'][:3]):
                text_y = label_y + 26 + line_idx * 22
                svg_lines.append(
                    f'<text x="{x + 14}" y="{text_y}" fill="#F5F5F5" font-size="18" font-family="Arial, Helvetica, sans-serif">{escape(str(line))}</text>'
                )
            item_index += 1
        current_y += row_height + vgap

    svg_lines.append('</svg>')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(svg_lines), encoding='utf-8')
    return output_path
