import argparse
import ast
import json
from pathlib import Path


PLOT_CALLS = {
    'plt.plot',
    'plt.scatter',
    'plt.bar',
    'plt.hist',
    'plt.imshow',
    'fig.savefig',
    'plt.savefig',
    'Image.save',
    'save_svg_figure',
    'save_svg_image',
    'save_image',
    'write_gallery_svg',
}


def call_name(node):
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f'{base}.{node.attr}' if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def function_has_visual_call(function_node):
    for node in ast.walk(function_node):
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name in PLOT_CALLS:
                return True
    return False


def scan_file(path):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and function_has_visual_call(node):
            functions.append(node.name)
    return functions


def main():
    parser = argparse.ArgumentParser('Scan visualization functions under visual/ and tools/.')
    parser.add_argument('--root', default='.', type=str)
    args = parser.parse_args()

    root = Path(args.root)
    targets = [root / 'visual', root / 'tools']
    result = {}
    for target in targets:
        if not target.exists():
            continue
        for path in sorted(target.rglob('*.py')):
            functions = scan_file(path)
            if functions:
                result[str(path.relative_to(root))] = functions
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
