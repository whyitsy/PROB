import argparse
import json
from pathlib import Path


TARGET_MODULE = {
    'training': 'util.visual.training',
    'evaluation': 'util.visual.evaluation',
    'embeddings': 'util.visual.embeddings',
    'cases': 'util.visual.cases',
}


def main():
    parser = argparse.ArgumentParser('Build a refactor mapping for visualization functions.')
    parser.add_argument('--classified_json', required=True, type=str)
    args = parser.parse_args()

    classified = json.loads(Path(args.classified_json).read_text(encoding='utf-8'))
    lines = []
    for category, items in classified.items():
        target_module = TARGET_MODULE.get(category, 'util.visual.cases')
        for item in items:
            lines.append(f'{item["file"]}::{item["function"]} -> {target_module}::{item["function"]}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
