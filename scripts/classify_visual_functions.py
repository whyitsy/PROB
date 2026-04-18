import argparse
import json
from pathlib import Path


RULES = {
    'training': ['loss', 'pseudo', 'train', 'step'],
    'evaluation': ['metric', 'correlation', 'error', 'percentage'],
    'embeddings': ['embedding', 'manifold', 'pca', 'tsne', 'umap', 'score_space'],
    'cases': ['prediction', 'ground_truth', 'contact_sheet', 'sampling', 'trajectory', 'gate', 'joint'],
}


def classify_name(name):
    lower = name.lower()
    for category, keywords in RULES.items():
        if any(keyword in lower for keyword in keywords):
            return category
    return 'cases'


def main():
    parser = argparse.ArgumentParser('Classify visualization functions into categories.')
    parser.add_argument('--scan_json', required=True, type=str)
    args = parser.parse_args()

    scan_result = json.loads(Path(args.scan_json).read_text(encoding='utf-8'))
    grouped = {key: [] for key in RULES}
    for file_path, functions in scan_result.items():
        for function_name in functions:
            category = classify_name(function_name)
            grouped[category].append({'file': file_path, 'function': function_name})
    print(json.dumps(grouped, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
