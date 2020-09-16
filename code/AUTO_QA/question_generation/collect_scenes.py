
import argparse, json, os
from datetime import datetime as dt

"""
All scenes are collected to form a single scene file
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../output/train_scenes')
parser.add_argument('--output_file', default='../output/ARGO_train_scenes.json')
parser.add_argument('--version', default='1.0')
parser.add_argument('--split',default='train')
parser.add_argument('--date', default=dt.today().strftime("%d/%m/%Y"))


def main(args):
    scenes = []
    split = args.split
    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.json'):
            continue
        path = os.path.join(args.input_dir, filename)
        with open(path, 'r') as f:
            scene_file = json.load(f)
        scenes.extend(scene_file['scenes'])
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': split
        },
        'scenes': scenes
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)