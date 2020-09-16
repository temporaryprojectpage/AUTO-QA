import argparse, json, os, h5py
import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--output_base', default='./output/processed')
parser.add_argument('--input_encodings', default='./output/processed/all_questions.h5')
parser.add_argument('--train_out', default='train_questions.h5')
parser.add_argument('--val_out', default='val_questions.h5')
parser.add_argument('--split', default='train')
parser.add_argument('--prefix' , default='ARGO')

def add_values_to_group(group, values):
    for k, v in values.items():
        group.create_dataset(k, data=v)

def main(args):
    try:
        file = h5py.File(args.input_encodings, 'r')
    except:
        raise ValueError('cannot open input file, check if file is absent '
                         'or open at some other place' + args.input_encodings)
    
    keys = list(file.keys())
    assert len(keys) > 0
    num_data = file[keys[0]].shape[0]

    idxs = np.asarray([i for i in range(num_data)])
    idxs = np.random.permutation(idxs)

    shuffled = {}
    for keys in file.keys():
        feature = file[keys][:]
        shuffled[keys] = feature[idxs]
        shuffled[keys] = shuffled[keys][:num_data]
        print(keys, feature.shape)

    file.close()
    split = (0.80, 0.20)
    index = [int(num_data * s) for s in split]

    train_hf = h5py.File(os.path.join(args.output_base, args.train_out), 'w')
    val_hf = h5py.File(os.path.join(args.output_base, args.val_out), 'w')
    
    print(index)
    for k, v in shuffled.items():
        train_hf.create_dataset(k, data=v[:index[0]])
        val_hf.create_dataset(k, data=v[index[0]:index[0]+index[1]])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
