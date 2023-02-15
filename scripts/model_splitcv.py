import os
import sys
from copy import deepcopy
from collections import deque
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.utils import load_labels_encoding

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", type=str, help='File with the samples metadata')
    parser.add_argument("--output-file", type=str, help='Output csv file')
    parser.add_argument("--prediction-type", type=str)
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--train-splits", type=int, default=2)
    parser.add_argument("--validation-splits", type=int, default=1)
    parser.add_argument("--test-splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    assert args.train_splits+args.validation_splits+args.test_splits == args.num_splits
    assert args.num_models <= args.num_splits

    seed = args.seed
    labels_df = pd.read_csv(args.samples_file)
    labels_df['sample_number'] -= 1

    encoder, decoder = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__), 
            '../static/'+args.prediction_type+'.json'
        )
    )

    excluded_samples = {
        'sample': list(),
        'class': list(),
    }

    label_idx_mapping = dict()
    labels = list()
    for i, row in labels_df.iterrows():
        enc_k = encoder[row.diagnostics_class]
        if enc_k == -1:
            excluded_samples['sample'].append(row.sample_number)
            excluded_samples['class'].append(row.diagnostics_class)
            continue
        labels.append(decoder[enc_k])
        if decoder[enc_k] not in label_idx_mapping.keys():
            label_idx_mapping[decoder[enc_k]] = list()
        label_idx_mapping[decoder[enc_k]].append(row.sample_number)
    labels = np.array(labels)

    splits = list()
    for _ in range(args.num_splits):
        splits.append(list())

    for label in np.unique(labels):
        label_idx = np.array(label_idx_mapping[label])

        if seed:
            np.random.seed(seed)
            seed += 1
        np.random.shuffle(label_idx)

        for i, a in enumerate(np.array_split(label_idx, args.num_splits)):
            splits[i].append(a)

    for i, split in enumerate(splits):
        splits[i] = np.concatenate(split)

    train_idxs = np.arange(0, args.train_splits, 1)
    validation_idxs = np.arange(0, args.validation_splits, 1) + 1 + train_idxs[-1]
    test_idxs = np.arange(0, args.test_splits, 1) + 1 + validation_idxs[-1]


    splits_df = {
        'model_n': list(),
        'set': list(),
        'sample': list(),
    }
    for n in range(args.num_models):
        splits_copy = deque(deepcopy(splits))
        splits_copy.rotate(n)
        for idxs, idxs_name in zip([train_idxs, validation_idxs, test_idxs], ['train', 'validation', 'test']):
            for i in idxs:
                splits_df['sample'] += splits_copy[i].tolist()
                splits_df['set'] += [idxs_name] * len(splits_copy[i])
                splits_df['model_n'] += [int(n)] * len(splits_copy[i])
                
    df = pd.DataFrame(splits_df)
    df['class'] = ''
    for i, s in enumerate(df['sample']):
        df.loc[i, 'class'] = labels_df.loc[labels_df['sample_number'] == s, 'diagnostics_class'].item()


    if len(excluded_samples) > 0:
        for n in range(args.num_models):
            excluded_samples = pd.DataFrame(excluded_samples)
            excluded_samples['model_n'] = int(n)
            excluded_samples['set'] = 'excluded'

            df = pd.concat([df, excluded_samples])

    df = df.astype({'sample':int})

    df.to_csv(args.output_file, header = True, index = False)
    

