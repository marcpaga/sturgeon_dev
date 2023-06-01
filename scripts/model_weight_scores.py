"""This script takes an ensemble of models and weights the produced combined
score based on the performance of each individual model on each class and 
timepoint.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.utils import load_labels_encoding, merge_predictions

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help='Path where the data for the dataloaders is stored', required = True)
    parser.add_argument("--num-models", type=int, help='Number of ensemble models', required = True)
    parser.add_argument("--output-file", type=str, help='Path where the model is saved', required = True)
    parser.add_argument("--prediction-type", type=str, required = True)

    args = parser.parse_args()

    print('Loading encodings')
    _, decoding_dict, merge_dict = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__), 
            '../static/'+args.prediction_type+'.json'
        ),
        return_merge = True
    )

    print('Reading test performance files')
    test_df = list()
    for model_num in range(args.num_models):
        df = pd.read_csv(os.path.join(args.model_dir, str(model_num), 'test_performance.csv'), header = 0, index_col = None)
        df['Model'] = model_num
        test_df.append(df)
    test_df = pd.concat(test_df)

    if merge_dict is not None:
        test_df, decoding_dict = merge_predictions(
            prediction_df = test_df, 
            decoding_dict = decoding_dict, 
            merge_dict = merge_dict,
        )
    encoding_dict = {v:k for k, v in decoding_dict.items()}

    scores = torch.from_numpy(np.array(test_df[list(decoding_dict.values())]))
    scores = torch.nn.functional.log_softmax(scores, dim = -1).exp()
    scores = scores.numpy()
    test_df[list(decoding_dict.values())] = scores


    test_df = test_df.sort_values(['Model', 'Timepoint', 'Label'])
    perc_correct = np.zeros(
        (
            len(np.unique(test_df['Model'])), 
            len(np.unique(test_df['Timepoint'])),
            len(decoding_dict), 
        ), 
        dtype=float,
    )
    avg_sites = np.zeros(
        (
            len(np.unique(test_df['Model'])), 
            len(np.unique(test_df['Timepoint']))
        ), 
        dtype=int,
    )

    print('Calculating performance per timepoint and model')
    for m in np.unique(test_df['Model']):
        s = test_df['Model'].searchsorted(m, side = 'left')
        n = test_df['Model'].searchsorted(m, side = 'right')
        subdf = test_df.iloc[s:n]

        for t in np.unique(subdf['Timepoint']):
            s = subdf['Timepoint'].searchsorted(t, side = 'left')
            n = subdf['Timepoint'].searchsorted(t, side = 'right')

            subsubdf = subdf.iloc[s:n]
            p = np.array(subsubdf[list(decoding_dict.values())]).argmax(-1)
            y = np.array(subsubdf['Label'].map(encoding_dict))
            c = p==y

            for yi in np.unique(y):
                where_y = y == yi
                total_y = np.sum(where_y)
                perc_correct[m, t, yi] = np.sum(c[where_y])/total_y

            avg_sites[m, t] = int(np.mean(subsubdf['NSites']))

    print('Saving results')
    np.savez(
        args.output_file, 
        avgsites = avg_sites,
        performance = perc_correct,
    )
