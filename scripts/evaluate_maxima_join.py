import sys
import os
import json
import zipfile
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.utils import merge_predictions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, help='Path where the data for the dataloaders is stored', required = True)
    parser.add_argument("--model-dir", type=str, help='Path where are the models', required = True)
    parser.add_argument("--output-file", type=str, help='Path where to save results', required = True)
    parser.add_argument("--model-file", type=str, help='zip model file', required = True)
    
    args = parser.parse_args()
    
    sample_names = np.load(args.data_file)['names']


    with zipfile.ZipFile(args.model_file, 'r') as zipf:
        
        weight_matrix = np.load(zipf.open('weight_scores.npz'))
        mean_probes_per_timepoint = weight_matrix['avgsites']
        accuracy_per_timepoint_per_model = weight_matrix['performance']
        decoding_dict = json.load(zipf.open('decoding.json'))
        temperatures = np.load(zipf.open('calibration.npy'))
        temperatures = temperatures.flatten()

        decoding_dict = json.load(zipf.open('decoding.json'))
        try:
            merge_dict = json.load(zipf.open('merge.json'))
        except KeyError:
            merge_dict = None

    sample_name_mapper = {i:k for i, k in enumerate(sample_names)}
    

    dfs = list()
    for d in os.listdir(args.model_dir):

        model_n = d
        d = os.path.join(args.model_dir, d)
        
        if not os.path.isdir(d):
            continue

        try:
            df = pd.read_csv(os.path.join(d, 'maxima_performance.csv'), header = 0, index_col = None)
        except FileNotFoundError:
            continue
        df['model_n'] = model_n

        scores = np.array(df[list(decoding_dict.values())])
        scores = scores/temperatures[int(model_n)]
        scores = torch.exp(torch.nn.functional.log_softmax(torch.from_numpy(scores), dim = -1))
        scores = scores.numpy()
        df[list(decoding_dict.values())] = scores

        dfs.append(df)
        
    dfs = pd.concat(dfs)

    if merge_dict is not None:
        dfs, decoding_dict = merge_predictions(dfs, decoding_dict, merge_dict)

    class_columns = list(decoding_dict.values())

    dfs['Idx'] = dfs['Idx'].map(sample_name_mapper)
    dfs = dfs.sort_values(['Idx', 'Timepoint', 'Seed_pregen', 'model_n'])

    scores = np.array(dfs[class_columns])
    idxs = np.array(dfs['Idx'])
    number_probes = np.array(dfs['NSites'])
    timepoints = np.array(dfs['Timepoint'])
    seed_pregens = np.array(dfs['Seed_pregen'])

    output_df = {
        'sample': list(),
        'number_probes': list(),
        'timepoint': list(),
        'seed_pregen': list(),
    }
    for k in decoding_dict.values():
        output_df[k] = list()

    for i in tqdm(range(0, scores.shape[0], 4)):

        idx = idxs[i]
        n = number_probes[i]
        timepoint = timepoints[i]
        seed_pregen = seed_pregens[i]
        sample_scores = scores[i:i+4, :]

        calculated_weights = np.ones(
            (
                accuracy_per_timepoint_per_model.shape[0],
                accuracy_per_timepoint_per_model.shape[2],
            ), 
        dtype=float)

        # for m in range(calculated_weights.shape[0]):

        #     weights = deepcopy(accuracy_per_timepoint_per_model[m])
        #     n_probes = mean_probes_per_timepoint[m]
        #     t = n_probes.searchsorted(n)
        #     t = int(t)
        #     if t == weights.shape[0]:
        #         calculated_weights[m, :] = weights[t-1]
        #     elif t == 0:
        #         calculated_weights[m, :] = weights[t]
        #     else:
        #         weights = weights[t-1:t+1]
        #         x = [n_probes[t-1], n_probes[t]]
        #         for i in range(weights.shape[1]):
        #             y = weights[:, i]
        #             calculated_weights[m, i] = np.interp(n, x, y)

        # final_scores = np.zeros(sample_scores.shape[1])
        # for i in range(sample_scores.shape[1]):
        #     final_scores[i] = np.average(sample_scores[:, i], weights = calculated_weights[:, i])

        best = np.where(sample_scores == np.max(sample_scores))[0]
        if len(best) > 0:
            best = best[0]
        final_scores = sample_scores[best.item(), :]

        output_df['sample'].append(idx)
        output_df['number_probes'].append(n)
        output_df['timepoint'].append(timepoint)
        output_df['seed_pregen'].append(seed_pregen)

        for j, s in enumerate(final_scores):
            try:
                output_df[decoding_dict[str(j)]].append(s)
            except:
                output_df[decoding_dict[int(j)]].append(s)

    output_df = pd.DataFrame(output_df)

    output_df.to_csv(
        args.output_file, 
        header = True,
        index = False
    )

    