import os
import json
import warnings
import shutil

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

def rchop(s, sub):
    return s[:-len(sub)] if s.endswith(sub) else s

def load_labels_encoding(json_file, return_merge = False, return_other = False):

    with open(json_file, 'r') as handle:
        json_dict = json.load(handle)

    assert 'encoding' in json_dict.keys()
    assert 'decoding' in json_dict.keys()

    encoding_dict = json_dict['encoding']
    decoding_dict_tmp = json_dict['decoding']

    try:
        merge_dict = json_dict['merge']
    except KeyError:
        merge_dict = None

    try:
        other_class = json_dict['other_class']
    except KeyError:
        other_class = None

    assert len(encoding_dict) == 91

    total_classes = np.unique(list(encoding_dict.values()))
    # we use -1 as an exclude label from training encoding
    total_classes = total_classes[total_classes != -1]
    if not other_class:
        assert len(decoding_dict_tmp) == len(total_classes)

    decoding_dict = dict()
    for k, v in decoding_dict_tmp.items():
        decoding_dict[int(k)] = v

    if return_merge:
        return encoding_dict, decoding_dict, merge_dict
    
    if return_other:
        return encoding_dict, decoding_dict, other_class
    
    return encoding_dict, decoding_dict

def mean_color(colors):
    """
    Takes a list of colors in hex code and calculates their average color
    """
    
    if not isinstance(colors, list):
        colors = [colors]

    rgb_colors = list()
    for color in colors:
        rgb_colors.append(to_rgb(color))
       
    return to_hex(np.array(rgb_colors).mean(0).tolist())

def calculate_color_dict(encoding_dict, decoding_dict, diagnostics_color_dict):

    color_dict = dict()
    for k, v in decoding_dict.items():
        color_dict[v] = list()

        for i, j in encoding_dict.items():
            if j == k:
                color_dict[v].append(diagnostics_color_dict[i])

        color_dict[v] = mean_color(color_dict[v])

    return color_dict

def get_best_checkpoint(
    log_file, 
    metric,
    max_or_min
):

    if max_or_min == 'max': 
        ascending = False
    elif max_or_min == 'min':
        ascending = True

    df = pd.read_csv(log_file, header = 0, index_col = None)
    df = df[df['checkpoint'] == 'yes']

    # calculate the difference between training and validation of the metric,
    # to filter out steps where we are overfitting
    df[metric] = df[metric+".train"] - df[metric+".val"]
    # reverse the metric if we want to minimize (non-overfit loss would be
    # negative)
    if max_or_min == 'max':
        df[metric] = -df[metric]
    df = df.sort_values(metric, ascending = ascending)

    # filter for non-overfit
    df = df[df[metric] >= 0]

    # sort again values based on the validation metric
    df = df.sort_values(metric+".val", ascending = ascending)

    return df


def clean_checkpoints(
    log_file, 
    checkpoint_dir, 
    max_checkpoints, 
    metric, 
    max_or_min,
):

    # if there are less or same amount of checkpoint files as the max do nothing
    if len(os.listdir(checkpoint_dir)) <= max_checkpoints:
        return None
    
    # change the sorting based on whether we want to maximize the metric
    # (e.g. accuracy), or minimize (e.g. loss/error)
    if max_or_min == 'max': 
        ascending = False
    elif max_or_min == 'min':
        ascending = True

    try:
        df = pd.read_csv(log_file, header = 0, index_col = None)
        df = df[df['checkpoint'] == 'yes']
    except FileNotFoundError:
        warnings.warn('{} not found, not cleaning checkpoints'.format(log_file))

    # calculate the difference between training and validation of the metric,
    # to filter out steps where we are overfitting
    df[metric] = df[metric+".train"] - df[metric+".val"]
    # reverse the metric if we want to minimize (non-overfit loss would be
    # negative)
    if max_or_min == 'max':
        df[metric] = -df[metric]
    df = df.sort_values(metric, ascending = ascending)

    
    # filter for non-overfit
    if np.sum(df[metric] >= 0) > 0:
        df = df[df[metric] >= 0]
        steps_to_remove = df[df[metric] < 0]['step'].tolist()
    else:
        steps_to_remove = list()

    # sort again values based on the validation metric
    df = df.sort_values(metric+".val", ascending = ascending)

    best_step = str(df['step'].tolist()[0])
    chk_file = os.path.join(checkpoint_dir, 'checkpoint_'+str(best_step)+'.pt')
    new_file = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
    try:
        shutil.copy(chk_file, new_file)
    except FileNotFoundError:
        pass

    # remove all that are not on the top max_checkpoints
    steps_to_remove += df[max_checkpoints:]['step'].tolist()
    for step in steps_to_remove:
        chk_file = os.path.join(checkpoint_dir, 'checkpoint_'+str(step)+'.pt')
        if os.path.isfile(chk_file):
            print('Removing: ' + chk_file)
            os.remove(chk_file)

    return None

def generate_log_df(keys):
    """Creates a data.frame to store the logging values
    """
    
    header = ['step',  # step number
              'time']  # time it took
    # add losses and metrics for train and validation
    for k in keys:
        header.append(k + '.train')
        header.append(k + '.val')
    # whether a checkpoint was saved at this step
    header.append('lr')
    header.append('checkpoint')
    
    log_dict = dict()
    for k in header:
        log_dict[k] = [None]
    return pd.DataFrame(log_dict)

def create_logger(application, output_file):

    import logging

    logger = logging.getLogger(application)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(output_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def softmax(x):
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp

def merge_predictions(prediction_df, decoding_dict, merge_dict):

    old_class_columns = list(decoding_dict.values())
    non_class_columns = np.array(prediction_df.columns[~np.isin(prediction_df.columns, old_class_columns)])
   
    rev_merge_dict = dict()
    for k, v in merge_dict.items():
        for c in v:
            rev_merge_dict[c] = k
    for v in decoding_dict.values():
        try:
            rev_merge_dict[v]
        except KeyError:
            rev_merge_dict[v] = v

    try:
        prediction_df['Label'] = prediction_df['Label'].map(rev_merge_dict)
    except KeyError:
        pass
    
    for k, v in merge_dict.items():
        prediction_df[k] = 0
        for c in v:
            prediction_df[k] += prediction_df[c]
            prediction_df = prediction_df.drop(c, axis=1)

    new_class_columns = np.array(prediction_df.columns[~np.isin(prediction_df.columns, non_class_columns)])

    new_class_columns = np.sort(new_class_columns)

    new_decoding_dict = dict()
    for i, c in enumerate(new_class_columns):
        new_decoding_dict[i] = c

    final_column_order = non_class_columns.tolist() + new_class_columns.tolist()

    return prediction_df[final_column_order], new_decoding_dict