
import os
import sys
import argparse
import gc

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.dataclasses import MicroarrayDatasetFromOnlineSims
from sturgeon_dev.utils import load_labels_encoding, create_logger
from sturgeon_dev import seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, help='Path where the data for the dataloaders is stored', required = True)
    parser.add_argument("--checkpoint-file", type=str, help='File with the weights to be loaded', required = True)
    parser.add_argument("--output-dir", type=str, help='Path where the model is saved', required = True)
    parser.add_argument("--prediction-type", type=str, required = True)

    parser.add_argument("--split-csv", type=str, required = True)
    parser.add_argument("--model-num", type=int, required = True)
    
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--noise", type=float, default = 0.1)
    parser.add_argument("--summary-file", type=str, default = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/stats.csv')))
    parser.add_argument("--probes-file", type=str, default = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/probes.csv')))
    parser.add_argument("--bin-time", type=int, default = 300)
    parser.add_argument("--min-time", type=int, default = 0)
    parser.add_argument("--max-time", type=int, default = 12)
    parser.add_argument("--min-coord", type=int, default = 0)
    parser.add_argument("--max-coord", type=int, default = 3000000000)
    parser.add_argument("--read-variation", type=int, default = 0)

    parser.add_argument("--model-type", type=str, choices = ['ce', 'triplet', 'famsub'], default='ce')
    parser.add_argument("--layer-sizes", type=int, nargs="+", default = [256, 128])
    parser.add_argument("--dropout", type=float, default = 0.5)

    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--seed", type=int, default = seeds.ONLINE_SIM_STARTING_SEED)

    args = parser.parse_args()
    
    logger = create_logger('model_test', os.path.join(args.output_dir, 'log.log'))

    logger.info('Using device: ' + str(device))

    encoding_dict, decoding_dict = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__), 
            '../static/'+args.prediction_type+'.json'
        )
    )

    logger.info('Using classification: ' + args.prediction_type)

    logger.info('Loading Nanopore run stats')
    summary_df = pd.read_csv(args.summary_file, header = 0, index_col = None)
    summary_df = summary_df.sort_values('start_time')
    read_lengths = np.sort(np.array(summary_df['sequence_length_template']))
    read_start_times = np.array(summary_df['start_time'])
    read_durations = np.array(summary_df['duration'])

    logger.info('Loading probe information')
    probes_df = pd.read_csv(args.probes_file, header = 0)
    probe_coords = np.array(probes_df['longcoord'])
    breakpoints = np.concatenate([\
        np.array([0]), 
        np.array(probes_df[probes_df['breakpoint']]['longcoord']),
    ])
    
    del probes_df
    del summary_df
    gc.collect()

    logger.info('Loading dataset')
    dataset = MicroarrayDatasetFromOnlineSims(
        data_file = args.data_file,
        label_encoding = encoding_dict,
        label_decoding = decoding_dict,
        split_csv = args.split_csv,
        model_num = args.model_num,
        read_lengths = read_lengths,
        read_start_times = read_start_times,
        read_durations = read_durations,
        bin_time = args.bin_time,
        min_time = args.min_time,
        max_time = args.max_time,
        min_coord = args.min_coord,
        max_coord = args.max_coord,
        read_variation = args.read_variation,
        probe_coords = probe_coords,
        breakpoints = breakpoints,
        test_mode = True,
        starting_seed = args.seed,
    )
    
    if args.model_type == 'ce':
        from sturgeon_dev.models import SparseMicroarrayModel as Model
    elif args.model_type == 'triplet':
        from sturgeon_dev.models import TripletMicroarrayModel as Model
    elif args.model_type == 'famsub':
        from sturgeon_dev.models import TwoLevelSparseMicroarrayModel as Model

    logger.info('Preparing model')
    model = Model(
        dataset = dataset,
        num_classes = len(decoding_dict),
        dropout = args.dropout,
        layer_sizes = args.layer_sizes,
        device = device,
        sub_to_fam_dict = None,
    )
    model = model.to(device)

    state_dict = torch.load(args.checkpoint_file)
    model.load_state_dict(state_dict['model_state'], strict = True)
    model = model.to(device)

    del state_dict
    gc.collect()

    for cvset in ['test', 'validation']:
        logger.info('Starting performance evaluation of set: ' + cvset)
        model.test_loop(
            cvset = cvset,
            batch_size = args.batch_size,
            output_file = os.path.join(args.output_dir, cvset + '_performance.csv'),
            decoding_dict = decoding_dict,
            processes = args.processes,
            logger = logger,
            progress_bar = True,
        )
            
            

