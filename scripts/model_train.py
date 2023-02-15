"""Script used for training"""

import os
import sys
import argparse
import gc

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.dataclasses import MicroarrayDatasetFromOnlineSims
from sturgeon_dev.schedulers import GradualWarmupScheduler
from sturgeon_dev.utils import load_labels_encoding, create_logger, calculate_color_dict
from sturgeon_dev import seeds
from sturgeon_dev.constants import CLASS_COLORS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, help='Path where the data for the dataloaders is stored', required = True)
    parser.add_argument("--output-dir", type=str, help='Path where the model is saved', required = True)
    parser.add_argument("--prediction-type", type=str, required = True)

    parser.add_argument("--split-csv", type=str, required = True)
    parser.add_argument("--model-num", type=int, required = True)
    
    parser.add_argument("--num-epochs", type=int, default = 500)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--noise", type=float, default = 0.1)
    parser.add_argument("--adaptive-sampling", type=bool, default = False)
    parser.add_argument("--adapt-every", type=int, default=1, help='Adapt sampling every N epochs')
    parser.add_argument("--adaptive-correction", type=float, default = 0.3)
    parser.add_argument("--summary-file", type=str, default = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/stats.csv')))
    parser.add_argument("--probes-file", type=str, default = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/probes.csv')))
    parser.add_argument("--bin-time", type=int, default = 300)
    parser.add_argument("--min-time", type=int, default = 0)
    parser.add_argument("--max-time", type=int, default = 12)
    parser.add_argument("--min-coord", type=int, default = 0)
    parser.add_argument("--max-coord", type=int, default = 3000000000)
    parser.add_argument("--read-variation", type=int, default = 0)

    parser.add_argument("--model-type", type=str, choices = ['ce', 'triplet', 'autoencoder', 'double', 'famsub'], default='ce')
    parser.add_argument("--layer-sizes", type=int, nargs="+", default = [256, 128])
    parser.add_argument("--dropout", type=float, default = 0.5)
    parser.add_argument("--start-lr", type=float, default = 1e-3)
    parser.add_argument("--final-lr", type=float, default = 1e-5)
    parser.add_argument("--weight-decay", type=float, default = 0.3)
    parser.add_argument("--warmup-steps", type=int, default = 1000)
    parser.add_argument("--cooldown-epochs", type=int, default = 500)

    parser.add_argument("--checkpoint-every", type=int, default = 500)
    parser.add_argument("--validation-multiplier", type=int, default = 25)
    parser.add_argument("--max-checkpoints", type=int, default = 1)
    parser.add_argument("--checkpoint-metric", type=str, default= 'metric.balanced_accuracy')
    parser.add_argument("--checkpoint-metricdirection", type=str, choices=['max', 'min'], default='max')
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--pretrain-checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default = seeds.ONLINE_SIM_STARTING_SEED)

    args = parser.parse_args()

    assert args.processes >= 2

    logger = create_logger('model_train', os.path.join(args.output_dir, 'log.log'))

    logger.info('Using device: ' + str(device))

    encoding_dict, decoding_dict, other_class = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__), 
            '../static/'+args.prediction_type+'.json'
        ),
        return_merge = False,
        return_other = True,
    )
    color_dict = calculate_color_dict(encoding_dict, decoding_dict, CLASS_COLORS)

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
        label_colors = color_dict,
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
        num_epochs = args.num_epochs,
        starting_seed = args.seed,
    )
    
    sub_to_fam_dict = None
    if args.model_type == 'ce':
        from sturgeon_dev.models import SparseMicroarrayModel as Model
    elif args.model_type == 'triplet':
        from sturgeon_dev.models import TripletMicroarrayModel as Model
    elif args.model_type == 'autoencoder':
        from sturgeon_dev.models import AutoencoderMicroarrayModel as Model
    elif args.model_type == 'double':
        from sturgeon_dev.models import DoubleModel as Model
    elif args.model_type == 'famsub':
        from sturgeon_dev.models import TwoLevelSparseMicroarrayModel as Model

        sub_to_fam_dict = dict()
        fam_labels = dict()
        c = 0
        for k in encoding_dict.keys():
            k = k.split(' - ')[0]
            try:
                fam_labels[k]
            except KeyError:
                fam_labels[k] = c
                c += 1
        for k, v in decoding_dict.items():
            sub_to_fam_dict[k] = fam_labels[v.split(' - ')[0]]
    

    logger.info('Preparing model')
    model = Model(
        dataset = dataset,
        num_classes = len(decoding_dict),
        dropout = args.dropout,
        layer_sizes = args.layer_sizes,
        device = device,
        adaptive_sampling_correction = args.adaptive_correction,
        sub_to_fam_dict = sub_to_fam_dict,
        other_class = other_class,
    )
    model = model.to(device)

    if args.pretrain_checkpoint is not None:
        logger.info('Loading pretrain checkpoint')

        state_dict = torch.load(args.pretrain_checkpoint)['model_state']
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)
        except RuntimeError:
            logger.info('New last layer size {}'.format(len(decoding_dict)))
            decoder_keys = list()
            for k in state_dict.keys():
                if k.startswith("decoder"):
                    decoder_keys.append(k)
            for k in decoder_keys:
                del state_dict[k]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)

        for k in missing_keys:
            assert not k.startswith('encoder')
        for k in unexpected_keys:
            assert k.startswith('decoder')
        model.to(device)


    ##    OPTIMIZATION     #############################################
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay = args.weight_decay)
    total_steps =  int(  (len(dataset.samplers['train']) * args.num_epochs)  /args.batch_size)
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_steps, eta_min=args.final_lr, last_epoch=-1, verbose=False)
    lr_scheduler = GradualWarmupScheduler(
        optimizer, 
        multiplier = 1.0, 
        total_epoch = args.warmup_steps, 
        after_scheduler = cosine_lr,
    )
    schedulers = {'lr_scheduler': lr_scheduler}
    clipping_value = 2


    ##   MODEL PART2        #############################################
    model.optimizer = optimizer
    model.schedulers = schedulers
    model.clipping_value = clipping_value
    
    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    # check output dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    
    logger.info('Starting training')

    train_batch_num = model.train_loop(
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        output_dir = args.output_dir,
        adaptive_sampling = args.adaptive_sampling,
        adapt_every = args.adapt_every,
        max_checkpoints = args.max_checkpoints,
        checkpoint_every = args.checkpoint_every,
        validation_multiplier = args.validation_multiplier,
        checkpoint_metric = args.checkpoint_metric,
        checkpoint_metricdirection = args.checkpoint_metricdirection,
        processes = args.processes,
        logger = logger,
    )

    if args.cooldown_epochs > 0:

        new_optimizer = torch.optim.AdamW(model.parameters(), lr=args.final_lr, weight_decay = args.weight_decay)
        new_optimizer.load_state_dict(model.optimizer.state_dict())
        schedulers = {}
        clipping_value = 2

        ##   MODEL PART2        #############################################
        model.optimizer = new_optimizer
        model.schedulers = schedulers
        model.clipping_value = clipping_value
        model.train_loop(
            num_epochs = args.cooldown_epochs,
            batch_size = args.batch_size,
            output_dir = args.output_dir,
            adaptive_sampling = args.adaptive_sampling,
            adapt_every = args.adapt_every,
            max_checkpoints = args.max_checkpoints,
            checkpoint_every = args.checkpoint_every,
            validation_multiplier = args.validation_multiplier,
            checkpoint_metric = args.checkpoint_metric,
            checkpoint_metricdirection = args.checkpoint_metricdirection,
            processes = args.processes,
            logger = logger,
            prev_train_batch_num = train_batch_num
        )
