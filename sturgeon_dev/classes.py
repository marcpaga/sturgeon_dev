import os
from abc import abstractmethod
import random
from copy import deepcopy
from itertools import islice
from functools import partial
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader
from sklearn.metrics import balanced_accuracy_score

import seeds
import constants
from utils import generate_log_df, clean_checkpoints

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

class MicroarrayDataset(Dataset):
    """Microarray dataset

    Expects data to be in binary format

    Args:
        data_file (str): npy file with the data
        label (list): which labels to use, can be multiples, ["class","main","sub"]
        dropout (float): fraction of values set to zero per sample
        noise (float): fraction of values that will be inverted per sample,
            note that this fraction is based on the number of values left
            after dropout
        test_fraction (float): fraction of samples used for testing
        methylation_encoding_dict (dict): dict mapping methylation values 
        mainlabel_encoding_dict (dict): dict mapping main label names to int
        sublabel_encoding_dict (dict): dict mapping sub label names to int
        train_test_split_seed (int): seed for splitting data
        sample_generator_start_seed (int): seed for generating data
    """

    def __init__(
        self,
        data_file,
        batch_size,
        label_encoding,
        split_csv = None,
        model_num = None,
        test_mode = False,
        noise = 0.1,
        methylation_encoding_dict = None,
        generator_seed = seeds.SAMPLE_GENERATOR_START,
        diagnostics_encoding = constants.DIAGNOSTICS_ENCODING,
        no_measurement = constants.NO_MEASURED,
        *args,
        **kwargs,
    ): 
        super(MicroarrayDataset, self).__init__()

        self.data_file = data_file
        self.label_encoding = label_encoding

        print('Splitting samples according to dataframe')
        split_df = pd.read_csv(split_csv)
        self.model_num = model_num
        self.split_df = split_df[split_df['model_n'] == self.model_num]
        
        self.train_idx = np.array(self.split_df.loc[self.split_df['set'] == 'train', 'sample'])
        self.val_idx = np.array(self.split_df.loc[self.split_df['set'] == 'validation', 'sample'])
        self.test_idx = np.array(self.split_df.loc[self.split_df['set'] == 'test', 'sample'])

        self.batch_size = batch_size
        self.test_mode = test_mode

        self.diagnostics_encoding = diagnostics_encoding
        self.methylation_encoding_dict = methylation_encoding_dict
        self.no_measurement = no_measurement
        
        print('Loading data')
        self.data = self.load_data(data_file)
       
        self.train_sampler = None
        self.validation_sampler = None
        self.noise = noise
        self.generator_seed = generator_seed

    def __len__(self):
        
        return len(self.train_idx)

    def __getitem__(self, idx):

        return {
            'x':  self.gen_sample(self.data['x'][idx]), 
            'y': self.data['y'][idx],
            'diagnostics': self.data['diagnostics'][idx],
            'idx': np.array([idx]),
            'X': self.data['x'][idx],
        }

    def load_data(self, data_file):
        """Load npy data

        Args:
            data_file (str): file to be loaded, should have x and y as keys
        """

        arr = np.load(data_file)
        print('Loading measurements')
        x = self.encode_x(arr['x'], self.methylation_encoding_dict)
        print('Encoding labels diag')
        diag = self.encode_labels(arr['diagnostics'], self.diagnostics_encoding)
        print('Encoding labels')
        y = self.encode_labels(arr['diagnostics'], self.label_encoding)
        data =  {
            'x': x, 
            'diagnostics': diag,
            'y': y, 
        }
        return data

    def encode_x(self, x, encoding_dict = None):

        if encoding_dict is not None:
            new_x = deepcopy(x)
            for k, v in encoding_dict.items():
                new_x[x == k] = v
            return new_x
        else:
            return x

    def encode_labels(self, y, encoding_dict):
        """Encode string labels as integers

        Args:
            y (list): list with labels as strings
            encoding_dict (dict): dict with labels as keys an integers as values
        """

        encoded_y = np.zeros((y.shape[0], ), dtype = int)
        for i, sample in enumerate(y):
            encoded_y[i] = encoding_dict[sample]
        return encoded_y

    def split_data(self, y, seed, validation_fraction, test_fraction):
        """Split samples between training and testing

        Args:
            y (list): list with samples classes
            seed (int): seed for reproducibility
            validation_fraction (float): fraction of samples for validation
            test_fraction (float): fraction of samples for testing
        """

        train_idx = list()
        val_idx = list()
        test_idx = list()
        idxs = np.arange(0, len(y), 1)
        labels = sorted(np.unique(y))

        for label in labels:
            idx = idxs[y == label]
            num_val_samples = int(len(idx) * validation_fraction)
            num_test_samples = int(len(idx) * test_fraction)
            num_train_samples = len(idx) - num_val_samples - num_test_samples
            random.seed(seed)
            random.shuffle(idx)
            train_idx.append(idx[:num_train_samples])
            val_idx.append(idx[num_train_samples:num_train_samples+num_val_samples])
            test_idx.append(idx[num_train_samples+num_val_samples:])

            seed += 1

        self.train_idx = np.concatenate(train_idx)
        self.val_idx = np.concatenate(val_idx)
        self.test_idx = np.concatenate(test_idx)

        return self.train_idx, self.val_idx, self.test_idx

    def gen_sample(self, x):
        raise NotImplementedError()

    def _prepare_samplers(self):
        
        self.train_sampler = BatchSampler(
            data_source = self, 
            idxs = self.train_idx, 
            batch_size = self.batch_size, 
            seed = seeds.TRAIN_ORDER,
            labels = self.data['y'],
            balanced_sampling = True,
            test_mode = self.test_mode
        )
        
        self.validation_sampler = BatchSampler(
            data_source = self, 
            idxs = self.val_idx, 
            batch_size = self.batch_size, 
            seed = seeds.VAL_ORDER,
            labels = self.data['y'],
            balanced_sampling = True,
            test_mode = self.test_mode
        )

    
    def reload_sampler(self, seed, sampler):

        if sampler == 'train':
            self.train_sampler = BatchSampler(
                data_source = self, 
                idxs = self.train_idx, 
                batch_size = self.batch_size, 
                seed = seeds.TRAIN_ORDER+seed,
                labels = self.data['y'],
                balanced_sampling = True,
                test_mode = self.test_mode
            )
        
        if sampler == 'validation':
            self.validation_sampler = BatchSampler(
                data_source = self, 
                idxs = self.val_idx, 
                batch_size = self.batch_size, 
                seed = seeds.VAL_ORDER+seed,
                labels = self.data['y'],
                balanced_sampling = True,
                test_mode = self.test_mode
            )

class MicroarrayDatasetFromPregens(MicroarrayDataset):

    def __init__(
        self,
        pregens_dir,
        *args,
        **kwargs
    ):
        super(MicroarrayDatasetFromPregens, self).__init__(*args, **kwargs)

        self.pregens_dir = pregens_dir

    def __getitem__(self, idx):

        return {
            'x':  self.gen_sample(self.data['x'][idx[0]], idx[1]), 
            'y': self.data['y'][idx[0]],
            'diagnostics': self.data['diagnostics'][idx[0]],
            'idx': np.array([idx[0]]),
            'idx_pregen': np.array([int(idx[1].split('/')[-1].split('.')[0].split('_')[-1])]),
            'X':  self.data['x'][idx[0]], 
        }

    def gen_sample(self, x, pregen_file):
        
        new_x = deepcopy(x)
        #self.generator_seed += 1

        return self._generate_sample(
            x = x, 
            new_x = new_x, 
            noise = self.noise, 
            seed = None,
            pregen_file = pregen_file,
            states = list(self.methylation_encoding_dict.values()),
            test_mode = self.test_mode
        )

    def _generate_sample(self, x, new_x, noise, pregen_file, states, test_mode, seed = None):

        # random select batch size pregen files
        pregens = np.load(pregen_file)['a']

        if test_mode:

            x = np.repeat(np.expand_dims(x, 0), repeats = pregens.shape[0], axis = 0)
            new_x = np.repeat(np.expand_dims(new_x, 0), repeats = pregens.shape[0], axis = 0)

            for i in range(pregens.shape[0]):
                new_x[i, ~pregens[i][0].astype(bool)] = 0 # TODO remove [0]

            if noise > 0:
                for i in range(pregens.shape[0]):
                    positions = np.where(pregens[i][0])[0] # TODO remove [0]
                    if seed:
                        random.seed(seed+1)
                    random.shuffle(positions)
                    positions = positions[:int(len(positions)*noise)]
                    pos = positions[x[i, positions] == states[0]]
                    neg = positions[x[i, positions] == states[1]]
                    new_x[i, pos] = states[1]
                    new_x[i, neg] = states[0]

        else:
            # put the values of the samples in the pregen positions
            idx = list(range(pregens.shape[0]))
            if seed:
                random.seed(seed)
            random.shuffle(idx)
            pregen_arr = pregens[idx[0]]
            new_x[~pregen_arr[0].astype(bool)] = 0 # TODO remove [0]

            # add noise
            if noise > 0: 
                positions = np.where(pregen_arr[0])[0] # TODO remove [0]
                if seed:
                    random.seed(seed+1)
                random.shuffle(positions)
                positions = positions[:int(len(positions)*noise)]
                pos = positions[x[positions] == states[0]]
                neg = positions[x[positions] == states[1]]
                new_x[pos] = states[1]
                new_x[neg] = states[0]

        return new_x

    def _prepare_samplers(self):
        
        self.train_sampler = BatchSampler(
            data_source = self, 
            idxs = self.train_idx, 
            batch_size = self.batch_size, 
            seed = seeds.TRAIN_ORDER,
            labels = self.data['y'],
            balanced_sampling = True,
            pregens_dir = os.path.join(self.pregens_dir, 'train'),
            test_mode = self.test_mode,
        )
        
        self.validation_sampler = BatchSampler(
            data_source = self, 
            idxs = self.val_idx, 
            batch_size = self.batch_size, 
            seed = seeds.VAL_ORDER,
            labels = self.data['y'],
            balanced_sampling = True,
            pregens_dir = os.path.join(self.pregens_dir, 'validation'),
            test_mode = self.test_mode,
        )

        if self.test_mode:
            self.test_sampler = BatchSampler(
                data_source = self, 
                idxs = self.test_idx, 
                batch_size = self.batch_size, 
                seed = seeds.VAL_ORDER,
                labels = self.data['y'],
                balanced_sampling = True,
                pregens_dir = os.path.join(self.pregens_dir, 'test'),
                test_mode = self.test_mode,
            )

    
    def reload_sampler(self, seed, sampler):

        if sampler == 'train':
            self.train_sampler = BatchSampler(
                data_source = self, 
                idxs = self.train_idx, 
                batch_size = self.batch_size, 
                seed = seeds.TRAIN_ORDER+seed,
                labels = self.data['y'],
                balanced_sampling = True,
                pregens_dir = os.path.join(self.pregens_dir, 'train'),
                test_mode = self.test_mode,
            )
        
        if sampler == 'validation':
            self.validation_sampler = BatchSampler(
                data_source = self, 
                idxs = self.val_idx, 
                batch_size = self.batch_size, 
                seed = seeds.VAL_ORDER+seed,
                labels = self.data['y'],
                balanced_sampling = True,
                pregens_dir = os.path.join(self.pregens_dir, 'validation'),
                test_mode = self.test_mode,
            )
        
        if sampler == 'test':
            self.test_sampler = BatchSampler(
                data_source = self, 
                idxs = self.test_idx, 
                batch_size = self.batch_size, 
                seed = seeds.TEST_ORDER+seed,
                labels = self.data['y'],
                balanced_sampling = True,
                pregens_dir = os.path.join(self.pregens_dir, 'test'),
                test_mode = self.test_mode,
            )

class MicroarraySubsetDataset(MicroarrayDataset):
    """Either give measured_probes and probes_df or pregens_dir
    """

    def __init__(self, *args, **kwargs):
        super(MicroarraySubsetDataset, self).__init__(*args, **kwargs)

        self._prepare_samplers()

    def __getitem__(self, idx):
        return {
            'x': self.data_subset[idx, :], 
            'y': self.data['y'][idx],
            'diagnostics': self.data['diagnostics'][idx],
            'idx': np.array([idx]),
        }

    def subset_data(self, measured_probes_indices):
        
        new_x = deepcopy(self.data['x'])
        mask = np.ones(new_x.shape, dtype=bool)
        mask[:, measured_probes_indices] = False
        new_x[mask] = self.no_measurement
        self.data_subset = new_x
        return self.data_subset
        
    def _prepare_samplers(self, seed = seeds.TRAIN_ORDER):
        
        self.train_sampler = FinetuningBatchSampler(
            data_source = self, 
            idxs = self.train_idx, 
            batch_size = self.batch_size, 
            seed = seed,
            test_mode = False,
        )
        
        self.validation_sampler = FinetuningBatchSampler(
            data_source = self, 
            idxs = self.val_idx, 
            batch_size = self.batch_size, 
            seed = seed,
            test_mode = False,
        )

        self.test_sampler = FinetuningBatchSampler(
            data_source = self, 
            idxs = self.test_idx, 
            batch_size = self.batch_size, 
            test_mode = True,
        )

class BatchSampler(Sampler):

    def __init__(self, idxs, batch_size, seed, labels = None, balanced_sampling = False, max_samples_per_class = None, pregens_dir = None, test_mode = False, *args, **kwargs):
        super(BatchSampler, self).__init__(*args, **kwargs)
        
        self.idxs = sorted(idxs)
        self.batch_size = batch_size
        self.seed = seed
        self.labels = labels
        self.balanced_sampling = balanced_sampling
        self.max_samples_per_class = max_samples_per_class
        if balanced_sampling:
            if self.labels is None:
                raise ValueError()
            
            self.idxs = self._upsample()

        self.pregens_dir = pregens_dir
        if self.pregens_dir:
            self.pregen_files = [os.path.abspath(os.path.join(self.pregens_dir, p)) for p in os.listdir(self.pregens_dir) if p.endswith(('.npz'))]
        else:
            self.pregen_files = None

        self.test_mode = test_mode
        if self.test_mode:
            self.batched_idxs = self._batch_idxs_test()
        else:
            self.batched_idxs = self._batch_idxs()

        self.total_samples_epoch = 0
        for b in self.batched_idxs:
            self.total_samples_epoch += len(b)

    def __iter__(self):
        return iter(self.batched_idxs)

    def __len__(self):
        return len(self.batched_idxs)

    def _batch_idxs(self):

        random.seed(self.seed)
        random.shuffle(self.idxs)
        batched_idxs = list(iter(partial(lambda it: tuple(islice(it, self.batch_size)), iter(self.idxs)), ()))

        if self.pregen_files:
            while len(self.pregen_files) < len(self.idxs):
                self.pregen_files += self.pregen_files
            random.seed(self.seed)
            random.shuffle(self.pregen_files)
            
            batches_pregen_files = list(iter(partial(lambda it: tuple(islice(it, self.batch_size)), iter(self.pregen_files)), ()))

            batched_idxs_with_pregens = list()
            for idx_batch, pregen_batch in zip(batched_idxs, batches_pregen_files[:len(batched_idxs)]):
                batch_list = list()
                for idx_b, idx_p in zip(idx_batch, pregen_batch):
                    batch_list.append((idx_b, idx_p))
                batched_idxs_with_pregens.append(batch_list)
            return batched_idxs_with_pregens

        return batched_idxs

    def _batch_idxs_test(self):

        batched_idxs_with_pregens = list()
        c = 0
        l = list()
        for idx in self.idxs:
            for pregen_file in self.pregen_files:
                l.append((idx, pregen_file))
                c += 1

                if c == self.batch_size:
                    batched_idxs_with_pregens.append(l)
                    l = list()
                    c = 0

        batched_idxs_with_pregens.append(l)
        return batched_idxs_with_pregens

    def _upsample(self):

        seed = deepcopy(self.seed)
        labels = np.array(self.labels)[self.idxs]
        if self.max_samples_per_class is None:
            max_label_counts = np.max(np.bincount(labels))
        else:
            assert self.max_samples_per_class >= np.max(np.bincount(labels))
            max_label_counts = self.max_samples_per_class

        upsampled_idxs = [np.array(self.idxs)]
        for label in np.unique(labels):
            label_idx = np.array(self.idxs)[np.where(labels == label)[0]]
            label_count = len(label_idx)
            np.random.seed(seed)
            needed_samples = max_label_counts-label_count
            if needed_samples > 0:
                upsampled_idxs.append(np.random.choice(label_idx, size=needed_samples, replace=True))
                seed += 1

        upsampled_idxs = np.concatenate(upsampled_idxs)
        assert len(np.unique(np.bincount(self.labels[upsampled_idxs]))) == 1
        return upsampled_idxs

class FinetuningBatchSampler(Sampler):

    def __init__(self, idxs, batch_size, seed = None, test_mode = False, *args, **kwargs):
        super(FinetuningBatchSampler, self).__init__(*args, **kwargs)

        self.idxs = sorted(idxs)
        self.batch_size = batch_size
        self.seed = seed
        self.batched_idxs = self._batch_idxs(test_mode = test_mode)

    def __iter__(self):
        return iter(self.batched_idxs)

    def __len__(self):
        return len(self.batched_idxs)

    def _batch_idxs(self, test_mode):

        if not test_mode:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.idxs)
        batched_idxs = list(iter(partial(lambda it: tuple(islice(it, self.batch_size)), iter(self.idxs)), ()))

        return batched_idxs