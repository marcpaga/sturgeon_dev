from abc import abstractmethod
from copy import deepcopy
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler
from numba import jit

from sturgeon_dev import seeds
from sturgeon_dev import constants

class MicroarrayDataset(Dataset):
    """Microarray dataset
    """

    def __init__(
        self,
        data_file,
        label_encoding,
        label_decoding,
        split_csv,
        model_num,
        test_mode = False,
        noise = 0.1,
        methylation_encoding_dict = None,
        generator_seed = seeds.SAMPLE_GENERATOR_START,
        diagnostics_encoding = constants.DIAGNOSTICS_ENCODING,
        no_measurement = constants.NO_MEASURED,
        label_colors = None,
        *args,
        **kwargs,
    ): 
        super(MicroarrayDataset, self).__init__()

        self.data_file = data_file
        self.label_encoding = label_encoding
        self.label_decoding = label_decoding
        self.label_colors = label_colors

        print('Splitting samples according to dataframe')
        split_df = pd.read_csv(split_csv)
        self.model_num = model_num
        self.split_df = split_df[split_df['model_n'] == self.model_num]

        self.idxs = {
            "train": np.array(self.split_df.loc[self.split_df['set'] == 'train', 'sample']),
            "validation": np.array(self.split_df.loc[self.split_df['set'] == 'validation', 'sample']),
            "test": np.array(self.split_df.loc[self.split_df['set'] == 'test', 'sample']),
        }

        self.test_mode = test_mode

        self.diagnostics_encoding = diagnostics_encoding
        self.methylation_encoding_dict = methylation_encoding_dict
        self.no_measurement = no_measurement
        
        self.noise = noise
        self.generator_seed = generator_seed

        print('Loading data')
        self.data = self.load_data(data_file)
       
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):

        raise NotImplementedError

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

    @abstractmethod
    def gen_sample(self, x):
        raise NotImplementedError

    @abstractmethod
    def _prepare_samplers(self):
        raise NotImplementedError
    
    @abstractmethod
    def reload_sampler(self, seed, sampler):
        raise NotImplementedError

class MicroarrayDatasetFromOnlineSims(MicroarrayDataset):

    def __init__(
        self,
        read_lengths: np.ndarray,
        read_start_times: np.ndarray,
        read_durations: np.ndarray,
        probe_coords: np.ndarray,
        breakpoints: np.ndarray,
        bin_time: int = 300,
        min_time: int = 0,
        max_time: int = 12,
        min_coord: int = 0,
        max_coord: int = 3000000000,
        read_variation: int = 0,
        starting_seed: int = seeds.ONLINE_SIM_STARTING_SEED,
        seed_ranges: dict = {
            "train": seeds.TRAIN_SEED_RANGE, 
            "validation": seeds.VALIDATION_SEED_RANGE, 
            "test": seeds.TEST_SEED_RANGE,
        },
        *args,
        **kwargs
    ):
        super(MicroarrayDatasetFromOnlineSims, self).__init__(*args, **kwargs)


        self.bin_time = bin_time
        self.min_time = min_time
        self.max_time = max_time
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.read_variation = read_variation
        self.probe_coords = probe_coords
        self.breakpoints = breakpoints
        self.samplers = dict()

        
        self.starting_seed = starting_seed
        self.seed_ranges = seed_ranges

        self.read_lengths = read_lengths
        self.reads_per_time_bin = self.calculate_reads_per_timebin(
            read_start_times = read_start_times,
            read_durations = read_durations,
            timebin = self.bin_time,
        )
        
        self.reads_per_time_bin = np.cumsum(self.reads_per_time_bin[self.min_time:self.max_time])
        self._read_variations = np.random.normal(0, read_variation, 10000).astype(int)
        self._read_variations[np.abs(self._read_variations) >= np.min(self.reads_per_time_bin)] = 0

        max_reads = np.max(self.reads_per_time_bin) + np.max(np.abs(self._read_variations))
        self._direction_possibilities = np.concatenate([
            np.full((max_reads, ), 1, dtype = int),
            np.full((max_reads, ), -1, dtype = int),
        ])

        self.read_lengths_len = len(self.read_lengths)
        self.read_variations_len = len(self._read_variations)
        self.direction_possibilities_len = len(self._direction_possibilities)

        assert np.sum((np.min(self.reads_per_time_bin) + self._read_variations) <= 0) == 0

        if not self.test_mode:
            self._prepare_samplers()

    def __len__(self):
        return len(self.samplers['train'].balanced_idxs)

    def __getitem__(self, idx):

        return {
            'x':  self.apply_run_to_sample(
                self.data['x'][idx[0]], 
                self.reads_per_time_bin[idx[1]], 
                idx[2],
            ), 
            'y': self.data['y'][idx[0]],
            'diagnostics': self.data['diagnostics'][idx[0]],
            'idx': np.array([idx[0]]),
            'timepoint': np.array(idx[1]),
            'seed_pregen': np.array(idx[2]),
            'X':  self.data['x'][idx[0]], 
        }

    def calculate_reads_per_timebin(
        self,
        read_start_times: np.ndarray,
        read_durations: np.ndarray,
        timebin: int,
    ) -> np.ndarray:

        max_time = read_start_times[-1] + read_durations[-1]
        read_end_times = read_start_times + read_durations
        num_timebins = int(max_time // timebin)

        l = list()
        for i in range(num_timebins):
            s = np.searchsorted(read_start_times, i*timebin)
            n = np.searchsorted(read_end_times, (i+1)*timebin)
            l.append(n-s)

        return l

    @staticmethod
    @jit(nopython=True, cache=True)
    def sample_numpy_numba(
        num_reads, 
        read_lengths, 
        max_coord, 
        direction_possibilities,
        read_variations, 
        seed,
        read_variations_len,
        read_lengths_len,
        direction_possibilities_len,
    ):

        np.random.seed(seed)
        num_reads = num_reads + read_variations[np.random.randint(0, read_variations_len)]

        np.random.seed(seed + num_reads)
        sampled_read_lengths = read_lengths[np.random.randint(0, read_lengths_len, size = num_reads)]

        np.random.seed(seed + sampled_read_lengths[0])
        sampled_read_st = np.random.randint(0, max_coord, size = num_reads)

        np.random.seed(seed + sampled_read_lengths[-1])
        sampled_read_direction = direction_possibilities[np.random.randint(0, direction_possibilities_len, size = num_reads)]

        return sampled_read_lengths, sampled_read_st, sampled_read_direction


    @staticmethod
    @jit(nopython=True)
    def apply_breakpoints_numba(
        read_st,
        read_nd,
        read_direction,
        breakpoints,
    ):

        bp_st = np.searchsorted(breakpoints, v = read_st, side = 'left')
        bp_nd = np.searchsorted(breakpoints, v = read_nd, side = 'right')

        bp_reads = np.where(bp_st != bp_nd)[0]

        for i in bp_reads:
            if read_direction[i] == 1:
                read_nd[i] = breakpoints[bp_st[i]]
            elif read_direction[i] == -1:
                read_st[i] = breakpoints[bp_st[i]]

        return read_st, read_nd

    @staticmethod
    @jit(nopython=True)
    def map_reads_numba(
        read_st,
        read_nd,
        probe_coords,
        present,
    ):

        bp_st = np.searchsorted(probe_coords, v = read_st, side = 'left')
        bp_nd = np.searchsorted(probe_coords, v = read_nd, side = 'right')

        for s, n in zip(bp_st, bp_nd):
            present[s:n] = True

        return present

    @staticmethod
    @jit(nopython=True)
    def apply_sim_to_sample(x, present, no_measurement):
        
        new_x = np.full(x.shape, no_measurement, dtype=np.float32)
        new_x[present] = x[present]
        return new_x

    @staticmethod
    @jit(nopython=True)
    def apply_noise_to_sample(x, present, noise):

        measured_sites = np.where(present)[0]
        num_noise_sites = int(len(measured_sites) * noise)
        np.random.shuffle(measured_sites)
        noise_sites = measured_sites[:num_noise_sites]
        x[noise_sites] = -x[noise_sites]

        return x

    @staticmethod
    @jit(nopython=True)
    def sort_read_directions(read_st, read_nd, sampled_read_st, sampled_read_nd, sampled_read_direction):

        read_st[sampled_read_direction == 1] = sampled_read_st[sampled_read_direction == 1]
        read_st[sampled_read_direction == -1] = sampled_read_nd[sampled_read_direction == -1]
        read_nd[sampled_read_direction == 1] = sampled_read_nd[sampled_read_direction == 1]
        read_nd[sampled_read_direction == -1] = sampled_read_st[sampled_read_direction == -1]

        return read_st, read_nd

    def simulate_sequencing_run(self, num_reads, seed):

        sampled_read_lengths, sampled_read_st, sampled_read_direction = self.sample_numpy_numba(
            num_reads = num_reads,
            read_lengths = self.read_lengths,
            max_coord = self.max_coord,
            direction_possibilities = self._direction_possibilities,
            read_variations = self._read_variations,
            read_lengths_len = self.read_lengths_len,
            read_variations_len = self.read_variations_len,
            direction_possibilities_len = self.direction_possibilities_len,
            seed = seed,
        )

        sampled_read_lengths *= sampled_read_direction
        sampled_read_nd = sampled_read_st + sampled_read_lengths
        read_st = np.zeros((sampled_read_st.shape), dtype=int)
        read_nd = np.zeros((sampled_read_nd.shape), dtype=int)

        read_st, read_nd = self.sort_read_directions(
            read_st, read_nd,
            sampled_read_st, 
            sampled_read_nd, 
            sampled_read_direction
        )

        read_st, read_nd = self.apply_breakpoints_numba(
            read_st,
            read_nd,
            sampled_read_direction,
            self.breakpoints,
        )

        present = np.zeros((len(self.probe_coords), ), dtype = bool)

        present = self.map_reads_numba(
            read_st,
            read_nd,
            self.probe_coords,
            present,
        )

        return present

    def apply_run_to_sample(self, x, num_reads, seed):

        present = self.simulate_sequencing_run(num_reads, seed)

        x = self.apply_sim_to_sample(x, present, self.no_measurement)

        if self.noise > 0:
            x = self.apply_noise_to_sample(x, present, self.noise)

        return x

    def _prepare_samplers(self):

        if self.test_mode:
            cv_sets = ['train', 'validation', 'test']
        else:
            cv_sets = ['train', 'validation']

        for cvset in cv_sets:

            self.samplers[cvset] = AdaptativeSimSampler(
                data_source = self,
                idxs = self.idxs[cvset], 
                labels = self.data['y'][self.idxs[cvset]], 
                min_time = self.min_time, 
                max_time = self.max_time, 
                seed = self.starting_seed, 
                seed_range = self.seed_ranges[cvset], 
                test_mode = self.test_mode
            )

    def prepare_sampler_test_mode(self, cvset):

        
        self.samplers[cvset] = AdaptativeSimSampler(
            data_source = self,
            idxs = self.idxs[cvset], 
            labels = self.data['y'][self.idxs[cvset]], 
            min_time = self.min_time, 
            max_time = self.max_time, 
            seed = self.starting_seed, 
            seed_range = self.seed_ranges[cvset], 
            test_mode = True
        )
        
    def reload_sampler(self, sampler):
        self.samplers[sampler].resample()


    
class SimSampler(Sampler):

    def __init__(self, idxs, labels, min_time, max_time, seed, seed_range, test_mode = False, *args, **kwargs):
        super(SimSampler, self).__init__(*args, **kwargs)
        """
        Args:
            idxs (np.ndarray): list with the idxs for this sampler
            labels (np.ndarray): list with the labels for ALL the samples
            seed (int): seed for reproducibility
            seedrange (tuple): tuple of ints that will determine the range of the seeds
        """

        assert len(idxs) == len(labels)
        assert len(np.unique(idxs)) == len(idxs)

        idx_order = np.argsort(idxs)

        self.idxs = idxs[idx_order]
        self.idx_from_idxs = np.arange(len(self.idxs))
        self.labels = labels[idx_order]
        self.min_time = min_time
        self.max_time = max_time
        self.starting_seed = seed
        self.seed = deepcopy(seed)
        self.seed_range = np.arange(seed_range[0], seed_range[1])
        self.test_mode = test_mode

        self.calculate_total_samples()

    def __iter__(self):
        try:
            return iter(self.balanced_idxs)
        except AttributeError as e:
            raise Exception('No balanced idxs in this sampler, did you run `sampler.resample()` ?') from e

    def __len__(self):
        return int(self.total_samples)

    def resample(self):
        if self.test_mode:
            self.balanced_idxs = self._upsample_test()
        else:
            self.balanced_idxs = self._upsample()

    def calculate_total_samples(self):

        self.total_samples = np.max(np.bincount(self.labels)) * len(np.unique(self.labels))

    def _upsample(self):

        max_label_counts = np.max(np.bincount(self.labels))
        available_timepoints = np.tile(np.arange(self.min_time, self.max_time),max_label_counts)[:max_label_counts]
        
        upsampled_idxs = list()
        upsampled_idx_from_idxs = list()
        upsampled_timepoints = list()
        upsampled_seeds = list()

        for label in np.unique(self.labels):
            label_idx = self.idx_from_idxs[np.where(self.labels == label)[0]]
            label_count = len(label_idx)

            upsampled_idx_from_idxs.append(label_idx)
            upsampled_idxs.append(self.idxs[label_idx])

            needed_samples = max_label_counts-label_count
            if needed_samples > 0:
                np.random.seed(self.seed)
                upsampling = np.random.choice(label_idx, size=needed_samples, replace=True)
                upsampled_idx_from_idxs.append(upsampling)
                upsampled_idxs.append(self.idxs[upsampling])
                self.seed += 1

            np.random.seed(self.seed)
            np.random.shuffle(available_timepoints)
            upsampled_timepoints.append(deepcopy(available_timepoints))
            self.seed += 1

            np.random.seed(self.seed)
            np.random.shuffle(self.seed_range)
            upsampled_seeds.append(deepcopy(self.seed_range[:max_label_counts]))
            self.seed += 1

        upsampled_idxs = np.concatenate(upsampled_idxs)
        upsampled_idx_from_idxs = np.concatenate(upsampled_idx_from_idxs)
        upsampled_timepoints = np.concatenate(upsampled_timepoints)
        upsampled_seeds = np.concatenate(upsampled_seeds)
        assert len(np.unique(np.bincount(self.labels[upsampled_idx_from_idxs]))) == 1

        balanced_idxs = list()
        for i, t, s in zip(upsampled_idxs, upsampled_timepoints, upsampled_seeds):
            balanced_idxs.append((i, t, s))

        random.seed(self.seed)
        random.shuffle(balanced_idxs)
        self.seed += 1

        return balanced_idxs

    def _upsample_test(self):

        balanced_idxs = list()
        
        available_timepoints = np.arange(self.min_time, self.max_time)

        for i in self.idxs:
            for t in available_timepoints:
                for s in self.seed_range:
                    balanced_idxs.append((i, t, s))

        return balanced_idxs

class AdaptativeSimSampler(SimSampler):

    def __init__(self, *args, **kwargs):
        super(AdaptativeSimSampler, self).__init__(*args, **kwargs)
        """
        Args:
        """

        self.warray = np.ones((len(np.arange(self.min_time, self.max_time)), len(np.unique(self.labels))))
        self.warray /= np.sum(self.warray)

    def _upsample(self):

        samples_per_category = self.warray * self.total_samples
        samples_per_category = samples_per_category.astype(int)

        np.random.seed(self.seed)
        np.random.shuffle(self.seed_range)
        upsampled_seeds = deepcopy(self.seed_range[:self.total_samples]).tolist()

        upsampled_idxs = list()
        upsampled_timepoints = list()
        for label in np.unique(self.labels):

            total_label_samples = np.sum(samples_per_category[:, label])

            label_idx = self.idx_from_idxs[np.where(self.labels == label)[0]]
            label_count = len(label_idx)

            if label_count >= total_label_samples:
                upsampled_idxs.append(self.idxs[label_idx][:total_label_samples])
            else:
                upsampled_idxs.append(self.idxs[label_idx])
                needed_samples = total_label_samples-label_count
                np.random.seed(self.seed)
                upsampling = np.random.choice(label_idx, size=needed_samples, replace=True)
                upsampled_idxs.append(self.idxs[upsampling])
                self.seed += 1

            timepoints = np.repeat(np.arange(self.min_time, self.max_time), samples_per_category[:, label])
            np.random.seed(self.seed)
            np.random.shuffle(timepoints)
            self.seed += 1
            upsampled_timepoints.append(timepoints)


        upsampled_idxs = np.concatenate(upsampled_idxs)
        upsampled_timepoints = np.concatenate(upsampled_timepoints)
        balanced_idxs = list()
        for i, t, s in zip(upsampled_idxs, upsampled_timepoints, upsampled_seeds):
            balanced_idxs.append((i, t, s))

        random.seed(self.seed)
        random.shuffle(balanced_idxs)
        self.seed += 1

        return balanced_idxs

    def add_weights_array(self, array):
        self.warray = array
    

class TargetedSimSampler(Sampler):

    def __init__(self, idxs, min_time, max_time, seed, seed_range, num_epochs = 1, test_mode = False, *args, **kwargs):
        super(TargetedSimSampler, self).__init__(*args, **kwargs)

        assert len(np.unique(idxs)) == len(idxs)

        idx_order = np.argsort(idxs)

        self.idxs = idxs[idx_order]
        self.min_time = min_time
        self.max_time = max_time
        self.starting_seed = seed
        self.seed = deepcopy(seed)
        self.seed_range = np.arange(seed_range[0], seed_range[1])
        self.test_mode = test_mode
        self.num_epochs = num_epochs

        self.balanced_idxs = list()
        self.test_seeds = list()
        if self.test_mode:
            self._prepare_test_idxs()
        else:
            self._prepare_train_idxs()
        
    def __iter__(self):
        return iter(self.balanced_idxs)

    def __len__(self):
        return len(self.balanced_idxs)

    def _prepare_train_idxs(self):

        self.balanced_idxs = list()
        for _ in range(self.num_epochs):

            random.seed(self.seed)
            random.shuffle(self.idxs)
            self.balanced_idxs += self.seed
            self.seed += 1

    def _prepare_test_idxs(self):

        self.balanced_idxs = self.idxs
        available_timepoints = np.arange(self.min_time, self.max_time)
        for t in available_timepoints:
            for s in self.seed_range:
                self.test_seeds.append((t, s))

class SimBatchSampler(Sampler):

    def __init__(self, idxs, labels, min_time, max_time, seed, seed_range, test_mode = False, *args, **kwargs):
        super(SimBatchSampler, self).__init__(*args, **kwargs)
        """
        Args:
            idxs (np.ndarray): list with the idxs for this sampler
            labels (np.ndarray): list with the labels for ALL the samples
            seed (int): seed for reproducibility
            seedrange (tuple): tuple of ints that will determine the range of the seeds
        """

        assert len(idxs) == len(labels)
        assert len(np.unique(idxs)) == len(idxs)

        self.idxs = idxs
        self.labels = labels
        self.min_time = min_time
        self.max_time = max_time
        self.times_range = np.arange(min_time, max_time, 1)
        self.starting_seed = seed
        self.start_sampling_seed = deepcopy(seed)
        self.current_sampling_seed = deepcopy(seed)
        self.seeds_range = np.arange(seed_range[0], seed_range[1])
        self.test_mode = test_mode

        self.total_samples = len(self.seeds_range) * len(self.times_range)

        if self.test_mode:
            self.sample_matrix = np.empty(
                (len(self.times_range), len(self.idxs)), 
                dtype = np.int16,
            )
            for i in range(len(self.times_range)):
                self.sample_matrix[i] = self.idxs
        else:
            self.sample_matrix = np.empty(
                (len(self.times_range), len(np.unique(self.labels))), 
                dtype = np.int16,
            )

        self.idxs_per_class = dict()
        for el in np.unique(self.labels):
            el_idxs = self.idxs[self.labels == el]
            if len(el_idxs) < len(self.times_range):
                el_idxs = np.tile(el_idxs, reps = (len(self.times_range)//len(el_idxs)) + 1)

            assert len(el_idxs) >= len(self.times_range)
            self.idxs_per_class[el] = el_idxs

    def __iter__(self):
        for sim_seed in self.seeds_range:
            
            if not self.test_mode:
                self.put_idxs_in_sample_matrix()

            for i, t in enumerate(self.times_range):
                yield(self.sample_matrix[i, :], t, sim_seed)


    def __len__(self):
        return int(self.total_samples)

    def put_idxs_in_sample_matrix(self):

        for el in np.unique(self.labels):    
            np.random.seed(self.current_sampling_seed)
            np.random.shuffle(self.idxs_per_class[el])
            self.sample_matrix[:, el] = self.idxs_per_class[el][:len(self.times_range)]
            self.current_sampling_seed += 1

class MicroarrayDatasetFromOnlineSimsBatched(MicroarrayDataset):

    def __init__(
        self,
        read_lengths: np.ndarray,
        read_start_times: np.ndarray,
        read_durations: np.ndarray,
        probe_coords: np.ndarray,
        breakpoints: np.ndarray,
        bin_time: int = 300,
        min_time: int = 0,
        max_time: int = 12,
        min_coord: int = 0,
        max_coord: int = 3000000000,
        read_variation: int = 0,
        starting_seed: int = seeds.ONLINE_SIM_STARTING_SEED,
        seed_ranges: dict = {
            "train": seeds.TRAIN_SEED_RANGE, 
            "validation": seeds.VALIDATION_SEED_RANGE, 
            "test": seeds.TEST_SEED_RANGE,
        },
        *args,
        **kwargs
    ):
        super(MicroarrayDatasetFromOnlineSimsBatched, self).__init__(*args, **kwargs)


        self.bin_time = bin_time
        self.min_time = min_time
        self.max_time = max_time
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.read_variation = read_variation
        self.probe_coords = probe_coords
        self.breakpoints = breakpoints
        self.samplers = dict()

        
        self.starting_seed = starting_seed
        self.seed_ranges = seed_ranges

        self.read_lengths = read_lengths
        self.reads_per_time_bin = self.calculate_reads_per_timebin(
            read_start_times = read_start_times,
            read_durations = read_durations,
            timebin = self.bin_time,
        )
        
        self.reads_per_time_bin = np.cumsum(self.reads_per_time_bin[self.min_time:self.max_time])
        
        self._read_variations = np.random.normal(0, read_variation, 10000).astype(int)
        self._read_variations[np.abs(self._read_variations) >= np.min(self.reads_per_time_bin)] = 0

        max_reads = np.max(self.reads_per_time_bin) + np.max(np.abs(self._read_variations))
        self._direction_possibilities = np.concatenate([
            np.full((max_reads, ), 1, dtype = int),
            np.full((max_reads, ), -1, dtype = int),
        ])

        self.read_lengths_len = len(self.read_lengths)
        self.read_variations_len = len(self._read_variations)
        self.direction_possibilities_len = len(self._direction_possibilities)

        assert np.sum((np.min(self.reads_per_time_bin) + self._read_variations) <= 0) == 0

        if not self.test_mode:
            self._prepare_samplers()

    def __len__(self):
        return len(self.samplers['train'].balanced_idxs)

    def __getitem__(self, idx):

        return {
            'x': self.apply_run_to_batch(
                self.data['x'][idx[0]], 
                self.reads_per_time_bin[idx[1]], 
                idx[2],
            ), 
            'y': self.data['y'][idx[0]],
            'diagnostics': self.data['diagnostics'][idx[0]],
            'idx': np.array([idx[0]]),
            'timepoint': np.array(idx[1]),
            'seed_pregen': np.array(idx[2]),
            'X':  self.data['x'][idx[0]], 
        }


    def calculate_reads_per_timebin(
        self,
        read_start_times: np.ndarray,
        read_durations: np.ndarray,
        timebin: int,
    ) -> np.ndarray:

        max_time = read_start_times[-1] + read_durations[-1]
        read_end_times = read_start_times + read_durations
        num_timebins = int(max_time // timebin)

        l = list()
        for i in range(num_timebins):
            s = np.searchsorted(read_start_times, i*timebin)
            n = np.searchsorted(read_end_times, (i+1)*timebin)
            l.append(n-s)

        return l

    @staticmethod
    @jit(nopython=True, cache=True)
    def sample_numpy_numba(
        num_reads, 
        read_lengths, 
        max_coord, 
        direction_possibilities,
        read_variations, 
        seed,
        read_variations_len,
        read_lengths_len,
        direction_possibilities_len,
    ):

        np.random.seed(seed)
        num_reads = num_reads + read_variations[np.random.randint(0, read_variations_len)]

        np.random.seed(seed + num_reads)
        sampled_read_lengths = read_lengths[np.random.randint(0, read_lengths_len, size = num_reads)]

        np.random.seed(seed + sampled_read_lengths[0])
        sampled_read_st = np.random.randint(0, max_coord, size = num_reads)

        np.random.seed(seed + sampled_read_lengths[-1])
        sampled_read_direction = direction_possibilities[np.random.randint(0, direction_possibilities_len, size = num_reads)]

        return sampled_read_lengths, sampled_read_st, sampled_read_direction

    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_breakpoints_numba(
        read_st,
        read_nd,
        read_direction,
        breakpoints,
    ):

        bp_st = np.searchsorted(breakpoints, v = read_st, side = 'left')
        bp_nd = np.searchsorted(breakpoints, v = read_nd, side = 'right')

        bp_reads = np.where(bp_st != bp_nd)[0]

        for i in bp_reads:
            if read_direction[i] == 1:
                read_nd[i] = breakpoints[bp_st[i]]
            elif read_direction[i] == -1:
                read_st[i] = breakpoints[bp_st[i]]

        return read_st, read_nd

    @staticmethod
    @jit(nopython=True, cache=True)
    def map_reads_numba(
        read_st,
        read_nd,
        probe_coords,
        present,
    ):

        bp_st = np.searchsorted(probe_coords, v = read_st, side = 'left')
        bp_nd = np.searchsorted(probe_coords, v = read_nd, side = 'right')

        for s, n in zip(bp_st, bp_nd):
            present[s:n] = True

        return present

    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_sim_to_batch(x, present, no_measurement):
        new_x = np.full(x.shape, no_measurement, dtype=np.float32)
        new_x[:, present] = x[:, present]
        return new_x

    @staticmethod
    #@jit(nopython=True, cache=True)
    def apply_noise_to_batch(x, present, noise):

        measured_sites = np.where(present)[0]
        num_noise_sites = int(len(measured_sites) * noise)
        for i in np.arange(x.shape[0]):
            
            np.random.shuffle(measured_sites)
            noise_sites = measured_sites[:num_noise_sites]
            x[i,noise_sites] = -x[i, noise_sites]

        return x

    @staticmethod
    @jit(nopython=True, cache=True)
    def sort_read_directions(read_st, read_nd, sampled_read_st, sampled_read_nd, sampled_read_direction):

        read_st[sampled_read_direction == 1] = sampled_read_st[sampled_read_direction == 1]
        read_st[sampled_read_direction == -1] = sampled_read_nd[sampled_read_direction == -1]
        read_nd[sampled_read_direction == 1] = sampled_read_nd[sampled_read_direction == 1]
        read_nd[sampled_read_direction == -1] = sampled_read_st[sampled_read_direction == -1]

        return read_st, read_nd

    def simulate_sequencing_run(self, num_reads, seed):

        sampled_read_lengths, sampled_read_st, sampled_read_direction = self.sample_numpy_numba(
            num_reads = num_reads,
            read_lengths = self.read_lengths,
            max_coord = self.max_coord,
            direction_possibilities = self._direction_possibilities,
            read_variations = self._read_variations,
            read_lengths_len = self.read_lengths_len,
            read_variations_len = self.read_variations_len,
            direction_possibilities_len = self.direction_possibilities_len,
            seed = seed,
        )

        sampled_read_lengths *= sampled_read_direction
        sampled_read_nd = sampled_read_st + sampled_read_lengths
        read_st = np.zeros((sampled_read_st.shape), dtype=int)
        read_nd = np.zeros((sampled_read_nd.shape), dtype=int)

        read_st, read_nd = self.sort_read_directions(
            read_st, read_nd,
            sampled_read_st, 
            sampled_read_nd, 
            sampled_read_direction
        )

        read_st, read_nd = self.apply_breakpoints_numba(
            read_st,
            read_nd,
            sampled_read_direction,
            self.breakpoints,
        )

        present = np.zeros((len(self.probe_coords), ), dtype = bool)

        present = self.map_reads_numba(
            read_st,
            read_nd,
            self.probe_coords,
            present,
        )

        return present

    def apply_run_to_batch(self, x, num_reads, seed):

        present = self.simulate_sequencing_run(num_reads, seed)

        x = self.apply_sim_to_batch(x, present, self.no_measurement)

        if self.noise > 0:
            
            x = self.apply_noise_to_batch(x, present, self.noise)

        return x

    def _prepare_samplers(self):

        if self.test_mode:
            cv_sets = ['validation', 'test']
        else:
            cv_sets = ['train', 'validation']

        for cvset in cv_sets:

            self.samplers[cvset] = SimBatchSampler(
                data_source = self,
                idxs = self.idxs[cvset], 
                labels = self.data['y'][self.idxs[cvset]], 
                min_time = self.min_time, 
                max_time = self.max_time, 
                seed = self.starting_seed, 
                seed_range = self.seed_ranges[cvset], 
                test_mode = self.test_mode
            )

    def prepare_sampler_test_mode(self, cvset):

        self.samplers[cvset] = SimBatchSampler(
            data_source = self,
            idxs = self.idxs[cvset], 
            labels = self.data['y'][self.idxs[cvset]], 
            min_time = self.min_time, 
            max_time = self.max_time, 
            seed = self.starting_seed, 
            seed_range = self.seed_ranges[cvset], 
            test_mode = True
        )
        
    def reload_sampler(self, sampler):
        self.samplers[sampler] = SimBatchSampler(
            data_source = self,
            idxs = self.idxs[sampler], 
            labels = self.data['y'][self.idxs[sampler]], 
            min_time = self.min_time, 
            max_time = self.max_time, 
            seed = self.starting_seed, 
            seed_range = self.seed_ranges[sampler], 
            test_mode = self.test_mode
        )