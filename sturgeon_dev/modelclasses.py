import os
from abc import abstractmethod
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
import psutil

from sturgeon_dev.utils import generate_log_df, clean_checkpoints
from sturgeon_dev.constants import DIAGNOSTICS_DECODING

class BaseModel(nn.Module):

    def __init__(
        self, 
        device, 
        dataset,
        optimizer = None, 
        schedulers = dict(), 
        criterions = dict(), 
        criterions_weights = dict(),
        clipping_value = 2, 
        adaptive_sampling_correction = 0.3,
        sub_to_fam_dict = None,
        other_class = None,
    ):
        super(BaseModel, self).__init__()

        self.device = device
        self.dataset = dataset

        # optimization
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterions = criterions
        self.criterions_weights = criterions_weights
        self.clipping_value = clipping_value
        self.adaptive_sampling_correction = adaptive_sampling_correction
        self.sub_to_fam_dict = sub_to_fam_dict
        self.other_class = other_class
        self.use_adaptive_sampling = False

        self.save_dict = dict()
        self.init_weights()

    @abstractmethod
    def forward(self, batch):
        """Forward through the network
        """
        raise NotImplementedError()
    
    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(torch.float).to(self.device)
        if len(x.shape) > 2:
            x = x.squeeze(0)
        y = batch['y'].to(self.device)
        if len(y.shape) > 1:
            y = y.squeeze(0)

        if self.other_class:
            y[y >= self.other_class] = self.other_class

        p = self.forward(x) # forward through the network
        loss, losses = self.calculate_loss(y = y, p = p)

        self.optimize(loss)
        
        return losses, p.detach()
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            if len(x.shape) > 2:
                x = x.squeeze(0)
            y = batch['y'].to(self.device)
            if len(y.shape) > 1:
                y = y.squeeze(0)
            
            if self.other_class:
                y[y >= self.other_class] = self.other_class

            p = self.forward(x) # forward through the network
            _, losses = self.calculate_loss(y = y, p = p)
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            p = self.forward(x)
            
        return p


    @abstractmethod    
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels
            p (tensor): tensor with predictions
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, 
            the weighed sum is named global_loss
        """
        
        raise NotImplementedError()
        return loss, losses
    
    
    def optimize(self, loss):
        """Optimizes the model by calculating the loss and doing backpropagation
        
        Args:
            loss (float): calculated loss that can be backpropagated
        """
        
       
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            self.clipping_value
        )
        self.optimizer.step()
        
        for scheduler in self.schedulers.values():
            if scheduler:
                scheduler.step()
            
        return None

    def init_adaptative_sampling_arrays(self):

        self.counts_array = np.ones(
            (len(np.arange(self.dataset.min_time, self.dataset.max_time)), len(self.dataset.label_decoding)),
            dtype=int,
        )
        self.correct_array = np.ones(
            self.counts_array.shape,
            dtype=int,
        )

    def update_adaptive_sampling(self, labels, predictions, timepoints):

        for y, p, t in zip(labels, predictions, timepoints):
            self.counts_array[t, y] += 1
            if y == p:
                self.correct_array[t, y] += 1

        self.warray = 1 - self.correct_array/self.counts_array + self.adaptive_sampling_correction
        self.warray /= np.sum(self.warray)
        return self.warray

    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [n samples]
            predictions (tensor): shape [samples, classes]
        """
        labels = batch['y'].cpu().numpy()
        if len(labels.shape) > 1:
            labels = labels[0]
        if self.other_class:
            labels[labels >= self.other_class] = self.other_class
        predictions = predictions.argmax(-1).cpu().numpy()
        timepoints = batch['timepoint'].cpu().numpy()
        accs = balanced_accuracy_score(labels, predictions)

        if self.use_adaptive_sampling:
            self.update_adaptive_sampling(labels, predictions, timepoints)
            
        return {'metric.balanced_accuracy': accs}
    
    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for _, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, checkpoint_file):
        """Save the model state
        """
            
        self.save_dict['model_state'] = self.state_dict()
        self.save_dict['optimizer_state'] = self.optimizer.state_dict()
        for k, v in self.schedulers.items():
            self.save_dict[k + '_state'] = v.state_dict()
        self.save_dict['idxs'] = self.dataset.idxs
        self.save_dict['encoding_dict'] = self.dataset.label_encoding
        self.save_dict['decoding_dict'] = self.dataset.label_decoding
        self.save_dict['color_dict'] = self.dataset.label_colors
        self._add_to_save()

        torch.save(self.save_dict, checkpoint_file)

        return self.save_dict

    def generate_correctness_arrays(self):

        self.counts_array = np.zeros(
            (len(np.arange(self.dataset.min_time, self.dataset.max_time)), len(self.dataset.decoding_dict)),
            dtype=int
        )
        self.correct_array = np.zeros(
            self.counts_array.shape,
            dtype=int
        )

    @abstractmethod
    def _add_to_save(self):
        raise NotImplementedError

        
    @abstractmethod
    def load_default_configuration(self, default_all = False):
        """Method to load default model configuration
        """
        raise NotImplementedError()

    def train_loop(
        self,
        batch_size,
        num_epochs,
        output_dir,
        adaptive_sampling,
        checkpoint_every,
        validation_multiplier,
        max_checkpoints,
        checkpoint_metric,
        checkpoint_metricdirection,
        processes,
        adapt_every = 1,
        logger = None,
        progress_bar = True,
        prev_train_batch_num = 0,
    ):

        if adaptive_sampling:
            self.use_adaptive_sampling = True
            self.init_adaptative_sampling_arrays()

        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        # keep track of losses and metrics to take the average
        train_batch_num = 0 + prev_train_batch_num
        train_results = dict()
        dataloaders = dict()
        for n_epoch in range(num_epochs):

            self.dataset.reload_sampler(sampler = 'train')
            dataloaders['train'] = DataLoader(
                dataset = self.dataset,
                batch_size = batch_size,
                sampler = self.dataset.samplers['train'], 
                num_workers = processes-1,
                pin_memory = False,
                prefetch_factor = 4,
            )
            
            self.dataset.reload_sampler(sampler = 'validation')
            dataloaders['validation'] = DataLoader(
                dataset = self.dataset,
                batch_size = batch_size,
                sampler = self.dataset.samplers['validation'], 
                num_workers = 1,
                pin_memory = False,
                prefetch_factor = validation_multiplier,
            )

            # use this to restart the in case we finish all the validation data
            val_iterator = iter(dataloaders['validation']) 
            
            st_time = time.time()
            # iterate over the train data
            for _, train_batch in enumerate(dataloaders['train']):
                
                if logger:
                    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                    logger.info('Training step number: {}, mem: {}'.format(train_batch_num, mem_mb))

                losses, predictions = self.train_step(train_batch)
                metrics = self.evaluate(train_batch, predictions)
                train_batch_num += 1

                for results in [losses, metrics]:            
                    for k, v in results.items():
                        if k not in train_results.keys():
                            train_results[k] = list()
                        train_results[k].append(v)

                if train_batch_num % checkpoint_every == 0:

                    if logger:
                        logger.info('Checkpoint')
                    valid_results = dict()
                    for _ in range(validation_multiplier):
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            if logger:
                                logger.info('Resetting validation dataloader')
                            self.dataset.reload_sampler(sampler = 'validation')
                            dataloaders['validation'] = DataLoader(
                                dataset = self.dataset,
                                batch_size = batch_size,
                                sampler = self.dataset.samplers['validation'], 
                                num_workers = 1,
                                pin_memory = False,
                                prefetch_factor = validation_multiplier,
                            )
                            val_iterator = iter(dataloaders['validation'])
                            val_batch = next(val_iterator)
                                        
                        # calculate and log the validation results
                        losses, predictions = self.validation_step(val_batch)
                        metrics = self.evaluate(val_batch, predictions)

                        for results in [losses, metrics]:            
                            for k, v in results.items():
                                if k not in valid_results.keys():
                                    valid_results[k] = list()
                                valid_results[k].append(v)

                    # log the train results
                    log_df = generate_log_df(list(losses.keys()) + list(metrics.keys()))
                    for cv, cv_dict in zip(['train', 'val'], [train_results, valid_results]):
                        for k, v in cv_dict.items():
                            if k == 'total_samples':
                                continue
                            log_df[k + '.' + cv] = np.mean(v[-validation_multiplier:])
                    train_results = dict()
                    valid_results = dict()
                    
                    # calculate time it took since last validation step
                    log_df['step'] = str(train_batch_num)
                    log_df['time'] = int(time.time() - st_time)
                    for param_group in self.optimizer.param_groups:
                        log_df['lr'] = param_group['lr']
                    st_time = time.time()
                        
                    log_df['checkpoint'] = 'yes'
                    
                    if logger:
                        logger.info('Saving checkpoint: ' + str(train_batch_num))

                    self.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(train_batch_num) + '.pt'))
                
                    if logger:
                        logger.info('Writing training log')
                    # write to log
                    if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                        log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                        header=True, index=False)
                    else: # else it exists so append without writing the header
                        log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                        mode='a', header=False, index=False)

                    if logger:
                        logger.info('Cleaning checkpoints')
                    clean_checkpoints(
                        log_file = os.path.join(output_dir, 'train.log'), 
                        checkpoint_dir = checkpoints_dir, 
                        max_checkpoints = max_checkpoints, 
                        metric = checkpoint_metric, 
                        max_or_min = checkpoint_metricdirection,
                    )

            if adaptive_sampling:
                if (n_epoch+1) % adapt_every == 0: 
                
                    self.dataset.samplers['train'].add_weights_array(self.warray)

                    if logger:
                        logger.info('Applying adaptive sampling')
                    Path(os.path.join(output_dir, 'adaptive')).mkdir(parents=True, exist_ok=True)

                if (n_epoch+1) % 25 == 0: 
                    np.save(
                        os.path.join(output_dir, 'adaptive', 'accuracy_{n}.npy'.format(n = n_epoch)),
                        self.correct_array/self.counts_array,
                    )
                    np.save(
                        os.path.join(output_dir, 'adaptive', 'samples_{n}.npy'.format(n = n_epoch)),
                        (self.warray * len(self.dataset.samplers['train'])).astype(int),
                    )

        return int(train_batch_num)
                

    def test_loop(
        self,
        cvset,
        batch_size,
        output_file,
        decoding_dict,
        processes,
        logger,
        progress_bar
    ):

        self.dataset.prepare_sampler_test_mode(cvset = cvset)
        self.dataset.reload_sampler(sampler = cvset)
        dataloader = DataLoader(
            dataset = self.dataset,
            batch_size = batch_size,
            sampler = self.dataset.samplers[cvset], 
            num_workers = processes,
        )

        for batch_num, batch in enumerate(dataloader):

            if logger:
                logger.info('Testing step number: ' + str(batch_num))

            predictions = self.predict_step(batch)

            results = {
                "Idx": list(),
                "Diagnostics_label": list(),
                "Label": list(),
                "Timepoint": list(),
                "Seed_pregen": list(),
                "NSites": list(),
            }
            for v in decoding_dict.values():
                results[v] = list()
            
            results['Idx'].append(batch['idx'].cpu().numpy().flatten())
            for y in batch['diagnostics'].cpu().numpy():
                results['Diagnostics_label'].append(DIAGNOSTICS_DECODING[y])
            for y in batch['y'].cpu().numpy():
                results['Label'].append(decoding_dict[y])
            results['Timepoint'].append(batch['timepoint'].cpu().numpy())
            results['Seed_pregen'].append(batch['seed_pregen'].cpu().numpy())
            results['NSites'].append(torch.sum(batch['x'] != 0, 1).cpu().numpy())

            for i in range(predictions.shape[1]):
                results[decoding_dict[i]].append(predictions[:, i].cpu().numpy())

            for k, v in results.items():
                try:
                    results[k] = np.concatenate(v)
                except ValueError:
                    results[k] = np.array(v)

            results = pd.DataFrame(results)
            if os.path.isfile(output_file):
                results.to_csv(output_file, mode='a', header = False, index = False)
            else:
                results.to_csv(output_file, mode='w', header = True, index = False)


class EnsembleModel(nn.Module):

    def __init__(
        self, 
        models, 
        device
    ):
        super(EnsembleModel, self).__init__()

        self.models = torch.nn.ModuleList(models)
        self.device = device

    def forward(self, x):
        """
        Args:
            x (torch.tensor): with dimensions [batch, channels]

        Returns:
            A torch.Tensor with dimensions [batch, num_models, channels]
        """

        p_list = list()
        for m in self.models:
            p_list.append(m(x))
        return torch.stack(p_list).permute(1, 0, 2)
