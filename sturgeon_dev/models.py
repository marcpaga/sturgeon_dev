from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
torch.nn.functional.log_softmax
from pytorch_metric_learning import distances, losses, miners
from sklearn.metrics import balanced_accuracy_score

from sturgeon_dev.modelclasses import BaseModel, BaseTripletModel
from sturgeon_dev.constants import NUM_PROBES

class SparseMicroarrayModel(BaseModel):

    def __init__(self, num_classes, layer_sizes, weight = None, dropout = 0.5, *args, **kwargs):
        super(SparseMicroarrayModel, self).__init__(*args, **kwargs)

        self.model_type = 'ce'
        self.criterions['ce'] = nn.NLLLoss(weight)
        self.num_classes = num_classes
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        self.encoder = self.build_encoder(layer_sizes)
        self.decoder = nn.Sequential(nn.Linear(layer_sizes[-1], self.num_classes))

    def forward(self, x, ):

        emb = self.encoder(x)
        p = self.decoder(emb)

        return p
        
    def calculate_loss(self, y, p):
        
        loss = self.criterions['ce'](nn.functional.log_softmax(p, dim = -1), y)
        return loss, {'loss.global': loss.item(), 'loss.ce': loss.item()}

    def build_encoder(self, layer_sizes):
    
        if not isinstance(layer_sizes, list):
            layer_sizes = [layer_sizes]

        l = list()
        for i, s in enumerate(layer_sizes):
            if i == 0:
                input_channels = NUM_PROBES
                output_channels = s
            else:
                output_channels = s
            l.append(nn.Sequential(nn.Linear(input_channels, output_channels), nn.Sigmoid(), nn.Dropout(self.dropout)))
            input_channels = s

        return nn.Sequential(*l)

    def _add_to_save(self):

        self.save_dict['model_type'] = 'ce'
        self.save_dict['num_classes'] = self.num_classes
        self.save_dict['layer_sizes'] = self.layer_sizes
        self.save_dict['dropout'] = self.dropout

            

class TripletMicroarrayModel(BaseTripletModel):

    def __init__(self, num_classes, layer_sizes, type_of_triplets = "semihard", margin = 1, weight = None, dropout = 0.5, *args, **kwargs):
        super(TripletMicroarrayModel, self).__init__(*args, **kwargs)

        self.model_type = 'triplet'
        self.layer_sizes = layer_sizes
        self.num_classes = num_classes
        self.distance = distances.LpDistance()
        self.margin = margin
        self.type_of_triplets = type_of_triplets 
        self.miner = miners.TripletMarginMiner(
            margin = self.margin, 
            distance = self.distance, 
            type_of_triplets = self.type_of_triplets,
        )
        self.criterions['triplet'] = losses.ProxyNCALoss(
            num_classes = num_classes,
            embedding_size = layer_sizes[-1]
        )
        self.loss_optimizer = optim.Adam(self.criterions['triplet'].parameters(), lr=0.001)
        self.criterions['ce'] = nn.NLLLoss(weight)
        self.criterions_weights = {'ce':1.0, 'triplet':1.0}
        self.num_classes = num_classes
        self.dropout = dropout

        self.encoder = self.build_encoder(layer_sizes)
        self.decoder = nn.Sequential(
            nn.Linear(layer_sizes[-1], self.num_classes),
            nn.LogSoftmax(dim = -1)
        )
        

    def forward(self, x):

        emb = self.encoder(x)
        p = self.decoder(emb)

        return p, emb

    def build_encoder(self, layer_sizes):
    
        if not isinstance(layer_sizes, list):
            layer_sizes = [layer_sizes]

        l = list()
        for i, s in enumerate(layer_sizes):
            if i == 0:
                input_channels = NUM_PROBES
                output_channels = s
            else:
                output_channels = s
            l.append(nn.Sequential(nn.Linear(input_channels, output_channels), nn.Sigmoid(), nn.Dropout(self.dropout)))
            input_channels = s

        return nn.Sequential(*l)

    def _add_to_save(self):

        self.save_dict['model_type'] = 'triplet'
        self.save_dict['num_classes'] = self.num_classes
        self.save_dict['layer_sizes'] = self.layer_sizes
        self.save_dict['dropout'] = self.dropout

class AutoencoderMicroarrayModel(BaseModel):

    def __init__(self, layer_sizes, num_classes = None, weight = None, dropout = 0.5, factor = 1, *args, **kwargs):
        super(AutoencoderMicroarrayModel, self).__init__(*args, **kwargs)

        self.criterions['mse'] = nn.MSELoss()
        self.dropout = dropout
        self.factor = factor
        self.layer_sizes = layer_sizes
        self.num_classes = num_classes

        self.encoder = self.build_encoder(self.layer_sizes)
        self.decoder = self.build_decoder(self.layer_sizes[::-1])

    def forward(self, x):

        emb = self.encoder(x)
        p = self.decoder(emb)
        p *= self.factor
        p = torch.tanh(p)

        return p

    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(torch.float).to(self.device)
        y = batch['X'].to(torch.float).to(self.device)

        p = self.forward(x) # forward through the network
        loss, losses = self.calculate_loss(y = y, p = p)

        self.optimize(loss)
        
        return losses, p
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            y = batch['X'].to(torch.float).to(self.device)

            p = self.forward(x) # forward through the network
            _, losses = self.calculate_loss(y = y, p = p)
            
        return losses, p
        
    def calculate_loss(self, y, p):
        
        loss = self.criterions['mse'](p, y)
        return loss, {'loss.global': loss.item(), 'loss.mse': loss.item()}


    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [n samples]
            predictions (tensor): shape [samples, classes]
        """
        y = batch['X'].cpu().numpy()
        p = predictions.detach().cpu().numpy()
        p[p>0] = 1
        p[p<0] = -1
            
        return {'metric.autoencoder_accuracy': np.sum(y == p)/y.size}

    def build_encoder(self, layer_sizes):
    
        if not isinstance(layer_sizes, list):
            layer_sizes = [layer_sizes]

        l = list()
        for i, s in enumerate(layer_sizes):
            if i == 0:
                input_channels = NUM_PROBES
                output_channels = s
            else:
                output_channels = s
            l.append(nn.Sequential(nn.Linear(input_channels, output_channels), nn.Sigmoid(), nn.Dropout(self.dropout)))
            input_channels = s

        return nn.Sequential(*l)

    def build_decoder(self, layer_sizes):

        if not isinstance(layer_sizes, list):
            layer_sizes = [layer_sizes]

        l = list()
        for i, s in enumerate(layer_sizes):
            input_channels = s
            try:
                output_channels = layer_sizes[i+1]
            except IndexError:
                output_channels = NUM_PROBES
            
            if i == len(layer_sizes) - 1:
                l.append(nn.Sequential(nn.Linear(input_channels, output_channels)))
            else:
                l.append(nn.Sequential(nn.Linear(input_channels, output_channels), nn.Sigmoid(), nn.Dropout(self.dropout)))
            
        return nn.Sequential(*l)

class DoubleModel(AutoencoderMicroarrayModel):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(DoubleModel, self).__init__(*args, **kwargs)

        self.criterions = {
            "mse":nn.MSELoss(),
            "ce":nn.NLLLoss()
        }
        
        self.encoder = self.build_encoder(self.layer_sizes)
        self.decoder_recon = self.build_decoder(self.layer_sizes[::-1])
        self.decoder_predict = nn.Sequential(
            nn.Linear(self.layer_sizes[-1], self.num_classes),
            nn.LogSoftmax(dim = -1)
        )

    def forward(self, x): 

        emb = self.encoder(x)
        p = self.decoder_predict(emb)
        r = self.decoder_recon(emb)
        r *= self.factor
        r = torch.tanh(r)

        return p, r

    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(torch.float).to(self.device)
        y = batch['y'].to(self.device)
        X = batch['X'].to(torch.float).to(self.device)

        p, r = self.forward(x) # forward through the network
        loss, losses = self.calculate_loss(y = y, p = p, x = X, r = r)

        self.optimize(loss)
        
        return losses, {'p': p, 'r': r}
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            y = batch['y'].to(self.device)
            X = batch['X'].to(torch.float).to(self.device)

            p, r = self.forward(x) # forward through the network
            _, losses = self.calculate_loss(y = y, p = p, x = X, r = r)
            
        return losses, {'p': p, 'r': r}

    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            p, _ = self.forward(x)
            
        return p
        
    def calculate_loss(self, y, p, x, r):
        
        loss_mse = self.criterions['mse'](r, x)
        loss_ce = self.criterions['ce'](p, y)
        loss = loss_mse + loss_ce
        losses = {
            'loss.global': loss.item(), 
            'loss.mse': loss_mse.item(),
            'loss.ce': loss_ce.item()
        }
        return loss, losses

    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [n samples]
            predictions (tensor): shape [samples, classes]
        """
        x = batch['X'].cpu().numpy()
        r = predictions['r'].detach().cpu().numpy()
        r[r>0] = 1
        r[r<0] = -1

        y = batch['y'].cpu().numpy()
        p = predictions['p'].argmax(-1).cpu().numpy()
        accs = balanced_accuracy_score(y, p)

        metrics = {
            'metric.autoencoder_accuracy': np.sum(x == r)/x.size,
            'metric.balanced_accuracy': accs,
        }
            
        return metrics

    def _add_to_save(self):

        self.save_dict['model_type'] = 'double'
        self.save_dict['num_classes'] = self.num_classes
        self.save_dict['layer_sizes'] = self.layer_sizes
        self.save_dict['dropout'] = self.dropout

class TwoLevelSparseMicroarrayModel(BaseModel):

    def __init__(self, sub_to_fam_dict, num_classes, layer_sizes, weight = None, dropout = 0.5, *args, **kwargs):
        super(TwoLevelSparseMicroarrayModel, self).__init__(*args, **kwargs)

        self.model_type = 'ce'
        self.criterions['ce_fam'] = nn.NLLLoss()
        self.criterions['ce_sub'] = nn.NLLLoss()
        self.dropout = dropout
        self.sub_to_fam_dict = sub_to_fam_dict

        self.num_classes = num_classes
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        self.encoder1 = nn.Sequential(
            nn.Sequential(
                nn.Linear(NUM_PROBES, 512), 
                nn.SiLU(), 
                nn.Dropout(self.dropout)
            ),
            nn.Sequential(
                nn.Linear(512, 256), 
                nn.SiLU(), 
                nn.Dropout(self.dropout)
            ),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(256, 128), 
            nn.SiLU(), 
            nn.Dropout(self.dropout)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(256, 14)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 91)
        )

    def forward(self, x, ):

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        x_fam = self.decoder1(e1)
        x_sub = self.decoder2(e2)

        return x_fam, x_sub
        
    def calculate_loss(self, y_fam, y_sub, p_fam, p_sub):
        
        loss_fam = self.criterions['ce_fam'](nn.functional.log_softmax(p_fam, dim = -1), y_fam)
        loss_sub = self.criterions['ce_sub'](nn.functional.log_softmax(p_sub, dim = -1), y_sub)

        loss = loss_fam + loss_sub

        return loss, {'loss.global': loss.item(), 'loss.ce_fam': loss_fam.item(), 'loss.ce_sub': loss_sub.item()}

    def _add_to_save(self):

        self.save_dict['model_type'] = 'ce'
        self.save_dict['dropout'] = self.dropout

    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(torch.float).to(self.device)
        y_sub = deepcopy(batch['y'])
        y_fam = deepcopy(y_sub)

        y_fam = y_fam.apply_(lambda val: self.sub_to_fam_dict.get(val)).to(self.device)
        y_sub = y_sub.to(self.device)

        p_fam, p_sub = self.forward(x) # forward through the network
        loss, losses = self.calculate_loss(y_fam, y_sub, p_fam, p_sub)

        self.optimize(loss)
        
        return losses, p_sub.detach()
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            y_sub = deepcopy(batch['y'])
            y_fam = deepcopy(y_sub)

            y_fam = y_fam.apply_(lambda val: self.sub_to_fam_dict.get(val)).to(self.device)
            y_sub = y_sub.to(self.device)

            p_fam, p_sub = self.forward(x) # forward through the network
            _, losses = self.calculate_loss(y_fam, y_sub, p_fam, p_sub)
            
        return losses, p_sub

    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(torch.float).to(self.device)
            _, p = self.forward(x)
            
        return p