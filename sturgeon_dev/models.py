from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
torch.nn.functional.log_softmax
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

            
