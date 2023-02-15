import os
import sys
import argparse

import torch
from torch import nn, optim
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.utils import load_labels_encoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-file", type=str, help='Path where the data for the dataloaders is stored', required = True)
    parser.add_argument("--output-file", type=str, help='Path where the model is saved', required = True)
    parser.add_argument("--prediction-type", type=str, required = True)
    parser.add_argument("--learning-rate", type=float, default = 0.01)
    parser.add_argument("--num-iterations", type=int, default = 500)

    args = parser.parse_args()

    _, decoding_dict = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__), 
            '../static/'+args.prediction_type+'.json'
        )
    )

    val_df = pd.read_csv(args.validation_file, header = 0, index_col = None)

    rev_decoding_dict = {v:k for k, v in decoding_dict.items()}

    # convert labels to encoding
    labels = np.vectorize(rev_decoding_dict.get)(np.array(val_df['Label']))
    # calculate relative weights of each class given the amount of logits per class
    _, label_counts = np.unique(np.sort(labels), return_counts = True)
    weights = 1 / (label_counts / np.sum(label_counts))
    weights = weights / np.sum(weights)
    weights = torch.from_numpy(weights).to(device)
    labels = torch.from_numpy(labels).to(device)

    # initial temperature
    temperature = nn.Parameter(torch.ones(1, device = device), requires_grad = True)

    # predicted scores
    logits = torch.from_numpy(np.array(val_df[list(decoding_dict.values())])).to(device)
    nll_criterion = nn.CrossEntropyLoss(weight = weights).to(device)

    def temperature_scale(logits, temperature):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    optimizer = optim.LBFGS([temperature], lr=args.learning_rate, max_iter=args.num_iterations)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    np.save(
        args.output_file,
        temperature.detach().cpu().numpy()
    )


    