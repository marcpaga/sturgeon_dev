import os
import sys
import argparse
import json
from zipfile import ZipFile

import numpy as np
import torch

sys.path.append('/hpc/compgen/users/mpages/deepsurg')
from sturgeon_dev.modelclasses import EnsembleModel
from sturgeon_dev.utils import load_labels_encoding, mean_color

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

probes_file = '/hpc/compgen/users/mpages/deepsurg/static/probes.csv'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--prediction-type', type=str)
    
    args = parser.parse_args()

    print('Loading models and calibration matrices')
    models = list()
    for model_num in os.listdir(args.model_dir):

        try:
            int(model_num)
        except ValueError:
            continue

        models.append({
            "checkpoint": os.path.join(args.model_dir, model_num, 'checkpoints/checkpoint_best.pt'),
            "calibration": os.path.join(args.model_dir, model_num, 'calibrated_temperature.npy'),
        })

    loaded_models = list()
    loaded_calibration_matrices = list()

    for model in models:

        loaded_calibration_matrices.append(np.load(model['calibration']))

        checkpoint_dict = torch.load(model['checkpoint'], map_location=device)
        model_type = checkpoint_dict['model_type']
        layer_sizes = checkpoint_dict['layer_sizes']
        decoding_dict = checkpoint_dict['decoding_dict']
        state_dict = checkpoint_dict['model_state']
        color_dict = checkpoint_dict['color_dict']

        if model_type == 'ce':
            from sturgeon_dev.models import SparseMicroarrayModel as Model # pyright: reportMissingImports=false
        elif model_type == 'triplet':
            from sturgeon_dev.models import TripletMicroarrayModel as Model # pyright: reportMissingImports=false

        model = Model(
            dataset = None,
            num_classes = len(decoding_dict),
            dropout = 0,
            layer_sizes = layer_sizes,
            device = device,
        )
        model = model.to(device)

        model.load_state_dict(state_dict, strict = True)
        model.to(device)
        loaded_models.append(model)

    calibration_matrices = np.dstack(loaded_calibration_matrices)

    ensemble_model = EnsembleModel(
        models = loaded_models, 
        device = device,
    )

    print('Exporting onnx model')
    onnx_file = os.path.join(args.output_dir, 'model.onnx')
    x = torch.randn(64, ensemble_model.models[0].encoder[0][0].in_features, requires_grad=True).to(device)
    
    torch.onnx.export(
        ensemble_model,               
        x,                        
        onnx_file,  
        export_params=True,       
        opset_version=10,          
        do_constant_folding=True, 
        input_names = ['x'],  
        output_names = ['predictions'], 
        dynamic_axes= {
            'x' : {0 : 'batch_size'},
            'predictions' : {0 : 'batch_size'},
        }
    )

    _, _, merge_dict = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__), 
            '../static/'+args.prediction_type+'.json'
        ),
        return_merge = True
    )

    print('Writing additional files')
    np.save(os.path.join(args.output_dir, 'calibration.npy'), calibration_matrices)
    with open(os.path.join(args.output_dir, 'decoding.json'), 'w') as fp:
        json.dump(decoding_dict, fp)

    if merge_dict is not None:
        with open(os.path.join(args.output_dir, 'merge.json'), 'w') as fp:
            json.dump(merge_dict, fp)

        for k, v in merge_dict.items():
            
            colors = list()
            for c in v:
                colors.append(color_dict[c])
            
            color_dict[k] = mean_color(colors)

    with open(os.path.join(args.output_dir, 'colors.json'), 'w') as fp:
        json.dump(color_dict, fp)

    print('Compressing in zip')
    with ZipFile(os.path.join(args.output_dir, 'model.zip'), 'w') as zipf:
        zipf.write(os.path.join(args.output_dir, 'model.onnx'), arcname='model.onnx')
        zipf.write(os.path.join(args.output_dir, 'colors.json'), arcname='colors.json')
        zipf.write(os.path.join(args.output_dir, 'decoding.json'), arcname='decoding.json')
        zipf.write(os.path.join(args.output_dir, 'calibration.npy'), arcname='calibration.npy')
        zipf.write(os.path.join(args.output_dir, 'weight_scores.npz'), arcname='weight_scores.npz')
        if merge_dict is not None:
            zipf.write(os.path.join(args.output_dir, 'merge.json'), arcname='merge.json')
        zipf.write(probes_file, arcname='probes.csv')

    print('Deleting temporary files')
    os.remove(os.path.join(args.output_dir, 'model.onnx'))
    os.remove(os.path.join(args.output_dir, 'colors.json'))
    os.remove(os.path.join(args.output_dir, 'decoding.json'))
    os.remove(os.path.join(args.output_dir, 'calibration.npy'))

    sys.exit()