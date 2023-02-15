import os
import sys
import json
import zipfile
from copy import deepcopy
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import onnxruntime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sturgeon_dev.utils import softmax, merge_predictions

def load_sample(bed_file, probes_df):

    print('Reading methylation status of sample')
    sample_df = pd.read_csv(bed_file, header = 0, index_col = None, delim_whitespace=True)
    sample_name = Path(bed_file).stem

    sample_df.loc[sample_df['methylation_call'] == 0, 'methylation_call'] = -1
    sample_df = sample_df.set_index('probe_id')
    sample_df = sample_df.reindex(index=probes_df['ID_REF'])
    sample_df = sample_df.reset_index()

    x = np.transpose(np.array(sample_df.iloc[:, 4:]))
    x = x.astype(np.float32)
    x[np.isnan(x)] = 0
    measured_probes = np.where(x[0] != 0)[0]
    print('\tTotal number of probes: ' + str(len(measured_probes)))

    return x, sample_name, measured_probes
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", 
        type=str, 
        help='Dir with bed files'
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        help='Dir where to write results', 
    )
    parser.add_argument(
        "-m", "--model-file",
        type=str,
        help='Model file'
    )
    args = parser.parse_args()

    ############################################################################
    # Loading data
    ############################################################################

    with zipfile.ZipFile(args.model_file, 'r') as zipf:
        probes_df = pd.read_csv(zipf.open('probes.csv'), header = 0, index_col = None)

    
    ############################################################################
    # Prediction
    ############################################################################

    print('Running inference', flush=True)
    print('\tStarting session', flush=True)

    with zipfile.ZipFile(args.model_file, 'r') as zipf:
        
        weight_matrix = np.load(zipf.open('weight_scores.npz'))
        temperatures = np.load(zipf.open('calibration.npy'))
        temperatures = temperatures.flatten()
        mean_probes_per_timepoint = weight_matrix['avgsites']
        accuracy_per_timepoint_per_model = weight_matrix['performance']

        so = onnxruntime.SessionOptions()

        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        ort_session = onnxruntime.InferenceSession(
            zipf.read('model.onnx'), 
            providers = ['CPUExecutionProvider','CUDAExecutionProvider'],
            sess_options = so,
        )

        for bed_file in os.listdir(args.input_dir):
            print('Predicting: ' + bed_file, flush=True)

            decoding_dict = json.load(zipf.open('decoding.json'))
            try:
                merge_dict = json.load(zipf.open('merge.json'))
            except KeyError:
                merge_dict = None

            if not bed_file.endswith('.bed'):
                continue
            
            bed_file = os.path.join(args.input_dir, bed_file)

            x, sample_name, measured_probes = load_sample(bed_file, probes_df)

            # compute ONNX Runtime output prediction
            ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x)
            ort_inputs = {ort_session.get_inputs()[0].name: ortvalue}
            ort_outs = ort_session.run(
                ['predictions'], 
                ort_inputs,
            )[0]

            uncalibrated_scores = ort_outs[0]

            ############################################################################
            # Score calibration
            ############################################################################

            calibrated_scores = np.empty_like(uncalibrated_scores)
            for i in range(uncalibrated_scores.shape[0]):
                calibrated_scores[i, :]  = np.exp(softmax(deepcopy(uncalibrated_scores[i, :])/temperatures[i]))
                uncalibrated_scores[i, :] = np.exp(softmax(uncalibrated_scores[i, :]))

            ############################################################################
            # Make a table
            ############################################################################

            calibrated_df = dict()
            for k, v in decoding_dict.items():
                calibrated_df[v] = calibrated_scores[:, int(k)]
            calibrated_df = pd.DataFrame(calibrated_df)

            uncalibrated_df = dict()
            for k, v in decoding_dict.items():
                uncalibrated_df[v] = uncalibrated_scores[:, int(k)]
            uncalibrated_df = pd.DataFrame(uncalibrated_df)

            if merge_dict is not None:
                calibrated_df, _ = merge_predictions(calibrated_df, decoding_dict, merge_dict)
                uncalibrated_df, decoding_dict = merge_predictions(uncalibrated_df, decoding_dict, merge_dict)

            encoding_dict = {v:k for k, v in decoding_dict.items()}
            ############################################################################
            # Take weighted mean of ensemble 
            ############################################################################

            n = len(measured_probes)
            calculated_weights = np.ones(
                (
                    accuracy_per_timepoint_per_model.shape[0],
                    accuracy_per_timepoint_per_model.shape[2],
                ), 
            dtype=float)

            for m in range(calculated_weights.shape[0]):

                weights = deepcopy(accuracy_per_timepoint_per_model[m])
                n_probes = mean_probes_per_timepoint[m]
                t = n_probes.searchsorted(n)
                t = int(t)
                if t == weights.shape[0]:
                    calculated_weights[m, :] = weights[t-1]
                elif t == 0:
                    calculated_weights[m, :] = weights[t]
                else:
                    weights = weights[t-1:t+1]
                    x = [n_probes[t-1], n_probes[t]]
                    for i in range(weights.shape[1]):
                        y = weights[:, i]
                        calculated_weights[m, i] = np.interp(n, x, y)

            
            avg_scores = dict()
            for colname in calibrated_df.columns:
                avg_scores[colname] = [
                    float(np.average(
                        calibrated_df[colname], 
                        weights = calculated_weights[:, int(encoding_dict[colname])]
                    ))
                ]
            max_scores = dict()
            for colname in calibrated_df.columns:
                max_scores[colname] = [np.max(np.array(calibrated_df[colname])).item()]

            best_scores = dict()
            for colname in calibrated_df.columns:
                best_scores[colname] = [np.array(calibrated_df[colname])[np.argmax(calculated_weights[:, int(encoding_dict[colname])])]]

            max_model_scores = dict()
            arr = np.array(calibrated_df[calibrated_df.columns])
            best_m = np.where(arr == np.max(arr))[0]
            if len(best_m) > 0:
                best_m = best_m[0]
            for colname in calibrated_df.columns:
                max_model_scores[colname] = [np.array(calibrated_df[colname])[best_m].item()]

            calibrated_df = pd.concat([
                calibrated_df, 
                pd.DataFrame(avg_scores, index = [0]),
                pd.DataFrame(max_scores, index = [0]),
                pd.DataFrame(best_scores, index = [0]),
                pd.DataFrame(max_model_scores, index = [0]),
            ])
            calibrated_df['number_probes'] = len(measured_probes)
            calibrated_df['model_num'] = ['0', '1', '2', '3', 'ensemble', 'max', 'best', 'max_model']
            calibrated_df['calibrated'] = True


            avg_scores = dict()
            for colname in uncalibrated_df.columns:
                avg_scores[colname] = [
                    float(np.average(
                        uncalibrated_df[colname], 
                        weights = calculated_weights[:, int(encoding_dict[colname])]
                    ))
                ]

            max_scores = dict()
            for colname in uncalibrated_df.columns:
                max_scores[colname] = [np.max(np.array(uncalibrated_df[colname])).item()]

            best_scores = dict()
            for colname in uncalibrated_df.columns:
                best_scores[colname] = [np.array(uncalibrated_df[colname])[np.argmax(calculated_weights[:, int(encoding_dict[colname])])]]

            max_model_scores = dict()
            arr = np.array(uncalibrated_df[uncalibrated_df.columns])
            best_m = np.where(arr == np.max(arr))[0]
            if len(best_m) > 0:
                best_m = best_m[0]
            for colname in uncalibrated_df.columns:
                max_model_scores[colname] = [np.array(uncalibrated_df[colname])[best_m].item()]


            uncalibrated_df = pd.concat([
                uncalibrated_df, 
                pd.DataFrame(avg_scores, index = [0]),
                pd.DataFrame(max_scores, index = [0]),
                pd.DataFrame(best_scores, index = [0]),
                pd.DataFrame(max_model_scores, index = [0]),
            ])
            uncalibrated_df['number_probes'] = len(measured_probes)
            uncalibrated_df['model_num'] = ['0', '1', '2', '3', 'ensemble', 'max', 'best', 'max_model']
            uncalibrated_df['calibrated'] = False

            ############################################################################
            # Output table
            ############################################################################

            output_df = pd.concat([uncalibrated_df, calibrated_df])

            print('Saving: ' + os.path.join(args.output_dir, sample_name+'.csv'), flush=True)

            pd.DataFrame(output_df).to_csv(
                os.path.join(args.output_dir, sample_name+'.csv'),
                header = True,
                index = False,
            )
