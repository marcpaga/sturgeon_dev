import numpy as np
import pandas as pd

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, help='File with the methylation data')
    parser.add_argument("--annotation-file", type=str, help='File with the samples metadata')
    parser.add_argument("--probes-csv", type=str, help='File with the probe info')
    parser.add_argument("--output-file", type=str, help='Path were to save the output npz file')
    parser.add_argument("--bin-threshold", type=float, default=0.6, help='Threshold to call a probe methylated')
    parser.add_argument("--methylated-encoding", type=int, default= 1, help='Value for methylated status')
    parser.add_argument("--unmethylated-encoding", type=int, default= -1, help='Value for unmethylated status')
    args = parser.parse_args()

    probes = list()
    measurements = list()

    with open(args.data_file, 'r') as f:
        for linenum, line in enumerate(f):
            if linenum == 0:
                header = np.array(line.strip('\n').split('\t'))
                probe_col = 0
                measurements_cols = np.arange(1, len(header), 2)
                continue
            
            line = np.array(line.strip('\n').split('\t'))
            probes.append(line[0])
            vals = line[measurements_cols].astype(float)
            vals[vals >= args.bin_threshold] = 1
            vals[vals < args.bin_threshold] = 0
            vals = vals.astype(int)
            measurements.append(vals)

    measurements = np.vstack(measurements)
    probes_arr = np.array(probes)
    labels = np.array(pd.read_csv(args.annotation_file)['diagnostics_class'].tolist(),).astype(str)

    print(measurements.shape)
    print(probes_arr.shape)
    print(labels.shape)
    assert measurements.shape[0] == probes_arr.shape[0]
    assert measurements.shape[1] == len(labels)


    # subset and order probes to probes.csv
    probes_df = pd.read_csv(args.probes_csv, header = 0)
    available_probes = np.array(probes_df['ID_REF']).astype(str)

    probes_in = np.isin(probes_arr, available_probes)

    measurements = measurements[probes_in]
    probes_arr = probes_arr[probes_in]

    sample_df = pd.DataFrame({
        'probe_name': probes_arr,
        'pos': np.arange(0, len(probes_arr), 1)
    })
    sample_df = sample_df.set_index('probe_name')
    sample_df = sample_df.reindex(index=probes_df['ID_REF'])
    sample_df = sample_df.reset_index()
    new_order = np.array(sample_df['pos'])

    assert len(new_order) == len(np.unique(new_order))
    assert np.sum(new_order > 0) == len(new_order)-1
    assert np.all(probes_arr[new_order] == available_probes)

    measurements = measurements[new_order]
    probes_arr = probes_arr[new_order]

    measurements = measurements.transpose(1, 0)

    if args.methylated_encoding != 1:
        measurements[measurements == 1] = args.methylated_encoding

    if args.unmethylated_encoding != 0:
         measurements[measurements == 0] = args.unmethylated_encoding

    np.savez(args.output_file, x = measurements, probes = probes_arr, diagnostics = labels)
