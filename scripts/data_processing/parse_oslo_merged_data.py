import numpy as np
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help='File with the metadata')
    parser.add_argument("--output-file", type=str, help='Output file csv')
    args = parser.parse_args()


    tsv_file = args.input_file

    sample_names = list()
    with open(tsv_file, 'r') as handle:
        for i, line in tqdm(enumerate(handle), total = 416):
            line = line.strip('\n').split(',')
            if i == 0:
                probes = line[1:]
                x = np.zeros((416, len(probes)), dtype = 'U1')
            else:
                sample_names.append(line[0])
                measurements = line[1:]
                x[i, :] = measurements
                
    x[x == '0'] = '2'
    x[x == ''] = '0'
    x = x.astype(int)
    x[x == 2] = -1

    np.savez_compressed(
        args.output_file,
        x = x,
        sample_names = np.array(sample_names),
        probe_names = np.array(probes),
    )