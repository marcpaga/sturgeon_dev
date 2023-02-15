"""
This script pre-processes the microarray data into a format that can be used
for training the neuralnetwork. It does:
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils import load_labels_encoding

import pandas as pd

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, help='File with the samples metadata')
    parser.add_argument("--output-file", type=str, help='Path were to save the output files')
    args = parser.parse_args()

    encoding_dict, _ = load_labels_encoding(
        os.path.join(
            os.path.dirname(__file__),
            '../../static/diagnostics.json'
        )
    )


    ### TRAIN DATA ANNOTATION ###
    # get the sample names 
    with open(args.annotation_file, 'r') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                break

    samples = list()

    # match the sample names to the names that we have in the constants
    line = line.strip('\n').split('\t')
    for i in range(len(line)):
        if i == 0:
            continue
        sample = line[i].strip('"').split(',')
        mainclass = sample[0].strip()
        subclass = sample[1].strip()
        sampleclass = " - ".join([mainclass, subclass])
        label_found = False
        for k in encoding_dict.keys():
            if k.endswith(sampleclass):
                label_found = True
                break
        # some hardcoding for different names
        if not label_found:
            if not label_found:
                if mainclass == 'PITUI':
                    mainclass = 'PITUI SCO GCT'
                    subclass = 'SCO GCT'
                elif mainclass == 'PITAD' and subclass == 'ACTH':
                    mainclass = 'PITAD ACTH'
                elif mainclass == 'PITAD' and subclass == 'STH DNS A':
                    mainclass = 'PITAD STH'
                elif mainclass == 'PITAD' and subclass == 'STH DNS B':
                    mainclass = 'PITAD STH'
                elif mainclass == 'PITAD' and subclass == 'STH SPA':
                    mainclass = 'PITAD STH'
                elif mainclass == 'PITAD' and subclass == 'PRL':
                    mainclass = 'PITAD PRL'
                elif mainclass == 'PITAD' and subclass == 'TSH':
                    mainclass = 'PITAD TSH'
                elif mainclass == 'PITAD' and subclass == 'FSH LH':
                    mainclass = 'PITAD FSH LH'
                elif mainclass == 'SCHW' and subclass == 'MEL':
                    subclass = 'SCHW MEL'
                elif mainclass == 'LGG' and subclass == 'PA PF':
                    mainclass = 'LGG PA'
                elif mainclass == 'LGG' and subclass == 'PA MID':
                    mainclass = 'LGG PA'
                elif mainclass == 'LGG' and subclass == 'PA/GG ST':
                    mainclass = 'LGG PA'
                elif mainclass == 'LGG' and subclass == 'SEGA':
                    mainclass = 'LGG SEGA'
                elif mainclass == 'MB' and subclass == 'WNT':
                    mainclass = 'MB WNT'
                elif mainclass == 'MB' and subclass == 'G3':
                    mainclass = 'MB G3G4'
                elif mainclass == 'MB' and subclass == 'G4':
                    mainclass = 'MB G3G4'
                elif mainclass == 'MB' and subclass == 'SHH INF':
                    mainclass = 'MB SHH'
                    subclass = 'INF'
                elif mainclass == 'MB' and subclass == 'SHH CHL AD':
                    mainclass = 'MB SHH'
                    subclass = 'CHL AD'
                elif mainclass == 'LGG' and subclass == 'MYB':
                    mainclass = 'LGG MYB'
                else:
                    subclass = mainclass
            sampleclass = " - ".join([mainclass, subclass])
        for k in encoding_dict.keys():
            if k.endswith(sampleclass):
                label_found = True
                break
        if not label_found:
            raise ValueError('Sample cannot be matched to class name: ' + sample)

        samples.append({
            'sample_number': int(sample[-1].strip().split(' ')[1]),
            'diagnostics_class': k,
        })
    # write the sample classes for each sample in the output file
    samples_df = pd.DataFrame(samples)
    samples_df.to_csv(args.output_file, header = True, index = None)