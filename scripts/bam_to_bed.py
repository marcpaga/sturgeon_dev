import os
import multiprocessing
import argparse

import numpy as np
import pandas as pd
from modbampy import ModBam

def get_methylation_status(chr_probes_df, bam_file, margin, score_threshold = 0.6):
    """Get the methylation status of probes from a guppy methylation bam file

    Args:
        chr_probes_df (pd.DataFrame): data.frame with the probe locations for
            one chromosome. Expected columns:
                - chr
                - start
                - end
                - methylation_calls
                - unmethylation_calls
                - total_calls
        bam_file (str): path to the bam file
        margin (int): location margin around the probe location, methylation
            score will be averaged.
        score_threshold (float): average scores higher than this will be 
            considered methylated

    Returns:
        Same pd.DataFrame as the input with the additional calls to each probe
    """

    chr_probes_df.reset_index(drop = True, inplace = True)

    results = {
        'read_id': list(),
        'chr': list(),
        'reference_pos': list(),
        'strand': list(),
        'score': list()
    }
    c = np.unique(chr_probes_df['chr']).item()

    with ModBam(bam_file) as bam:

        st = 0
        nd = chr_probes_df['end'].max()

        chrom = 'chr'+str(c)

        for read in bam.reads(chrom, st, nd):
            
            for pos_mod in read.mod_sites:
                read_id, reference_pos, _, strand, _, _, _, score = pos_mod

                if reference_pos == -1:
                    continue

                results['read_id'].append(read_id)
                results['chr'].append(c)
                results['reference_pos'].append(reference_pos)
                results['strand'].append(strand)
                results['score'].append((1+score)/256)

    results = pd.DataFrame(results)
    results = results.sort_values(['reference_pos'])

    starts = np.array(chr_probes_df['start']) - margin
    ends = starts + margin * 2 + 1

    queries = np.array(results['reference_pos'])
    scores = np.array(results['score'])

    s = np.searchsorted(queries, starts, 'left')
    n = np.searchsorted(queries, ends, 'right')

    f = np.where(s != n)[0]

    s = s[f]
    n = n[f]

    for ss, nn, ff in zip(s, n, f):
        final_score = np.mean(scores[ss:nn].astype(float))
        
        if final_score > score_threshold:
            chr_probes_df.loc[ff, 'methylation_calls'] += 1
        else:
            chr_probes_df.loc[ff, 'unmethylation_calls'] += 1

    chr_probes_df.loc[:, 'total_calls'] = chr_probes_df.loc[:, 'methylation_calls'] + chr_probes_df.loc[:, 'unmethylation_calls']
    return chr_probes_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-dir", type=str, help='Path where the bam files output from Guppy are', required = True)
    parser.add_argument("--output-dir", type=str, help='Dir where the results are saved', required = True)
    parser.add_argument("--margin", type=int, default = 0, help='Margin, on either side of a probe location, to consider methylation calls. If more than one methylation call falls within the margin, the average is taken.')
    parser.add_argument("--threshold", type=float, default = 0.6, help='Average scores higher than this threshold will be considered methylated')
    parser.add_argument("--processes", type=int, default = 4, help = 'Number of parallel processes to run')
    parser.add_argument("--probes-file", type=str, default = '/hpc/compgen/projects/asmethylation/gw_profiles/analysis/cvermeulen/fishname_wrapper/probelocs_chm13.bed', help='Path to the probes locations')
    
    args = parser.parse_args()

    resume_previous_file = False
    probe_calls_summary_file = os.path.join(
        args.output_dir, 
        'probe_calls_summary_file.csv'
    )
    processed_bam_files = list()
    if os.path.isfile(probe_calls_summary_file):
        with open(probe_calls_summary_file, 'r') as ofile:
            for line_i, line in enumerate(ofile):
                if line_i == 0:
                    bam_dir = line[1:].strip('\n')
                    if bam_dir != args.bam_dir:
                        break
                    resume_previous_file = True

                if line.startswith('#'):
                    processed_bam_files.append(line[1:].strip('\n'))
                else:
                    break

    if resume_previous_file:
        existing_calls = pd.read_csv(
            probe_calls_summary_file,
            header = 0, 
            index_col = None, 
            sep = ' ',
            comment = '#',
        )
    else:
        existing_calls = pd.read_csv(
            args.probes_file, 
            header = 0, 
            index_col = None, 
            sep = ' ',
        )

        existing_calls['methylation_calls'] = 0
        existing_calls['unmethylation_calls'] = 0
        existing_calls['total_calls'] = 0

    chroms = np.unique(existing_calls['chr'])

    existing_call_chroms = dict()
    for c in chroms:
        existing_call_chroms[c] = existing_calls[existing_calls['chr'] == c]

    with multiprocessing.Pool(processes = args.processes) as pool:

        for bam_file in os.listdir(args.bam_dir):

            if not bam_file.endswith('.bam'):
                continue

            bam_file = os.path.join(args.bam_dir, bam_file)
            if bam_file in processed_bam_files:
                continue

            print('Processing bam file: {}'.format(bam_file))

            processed_bam_files.append(bam_file)

            results = list()
            for c in chroms:

                results.append(pool.apply_async(
                    get_methylation_status,
                    (existing_call_chroms[c], bam_file, args.margin, args.threshold)
                ))

            for res in results:
                df = res.get()
                chrom_num = np.unique(df['chr']).item()
                existing_call_chroms[chrom_num] = df
                print('Processing chr: {} - Calls: {}'.format(chrom_num, df['total_calls'].sum()))

    final_result = pd.concat(list(existing_call_chroms.values()))
    with open(probe_calls_summary_file, 'w') as ofile:
        ofile.write('#'+args.bam_dir+'\n')
        for pbf in processed_bam_files:
            ofile.write('#'+pbf+'\n')

    final_result.to_csv(
        probe_calls_summary_file,
        mode = 'a',
        header = True,
        index = False,
        sep = ' ',
    )








