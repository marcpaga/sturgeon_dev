import os
import argparse
from copy import deepcopy
import multiprocessing

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

import sturgeon

def sim_run(output_dir, cpg_sites, cumsum, tmp_read_ids, mega_df, probes_df, sim_num):

    mega_calls = list()
    for time, n_cpg in enumerate(cpg_sites):

        output_file = os.path.join(output_dir, 'sim_{}_time_{}.bed'.format(sim_num, time))
        if os.path.isfile(output_file):
            continue
        seq_read_ids = tmp_read_ids[:np.searchsorted(cumsum, n_cpg)]

        for rid in seq_read_ids:
            st = mega_df['read_id'].searchsorted(rid, 'left')
            nd = mega_df['read_id'].searchsorted(rid, 'right')

            mega_calls.append(deepcopy(mega_df[st:nd]))

        mega_calls_df = pd.concat(mega_calls)
        mega_calls_df = mega_calls_df.rename(columns = {
            'chrm':'chr', 
            'pos': 'reference_pos',
        })
        mega_calls_df['score'] = np.exp(mega_calls_df['mod_log_prob'])
        mega_calls_df = mega_calls_df.drop(columns = [
            'mod_log_prob', 
            'can_log_prob', 
            'mod_base',
        ])

        probes_methyl_df = deepcopy(probes_df)

        chromosomes = np.unique(probes_df['chr'])

        probes_methyl_df['methylation_calls'] = 0
        probes_methyl_df['unmethylation_calls'] = 0
        probes_methyl_df['total_calls'] = 0

        calls_per_probe = list()
        for chrom in chromosomes:

            calls_per_probe_chr =  sturgeon.callmapping.map_methyl_calls_to_probes_chr(
                probes_df =  probes_methyl_df[probes_methyl_df['chr'] == chrom.item()],
                methyl_calls_per_read = mega_calls_df[mega_calls_df['chr'] == 'chr'+str(chrom.item())],
                margin = 25, 
                neg_threshold = 0.3,
                pos_threshold = 0.7,
            )
            calls_per_probe.append(calls_per_probe_chr)

        calls_per_probe = pd.concat(calls_per_probe)

        bed_df = {
            "chrom": list(), 
            "chromStart": list(), 
            "chromEnd": list(), 
            "methylation_call": list(), 
            "probe_id": list(),
        }

        calls_per_probe = calls_per_probe[calls_per_probe['total_calls'] > 0]

        for _, row in calls_per_probe.iterrows():
            if row.methylation_calls == row.unmethylation_calls:
                continue

            if row.methylation_calls > row.unmethylation_calls:
                m = 1
            else:
                m = 0

            bed_df['chrom'].append(row.chr)
            bed_df['chromStart'].append(row.start)
            bed_df['chromEnd'].append(row.end)
            bed_df['methylation_call'].append(m)
            bed_df['probe_id'].append(row.ID_REF)

        bed_df = pd.DataFrame(bed_df)
        
        bed_df.to_csv(
            output_file,
            header = True,
            index = False,
            sep = '\t'
        )
        print(output_file, flush = True)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--megalodon-file", type=str, help='Megalodon file', required = True)
    parser.add_argument("--output-dir", type=str, help='Path where sims are saved', required = True)
    parser.add_argument("--num-simulations", type=int, default = 1)
    parser.add_argument("--cpg-sites", type=int, nargs = '+', default = [
        51924,
        104073,
        124078,
        149111,
        173504,
        194399,
        207456,
        217193,
        232101,
        241278,
        247600,
        258197,]
    )
    parser.add_argument("--probes-file", type=str, default = '/hpc/compgen/users/mpages/sturgeon/sturgeon/include/static/probes.bed')
    parser.add_argument("--pseudotime", action='store_true', help='dont simulate, do the first reads in chronological order')
    parser.add_argument("--processes", type=int, default = 1)

    args = parser.parse_args()

    print('Reading probes file', flush = True)
    probes_df = pd.read_csv(
        args.probes_file, 
        header = 0, 
        index_col = None, 
        sep = ' ',
    )

    print('Reading megalodon file', flush = True)
    mega_df = pd.read_csv(
        args.megalodon_file, 
        delim_whitespace = True,
        header = 0, 
        index_col = None,
        nrows = 30000000
    )

    print('Processing methylation calls', flush = True)
    if args.pseudotime:
        
        read_ids = mega_df['read_id'].unique().astype(str)
        counts = mega_df['read_id'].value_counts()
        cpgsites_per_read = np.array(counts.loc[read_ids])

        cpg_sites = np.cumsum(args.cpg_sites)
        cumsum = np.cumsum(cpgsites_per_read)

        mega_df = mega_df.sort_values(['read_id'])
        mega_df.reset_index(drop = True, inplace = True)
        
        print('Simulating', flush = True)
        sim_run(args.output_dir, cpg_sites, cumsum, read_ids, mega_df, probes_df, -1)

    else:
        mega_df = mega_df.sort_values(['read_id'])
        mega_df.reset_index(drop = True, inplace = True)

        read_ids, cpgsites_per_read = np.unique(mega_df['read_id'], return_counts = True)

        read_order = np.arange(len(read_ids))
        cpg_sites = np.cumsum(args.cpg_sites)

        results = list()
        print('Simulating', flush = True)
        with multiprocessing.Pool(processes = args.processes) as pool:
            for sim_num in range(args.num_simulations):

                np.random.seed(sim_num)
                np.random.shuffle(read_order)

                tmp_read_ids = deepcopy(read_ids[read_order])
                tmp_cpgsites_per_read = cpgsites_per_read[read_order]
                cumsum = np.cumsum(tmp_cpgsites_per_read)

                res = pool.apply_async(
                    sim_run,
                    (args.output_dir, cpg_sites, cumsum, tmp_read_ids, mega_df, probes_df, sim_num)
                )

                results.append(res)
            for res in results:    
                res.get()
                

        



