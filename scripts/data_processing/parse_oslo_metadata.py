import pandas as pd
import argparse 
import os
#https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6402nnn/GSM6402677/suppl/GSM6402677_2197T.RData.gz


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-file", type=str, help='File with the metadata')
    parser.add_argument("--output-file", type=str, help='Output file csv')
    args = parser.parse_args()

    soft_file = args.metadata_file

    output_dict = {
        "geo_code": list(),
        "sample_code": list(),
        "reference_diagnosis": list(),
        "methylation_call": list(),
        "methylation_family_call": list(),
    }

    c = 0
    with open(soft_file, 'r') as handle:

        for line_num, line in enumerate(handle):
            line = line.strip('\n')

            if line.startswith("!Sample_title"):
                output_dict['sample_code'].append(line.split(' = ')[-1])

            if line.startswith("!Sample_geo_accession"):
                output_dict['geo_code'].append(line.split(' = ')[-1])
            
            if line.startswith("!Sample_characteristics_ch1"):
                if c == 0:
                    pass
                elif c == 1:
                    output_dict['reference_diagnosis'].append(line.split(' = ')[-1].split(': ')[-1])
                elif c == 2:
                    output_dict['methylation_call'].append(line.split(' = ')[-1].split(': ')[-1])
                elif c == 3:
                    output_dict['methylation_family_call'].append(line.split(' = ')[-1].split(': ')[-1])
                    c = -1
                c += 1
                    

    df = pd.DataFrame(output_dict)

    df.to_csv(args.output_file, header = True, index = False)

    import urllib.request
    import gzip
    import shutil
    
    url_stem = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6402nnn/'
    url_stem2 = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/'
    output_dir = "/".join(args.output_file.split('/')[:-1])

    try:
        os.mkdir(os.path.join(output_dir, 'bed'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(output_dir, 'RData'))
    except FileExistsError:
        pass


    with open(os.path.join(output_dir, 'download_links_data_bed.txt'), 'w') as handle:
        for i, row in df.iterrows():

            url = url_stem + row.geo_code + '/suppl/' + row.geo_code + '_' + row.sample_code + '.bed.gz'
            gzip_file = os.path.join(output_dir, 'bed', row.geo_code + '_' + row.sample_code + '.bed.gz')
            f_file = os.path.join(output_dir, 'bed', row.geo_code + '_' + row.sample_code + '.bed')

            if os.path.isfile(gzip_file) or os.path.isfile(f_file):
                continue

            try:
                urllib.request.urlretrieve(url, filename=gzip_file)
            except:
                try:
                    url = url_stem2 + row.geo_code + '/suppl/' + row.geo_code + '_' + row.sample_code + '.bed.gz'
                    urllib.request.urlretrieve(url, filename=gzip_file)
                except:
                    print('Failed to download: ' + gzip_file)
                    continue

            with gzip.open(gzip_file, 'rb') as f_in:
                with open(f_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gzip_file)
    
            url  = url_stem + row.geo_code + '/suppl/' + row.geo_code + '_' + row.sample_code + '.RData.gz'
            gzip_file = os.path.join(output_dir, 'RData', row.geo_code + '_' + row.sample_code + '.RData.gz')
            f_file = os.path.join(output_dir, 'RData', row.geo_code + '_' + row.sample_code + '.RData')

            if os.path.isfile(gzip_file) or os.path.isfile(f_file):
                continue

            try:
                urllib.request.urlretrieve(url, filename=gzip_file)
            except:
                try:
                    url = url_stem2 + row.geo_code + '/suppl/' + row.geo_code + '_' + row.sample_code + '.RData.gz'
                    urllib.request.urlretrieve(url, filename=gzip_file)
                except:
                    print('Failed to download: ' + gzip_file)
                    continue

            with gzip.open(gzip_file, 'rb') as f_in:
                with open(f_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gzip_file)
 