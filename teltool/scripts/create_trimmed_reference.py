import pysam
import pandas as pd

coords = pd.read_csv("/home/alex/Desktop/uni/PhD/TL_prediction/teltool/telomere_regions/fast/hg38.tsv", sep="\t")
ref_path = "/home/alex/Desktop/uni/PhD/TL_prediction/reference_genomes/hg38.fa"

def convert_coords_tuple(coords):
    coords_list = []
    for i in range(len(coords.index)):
        coords_list.append((coords.loc[i]["chrom"], coords.loc[i]["chromStart"], coords.loc[i]["chromEnd"]))
    print(coords_list)
    return coords_list

coords_list = convert_coords_tuple(coords)

ref = pysam.FastaFile(ref_path)

for c, i in enumerate(coords_list):
    print(f">{c+1}")
    print(ref.fetch(i[0], i[1]-150, i[2]+150).upper())
    
