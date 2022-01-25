from collections import deque
import pandas as pd
import pysam

def convert_coords_tuple(coords):
    coords_list = []
    for i in range(len(coords.index)):
        coords_list.append((coords.loc[i]["chrom"], coords.loc[i]["chromStart"], coords.loc[i]["chromEnd"]))
    print(coords_list)
    return coords_list

def count_variant_repeats(seq, targets, targets_to_array, direction, k=6):
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    kmers_c = 0
    blocks = []
    a = [0] * (len(targets_to_array) + 1)  # last column is for 'others'
    for item in kmers:
        if item in targets:
            a[targets[item]] += 1
            kmers_c += 1
        else:
            a[-1] += 1
    i = 0
    while i < len(kmers):
        current = i
        mm = 0
        while current < len(kmers):
            if kmers[current] in targets:
                current += 1
            else:
                break
        if i != current:
            blocks.append(current - i + k - 1)
            i = current + 1
        i += 1

    kmers_7 = [seq[i:i+7] for i in range(len(seq)-7)]
    if direction == "forward":
        counts_7 = len([1 for i in kmers_7 if i == "CCCTTAA"])
        kmers_c += counts_7
        a[targets["CCCTTAA"]] = counts_7

    if direction == "reverse":
        counts_7 = len([1 for i in kmers_7 if i == "TTAAGGG"])
        kmers_c += counts_7
        a[targets["TTAAGGG"]] = counts_7

    if direction == "both":
        counts_7 = len([1 for i in kmers_7 if i == "CCCTTAA"])
        kmers_c += counts_7
        a[targets["CCCTTAA"]] = counts_7
        counts_7 = len([1 for i in kmers_7 if i == "TTAAGGG"])
        kmers_c += counts_7
        a[targets["TTAAGGG"]] = counts_7

    return kmers_c, a, blocks

def tel_tokens(telomere_f, key_index):
    d = deque(telomere_f)
    f_rotations = {"".join(d): key_index}
    for i in range(len(telomere_f)-1):
        d.rotate()
        f_rotations["".join(d)] = key_index
    return f_rotations

def make_rotation_keys(tta):
    variant_rotations = {}
    for key, key_index in tta.items():
        variant_rotations.update(tel_tokens(key, key_index))
    return variant_rotations





targets_to_array_f = {"CCCTAA": 0, "CCCTGA": 1, "CCCGAA": 2, "CCCTAC": 3, "CCCTCA": 4, "CCCCAA": 5, "CCCTTA": 6,
                   "CCCTAT": 7, "CCCTAG": 8, "CCCAAA": 9}
targets_to_array_f.update({"CCCTTAA": 10, "CCCACT": 11, "CCCCAT": 12, "CCCGCA": 13, "CCCGCT": 14, "CCCTCT": 15})

targets_to_array_r = {"TTAGGG": 0, "TCAGGG": 1, "TTCGGG": 2, "GTAGGG": 3, "TGAGGG": 4, "TTGGGG": 5, "TAAGGG": 6,
                   "ATAGGG": 7, "CTAGGG": 8, "TTTGGG": 9}
targets_to_array_r.update({"TTAAGGG": 10, "AGTGGG": 11, "ATGGGG": 12, "TGCGGG": 13, "AGCGGG": 14, "AGAGGG": 15})

targets_to_array_both = {}
targets_to_array_both.update(targets_to_array_f)
targets_to_array_both.update(targets_to_array_r)

targets_f = make_rotation_keys(targets_to_array_f)
targets_r = make_rotation_keys(targets_to_array_r)
targets_both = make_rotation_keys(targets_to_array_both)

targets_dict = {"forward": targets_f, "reverse": targets_r}
targets_to_array_dict = {"forward": targets_to_array_f, "reverse": targets_to_array_r}






coords = pd.read_csv("../../telomere_regions/hg38.tsv", sep="\t")
co = convert_coords_tuple(coords)

ref = pysam.FastaFile("../../../reference_genomes/hg38.fa")

regs = {}


coords_chromo_list = list(set(sorted(coords.chrom)))
for chromo in coords_chromo_list:
    chrom_reg = coords[coords["chrom"] == chromo]
    for i in range(len(chrom_reg["chrom"])):
        low = int(chrom_reg["chromStart"].iloc[i])
        up = int(chrom_reg["chromEnd"].iloc[i])
        direction = chrom_reg["direction"].iloc[i]
        regs[chromo+"_"+str(low)+"_"+str(up)] = {}
        regs[chromo+"_"+str(low)+"_"+str(up)]["d"] = direction
        region = ref.fetch(chromo, low, up).upper()
        regs[chromo+"_"+str(low)+"_"+str(up)]["r"] = region
        regs[chromo+"_"+str(low)+"_"+str(up)]["l"] = len(region)
        kc, a, blocks = count_variant_repeats(region, targets_dict[direction], targets_to_array_dict[direction], direction)
        regs[chromo+"_"+str(low)+"_"+str(up)]["c"] = kc
        regs[chromo+"_"+str(low)+"_"+str(up)]["p"] = kc/len(region)
        regs[chromo+"_"+str(low)+"_"+str(up)]["a"] = a
        regs[chromo+"_"+str(low)+"_"+str(up)]["b"] = blocks




