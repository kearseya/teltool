import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy import stats
import re
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
import math
import os
import array
from collections import deque
import itertools
import skbio
from collections import defaultdict

import array
import random
import time
import pysam
import networkx as nx


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


print(targets_both)


def count_variant_repeats(seq, targets, k=6):
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    kmers_c = 0
    blocks = []

    for item in kmers:
        if item in targets:
            yield targets[item]
            # a[targets[item]] += 1
            kmers_c += 1
        else:
            yield -1
            # a[-1] += 1
    # i = 0
    # while i < len(kmers):
    #     current = i
    #     mm = 0
    #     while current < len(kmers):
    #         if kmers[current] in targets:
    #             current += 1
    #         else:
    #             break
    #     if i != current:
    #         blocks.append(current - i + k - 1)
    #         i = current + 1
    #     i += 1
    # print(blocks)

# print(list(count_variant_repeats("TTAGGGTTAGGGTTAGGG", targets_both)))


def sliding_window_minimum(w, m, s):
    '''
    A iterator which takes the size of the window, `k`, and an iterable,
    `li`. Then returns an iterator such that the ith element yielded is equal
    to min(list(li)[max(i - k + 1, 0):i+1]).
    Each yield takes amortized O(1) time, and overall the generator takes O(k)
    space.
    '''

    found = None
    window2 = deque()
    i = 0
    end = len(s)
    last_idx = end - 1
    while i < end:
        hx2 = hash(s[i:i+m])
        if i == 0 or i == last_idx:
            yield hx2
            # found.append((hx2, i))
        while len(window2) != 0 and window2[-1][0] >= hx2:
            window2.pop()
        window2.append((hx2, i))
        while window2[0][1] <= i - w:
            window2.popleft()

        minimizer_i = window2[0]
        if minimizer_i != found:
            found = minimizer_i
            yield minimizer_i[0]
        i += 1


print(list(sliding_window_minimum(10, 4, "TTAGGGTTAGGGTTAGGG")))


ref = pysam.FastaFile("/Users/kezcleal/Documents/Data/db/hg38/hg38.fa")

all_minimizers = set([])
G = nx.DiGraph()
G2 = nx.DiGraph()

def rev_comp(s):
    a = {"A": "T", "C": "G", "T": "A", "G": "C", "N": "N"}
    return "".join(a[i] for i in s)[::-1]


with open("hg38_for_kmer_collection.tsv", "r") as reg:
    next(reg)
    for l in reg:

        # Add to minimizer embedding
        r = l.strip().split("\t")
        s = ref.fetch(r[0], int(r[1]), int(r[2])).upper()
        mm = list(sliding_window_minimum(12, 6, s))
        for m in mm:
            all_minimizers.add(m)
        G.add_weighted_edges_from(zip(mm[:-1], mm[1:], [1 for _ in range(len(mm)-1)]))

        # Reverse complement
        s2 = rev_comp(s)
        mm = list(sliding_window_minimum(12, 6, s2))
        for m in mm:
            all_minimizers.add(m)
        G.add_weighted_edges_from(zip(mm[:-1], mm[1:], [1 for _ in range(len(mm) - 1)]))

        # Add to telmer embedding
        t = list(count_variant_repeats(s, targets_both))
        G2.add_weighted_edges_from(zip(t[:-1], t[1:], [1 for _ in range(len(mm) - 1)]))
        t = list(count_variant_repeats(s2, targets_both))
        G2.add_weighted_edges_from(zip(t[:-1], t[1:], [1 for _ in range(len(mm) - 1)]))

print("N minimizers", len(all_minimizers))
print("N edges minimizer embedding", len(G.edges()), "Nodes", len(G.nodes()))
print("N edges telmer embedding", len(G2.edges()), "Nodes", len(G2.nodes()))

# Try for HG005
G3 = nx.DiGraph()
G4 = nx.DiGraph()
bam = pysam.AlignmentFile("/Users/kezcleal/Documents/Data/fusion_finder_development/ChineseTrio/alignments/HG005.pacbio.bam", "rb")


with open("hg38_for_kmer_collection.tsv", "r") as reg:
    next(reg)
    for l in reg:

        # Add to minimizer embedding
        r = l.strip().split("\t")
        for s in bam.fetch(r[0], int(r[1]), int(r[2])):
            s = s.seq
            if not s:
                continue
            mm = list(sliding_window_minimum(12, 6, s))
            for m in mm:
                all_minimizers.add(m)
            G3.add_weighted_edges_from(zip(mm[:-1], mm[1:], [1 for _ in range(len(mm)-1)]))

            # Reverse complement
            s2 = rev_comp(s)
            mm = list(sliding_window_minimum(12, 6, s2))
            for m in mm:
                all_minimizers.add(m)
            G3.add_weighted_edges_from(zip(mm[:-1], mm[1:], [1 for _ in range(len(mm) - 1)]))

            # Add to telmer embedding
            t = list(count_variant_repeats(s, targets_both))
            G4.add_weighted_edges_from(zip(t[:-1], t[1:], [1 for _ in range(len(mm) - 1)]))
            t = list(count_variant_repeats(s2, targets_both))
            G4.add_weighted_edges_from(zip(t[:-1], t[1:], [1 for _ in range(len(mm) - 1)]))

print()
print("N edges minimizer embedding HG005", len(G3.edges()), "Nodes", len(G3.nodes()))
print("N edges telmer embedding HG005", len(G4.edges()), "Nodes", len(G4.nodes()))

print(nx.to_numpy_matrix(G))