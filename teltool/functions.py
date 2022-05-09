#!/bin/python3
import sys, os
import warnings
import click
import pysam
import subprocess
#import argparse
import xml.etree.ElementTree as et
import json
#import logging
#import pickle #maybe
#from joblib import dump, load
import joblib
import pandas as pd
import numpy as np
from numpy.random import choice
import math
from scipy import stats
from statistics import mode
import itertools
from itertools import chain
import re
#from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
# import joypy
import seaborn as sns
# import glob
# from contextlib import contextmanager
# from mlxtend.regressor import LinearRegression
# from nltk import DecisionTreeClassifier, sys
# #from collections import defaultdict
#
#from sknn.mlp import Classifier, Convolution, Layer
import sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection

import collections
from collections import deque
from collections import defaultdict
import networkx as nx
#from ncls import NCLS
#import loose_functions
# from sklearn import neighbors
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
# Regressors
from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
# Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
#import thread
import threading
import multiprocessing
import logging
import concurrent.futures

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV

from teltool.cteltool import *
from time_test import *

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
        # mm = 0
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



def add(a, arr, low, mapq, chromo):
    if a.flag & 1284 or a.cigartuples is None: # or a.mapq == 0:
        return arr
    if int(a.reference_start)-low+100 > len(arr)-1:
        return arr
    if mapq == False:
        index_start = a.reference_start-low+100
    if mapq == True:
        index_start = 100
    index_bin = index_start
    for opp, length in a.cigartuples:
        if index_bin > len(arr) - 1:
            break
        if opp == 2:
            index_start += length
            index_bin = index_start
            #arr[index_bin] -= 1
        elif opp == 0 or opp == 7 or opp == 8: # cigar is mapped
            arr[index_bin] += 1
            index_start += length
            index_bin = index_start
            if index_bin < len(arr):
                arr[index_bin] -= 1
    return arr

# def add_fr(a, arr, low, mapq, chromo):
#     if a.flag & 1284: #or a.cigartuples is None: # or a.mapq == 0:
#         return arr
#     #if a.flag & 4:
#         #print(f"unmapped {a.seq}")
#     if int(a.reference_start)-low+100 > len(arr)-1:
#         return arr
#     if mapq == False:
#         index_start = a.reference_start-low+100
#     if mapq == True:
#         index_start = 100
#     arr[index_start] += 1
#     if index_start+a.query_alignment_length < len(arr):
#         arr[index_start+a.query_alignment_length] -= 1
#     return arr

def strandedness(chromo, read):
    if chromo == "r":
        if read.is_reverse:
            return "f"
        else:
            return"r"
    if chromo == "f":
        if read.is_reverse:
            return "r"
        else:
            return "f"



def good_quality_clip(r, clip_length):
    # Use sliding window to check that soft-clip has at least n good bases over qual 20
    seq = r.seq
    # print("LEN", len(seq), len(r.seq))
    #print(seq)
    quals = r.query_qualities
    window_length = 10
    if clip_length < window_length:
        window_length = clip_length
    poly = window_length - 1
    total_good = window_length - 1
    ct = r.cigartuples
    ct = r.cigartuples
    if ct[0][0] != 4 and ct[-1][0] != 4:
        return 1
    if r.flag & 2304:  # supplementary, usually hard-clipped, assume good clipping
        if ct[0][0] == 5 or ct[-1][0] == 5:
            return 1
    if ct[0][0] == 4:
        length = r.cigartuples[0][1]
        if length >= window_length and length >= clip_length:
            for i in range(length - window_length, -1, -1):
                # average of current window by counting leftwards
                basecounts = {"A": 0, "C": 0, "G": 0, "T": 0, "N":0}
                #seq = r.query_alignment_sequence
                homopolymer = False
                w_sum = 0
                for j in range(i, i + window_length):
                    letter = seq[j]
                    w_sum += quals[j]
                    basecounts[letter] += 1
                    if basecounts[letter] == poly:
                        homopolymer = True
                        #break
                avg_qual = w_sum / window_length
                if avg_qual > 10:
                    if not homopolymer:
                        # Make sure stretch is not homopolymer
                        total_good += 1
                else:
                    break
            if total_good >= clip_length:
                return 1
    total_good = window_length - 1
    if ct[-1][0] == 4:
        length = r.cigartuples[-1][1]
        if length >= window_length and length >= clip_length:
            for i in range(len(r.query_qualities) - length, len(r.query_qualities) - window_length):
                # average of current window by counting rightwards
                basecounts = {"A": 0, "C": 0, "G": 0, "T": 0, "N":0}
                homopolymer = False
                w_sum = 0
                for j in range(i, i + window_length):
                    letter = seq[j]
                    w_sum += quals[j]
                    basecounts[letter] += 1
                    if basecounts[letter] == poly:
                        homopolymer = True
                        break
                avg_qual = w_sum / float(window_length)
                if avg_qual > 10:
                    if not homopolymer:
                        total_good += 1
                else:
                    break
            if total_good >= clip_length:
                return 1
    return 0



def soft_clip_qual_corr(reads):
    """Function to compute the correlation of quality values between the soft-clipped portions of reads. values of
    1 imply no correlation whereas closer to 0 implies that quality values are correlated"""
    qq = defaultdict(list)
    for r in reads:
        quals = r.query_qualities
        if r.cigartuples is None or quals is None:
            continue
        if r.cigartuples[0][0] == 4:
            idx = 0
            for x in range(r.pos - r.cigartuples[0][1], r.pos):
                qq[x].append(quals[idx])
                idx += 1
        if r.cigartuples[-1][0] == 4:
            end_pos = r.query_alignment_end + r.cigartuples[-1][1]
            for idx in range(-1, - r.cigartuples[-1][1], -1):
                qq[end_pos].append(quals[idx])
                end_pos -= 1
    if len(qq) == 0:
        return -1
    all_z = 0
    all_v = []
    seen = False
    sum_all_v = 0
    for _, second in qq.items():
        if len(second) > 1:
            sum_second = 0
            sum_second += sum(second)
            mean_second = sum_second / len(second)
            all_z += sum([abs(j - mean_second) for j in second])
            sum_all_v += sum_second
            all_v += second # concat
            seen = True
    if not seen:
        return -1
    mean_all_v = sum_all_v / len(all_v)
    z = sum([abs(j - mean_second) for j in all_v])
    if z == 0:
        return -1
    return all_z / z



def collect_coverage(i, step="other"):
    if step == "trim":
        print("Collecting coverage before trimming (saved to cwd/coverages.csv)")
    else:
        print("Measuring coverage")
    coverages = {}
    def index_stats(fn, rl):
        f = pysam.AlignmentFile(fn)
        s = f.get_index_statistics()
        total = 0
        ref_len = 0
        for i in s:
            total += i.mapped
            ref_len += f.get_reference_length(i.contig)
        cov = (total*rl/ref_len)*0.98
        return cov

    if os.path.isdir(i) == False:
        if i.endswith(".bam"):
            files = [i]
    if os.path.isdir(i) == True:
        files = os.listdir(i)
        files = [f for f in files if f.endswith(".bam")]

    for f in files:
        lengths = set()
        af = pysam.AlignmentFile(f)
        seg = af.head(100000)
        for read in seg:
            lengths.add(read.query_alignment_length)
        cov = index_stats(f, max(lengths))
        fnwe = os.path.basename(f)
        fn = os.path.splitext(fnwe)[0]
        coverages[fn] = cov

    return coverages

def convert_coords_tuple(coords):
    coords_list = []
    for i in range(len(coords.index)):
        coords_list.append((coords.loc[i]["chrom"], coords.loc[i]["chromStart"], coords.loc[i]["chromEnd"]))
    print(coords_list)
    return coords_list

def test_py_code(coords):
    # test_bam("/home/alex/Desktop/uni/PhD/TL_prediction/tmp/DB144_tel.bam")
    # coords_list = convert_coords_tuple(coords)
    # collect_wanted_kmers("/home/alex/Desktop/uni/PhD/TL_prediction/reference_genomes/hg38.fa", coords_list, 32)
    scan_files([f for f in os.listdir("/home/alex/Desktop/uni/PhD/TL_prediction/tmp") if f.endswith(".bam")],
    "/home/alex/Desktop/uni/PhD/TL_prediction/fastq_scans",
    32,
    "/home/alex/Desktop/uni/PhD/TL_prediction/teltool/telmer_set/telmers_rolling_32.set")

def find_telmers(in_dir, ctx, file_type, chr_pres, ref_build):
    if file_type == "bam":
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(".bam")]

    if file_type == "fastq":
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(".fq") or f.lower().endswith(".fastq")]
    if file_type == "fasta":
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(".fa") or f.lower().endswith(".fasta")]

    if file_type in ["fastq", "fasta"]:
        for f in files:
            with pysam.FastxFile(os.path.join(in_dir, f)) as fh:
                for entry in fh:
                    print(entry.sequence)


def read_tl_bam(in_dir, ctx, n_threads, bam_files, coords, avg_coverages, chr_pres):#, std_coverages):
    total_files = len(bam_files)

    data = []

    chromo_reg_len = {}
    telmer_dist = {}
    coverage_dict = {}
    coords_chromo_list = list(set(sorted(coords.chrom)))

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

    for idx, f in enumerate(bam_files):
        print("ON: ", idx+1, "/", total_files, "    (",f,")                 ", end="\r")
        data.append({})
        sample = os.path.basename(f).split("_")[0]
        data[idx]["sample"] = sample
        telmer_dist[sample] = {}
        bamfile = pysam.AlignmentFile(os.path.join(in_dir, f), "rb", threads=n_threads)
        coverage_dict[idx] = {}
        mapped_unmapped = bamfile.get_index_statistics()
        for mapq in ["mapq0", "mapq1", "short"]: #, "unmapped"]:
            data[idx][f"{mapq}.length"] = 0
            data[idx][f"{mapq}.coverage"] = 0
            data[idx][f"{mapq}.gc"] = 0
            data[idx][f"{mapq}.num_reads"] = 0
            data[idx][f"{mapq}.kc_len"] = 0
            data[idx][f"{mapq}.fragments"] = 0
            telmer_dist[sample][mapq] = []

            coverage_array = np.array([], dtype="int64")
            coverage_array = np.resize(coverage_array, int(310))
            coverage_array.fill(0)
            coverage_dict[idx][mapq] = coverage_array

        short = {"length": 0, "gc": 0, "mapq": 0, "kc_len": 0, "fragments": 0, "num_reads": 0, "coverage": []}
        data[idx]["short.mapq"] = 0
        data[idx]["short.len_reg"] = 0
        data[idx]["short.kc_len_reg"] = 0
        num_short_regions = 0
        seq_lens = set()
        # seq_lens.add(100)
        using_fr = all(chromo in coords_chromo_list for chromo in ["f", "r"])
        if using_fr:
            num_reads_fr = {"f": 0, "r": 0}
            coverage_array_fr = {"f": np.array([], dtype="int64"), "r": np.array([], dtype="int64")}
            coverage_array_strand = {"f": np.array([], dtype="int64"), "r": np.array([], dtype="int64")}
            processed_f = False
            processed_r = False
            # unmapped = {}
            # paired = {}
            # unpaired = {}
            for strand in ["f", "r"]:
                strand_dict = {"f": {"f": 0, "r": 0}, "r": {"f": 0, "r": 0}}
                data[idx][f"{strand}_0.length"] = 0
                data[idx][f"{strand}_0.coverage"] = 1
                data[idx][f"{strand}_0.gc"] = 0
                data[idx][f"{strand}_0.num_reads"] = 0
                data[idx][f"{strand}_0.mapq"] = 0
                data[idx][f"{strand}_0.kc_len"] = 0
                data[idx][f"{strand}_0.fragments"] = 0
                telmer_dist[sample][f"{strand}_0"] = []
                # unmapped[strand] = 0
                # paired[strand] = 0
                # unpaired[strand] = 0
                low = int(coords[coords["chrom"] == strand]["chromStart"].iloc[0])
                up = int(coords[coords["chrom"] == strand]["chromEnd"].iloc[0])
                region_len = up-low
                coverage_array_fr[strand] = np.resize(coverage_array_fr[strand], int(region_len)+(2*100))
                coverage_array_fr[strand].fill(0)
                coverage_array_strand[strand] = np.resize(coverage_array_strand[strand], int(region_len)+(2*100))
                coverage_array_strand[strand].fill(0)
                #coverage_array = np.zeros((coverage_array, int(region_len)), dtype="int64")
        for chromo in coords_chromo_list:
            chrom_reg = coords[coords["chrom"] == chromo]
            chromo_reg_len[chromo] = 0
            for i in range(len(chrom_reg["chrom"])):
                if chrom_reg["diff"].iloc[i] < 100:
                    continue
                if 100 <= chrom_reg["diff"].iloc[i] < 700:
                    low = int(chrom_reg["chromStart"].iloc[i])
                    up = int(chrom_reg["chromEnd"].iloc[i])
                    reg_len = up-low
                    region_len = reg_len+(2*100)
                    chromo_reg_len[chromo] += reg_len
                    coverage_array = np.array([], dtype="int64")
                    coverage_array = np.resize(coverage_array, int(region_len)+(2*100))
                    coverage_array.fill(0)

                    direction = chrom_reg["direction"].iloc[i]
                    num_reads = 0

                    num_short_regions += 1
                    # comp = [0]*len(targets_f)
                    bad_qual = 0
                    ## chr.replace ??
                    if chr_pres == True:
                        fchromo = chromo
                    if chr_pres == False:
                        fchromo = chromo.split("chr")[1]
                        #.rplace("chr", ")
                    for read in bamfile.fetch(str(fchromo), low, up): #, multiple_iterators=True):
                        if read.flag & 3840 > 0:
                            continue
                        else:
                            if read.cigartuples != None:
                                good_qual = good_quality_clip(read, 15)
                                if good_qual == 0:
                                    bad_qual += 1
                                    continue
                            seq_len = read.query_alignment_length
                            seq = read.query_alignment_sequence
                            kc, a, blocks = count_variant_repeats(seq, targets_dict[direction], targets_to_array_dict[direction], direction)
                            # comp = [x+y for x,y in zip(comp, a)]
                            if len(blocks) == 0:
                                blocks.append(0)
                            if kc >= 1: #max(blocks) >= 30 or len(blocks) >= 5:
                                num_reads += 1
                                short["num_reads"] += 1
                                # make integer check length
                                seq_lens.add(seq_len)
                                short["length"] += seq_len
                                coverage_array = add(read, coverage_array, low, False, "short")
                                short["gc"] += (seq.count("G")+seq.count("C"))/seq_len
                                short["mapq"] += read.mapq
                                short["kc_len"] += sum(blocks)
                                short["fragments"] += len(blocks)
                            else:
                                continue
                    if num_reads != 0:
                        ## length is query_alignment_length so does not include soft clip, not real read length
                        # data[idx]["short.length"] += (short["length"]/max(seq_lens))*100
                        # data[idx]["short.gc"] += short["gc"]/num_reads
                        # data[idx]["short.num_reads"] += num_reads
                        # data[idx]["short.mapq"] += short["mapq"]/num_reads

                        ## coverage
                        current_cov = 0
                        for k in range(len(coverage_array)-(2*100)):
                            v = coverage_array[k+100]
                            current_cov += v
                            coverage_array[k+100] = current_cov
                            if current_cov < 0:
                                coverage_array[k+100] = 0  # sanity check
                        short["coverage"].append(np.mean(coverage_array[100:-100]))
                        # if coverage > 0:
                        #     data[idx]["short.coverage"] = coverage
                        # else:
                        #     data[idx]["short.coverage"] += 1

                        # data[idx]["short.fragments"] += short["fragments"]/num_reads
                        # data[idx]["short.len_reg"] += (((short["length"]/max(seq_lens))*100)/reg_len)#/num_reads
                        # data[idx]["short.kc_len_reg"] += (short["kc_len"]/reg_len)#/num_reads

                        # sum_comp = sum(comp)
                        # for x, v in enumerate(comp):
                        #     print(v)
                        #     data[idx]["short."+str(x)] = v/sum_comp
                        # data[idx]["short.TTTAGG"] = comp[0]/sum_comp

                    data[idx]["short.num_regs"] = num_short_regions
                    # for m in ["mapq", "gc", "fragments", "coverage"]:
                    #     data[idx]["short."+m] = data[idx]["short."+m]/num_short_regions


                if chrom_reg["diff"].iloc[i] >= 700 and chromo not in set(["f", "r"]):
                    low = int(chrom_reg["chromStart"].iloc[i])
                    up = int(chrom_reg["chromEnd"].iloc[i])
                    reg_len = up-low
                    if chr_pres == True:
                        fchromo = chromo
                    if chr_pres == False:
                        fchromo = chromo.split("chr")[1]
                    aligns = bamfile.fetch(str(fchromo), low, up, multiple_iterators=True)
                    grps = itertools.groupby(sorted([x for x in aligns if x.cigartuples != None], key=lambda x: x.pos), key=lambda x: (x.pos, x.cigartuples[0][0]))
                    scqc = []
                    for pos, grp in grps:
                        scqc_grp = soft_clip_qual_corr(grp)
                        if scqc_grp != -1:
                            scqc.append(scqc_grp)
                    if math.isnan(np.mean(scqc)) == False:
                        data[idx][f"{chromo}_{i}.scqc"] = np.mean(scqc)
                    else:
                        data[idx][f"{chromo}_{i}.scqc"] = 1

                    region_len = reg_len+(2*100)
                    chromo_reg_len[chromo] += reg_len
                    data[idx][f"{chromo}_{i}.length"] = 0
                    telmer_dist[sample][f"{chromo}_{i}"] = []

                    coverage_array = np.array([], dtype="int64")
                    coverage_array = np.resize(coverage_array, int(region_len)+(2*100))
                    coverage_array.fill(0)
                    #coverage_array = np.zeros((coverage_array, int(region_len)), dtype="int64")

                    data[idx][f"{chromo}_{i}.coverage"] = 1
                    num_reads = 0
                    data[idx][f"{chromo}_{i}.gc"] = 0
                    data[idx][f"{chromo}_{i}.num_reads"] = 0
                    data[idx][f"{chromo}_{i}.mapq"] = 0

                    data[idx][f"{chromo}_{i}.kc_len"] = 0
                    data[idx][f"{chromo}_{i}.fragments"] = 0

                    direction = chrom_reg["direction"].iloc[i]
                    # comp = [0]*len(targets_f)
                    if chr_pres == True:
                        fchromo = chromo
                    if chr_pres == False:
                        fchromo = chromo.split("chr")[1]
                    for read in bamfile.fetch(str(fchromo), low, up, multiple_iterators=True):
                        if read.flag & 3840 > 0:
                            continue
                        else:
                            seq_len = read.query_alignment_length
                            seq_lens.add(seq_len)
                            seq = read.query_alignment_sequence
                            kc, a, blocks = count_variant_repeats(seq, targets_dict[direction], targets_to_array_dict[direction], direction)
                            # comp = [x+y for x,y in zip(comp, a)]
                            if chromo not in ["f", "r"]:
                                if read.mapq == 0:
                                    mq = "mapq0"
                                if read.mapq == 1:
                                    mq = "mapq1"
                            if len(blocks) == 0:
                                blocks.append(0)
                            if kc >= 1:
                                num_reads += 1
                                data[idx][f"{chromo}_{i}.length"] += seq_len
                                coverage_array = add(read, coverage_array, low, False, chromo)
                                data[idx][f"{chromo}_{i}.gc"] += (seq.count("G")+seq.count("C"))/seq_len
                                data[idx][f"{chromo}_{i}.mapq"] += read.mapq
                                for b in blocks:
                                    telmer_dist[sample][f"{chromo}_{i}"].append(b)
                                data[idx][f"{chromo}_{i}.kc_len"] += sum(blocks)
                                data[idx][f"{chromo}_{i}.fragments"] += len(blocks)
                            if chromo not in set(["f", "r"]):
                                if read.mapq < 2:
                                    seq_lens.add(seq_len)
                                    data[idx][f"{mq}.num_reads"] += 1
                                    data[idx][f"{mq}.length"] += seq_len
                                    data[idx][f"{mq}.gc"] += (seq.count("G")+seq.count("C"))/seq_len
                                    for b in blocks:
                                        telmer_dist[sample][f"{mq}"].append(b)
                                    data[idx][f"{mq}.kc_len"] += sum(blocks)
                                    data[idx][f"{mq}.fragments"] += len(blocks)

                            else:
                                continue
                    if num_reads != 0:
                        data[idx][f"{chromo}_{i}.length"] = (data[idx][f"{chromo}_{i}.length"]/max(seq_lens))*100
                        data[idx][f"{chromo}_{i}.gc"] = data[idx][f"{chromo}_{i}.gc"]/num_reads
                        data[idx][f"{chromo}_{i}.num_reads"] = num_reads
                        data[idx][f"{chromo}_{i}.mapq"] = data[idx][f"{chromo}_{i}.mapq"]/num_reads

                        # if len(telmer_dist[sample][f"{chromo}_{i}"]) != 0:
                        #     data[idx][f"{chromo}_{i}.longest"] = max(telmer_dist[sample][f"{chromo}_{i}"])
                        # if len(telmer_dist[sample][f"{chromo}_{i}"]) == 0:
                        #     data[idx][f"{chromo}_{i}.longest"] = 0

                        current_cov = 0

                        bpe = 100
                        for k in range(len(coverage_array)):
                            v = coverage_array[k]
                            current_cov += v
                            coverage_array[k] = current_cov
                            if current_cov < 0:
                                coverage_array[k] = 0  # sanity check
                        coverage = np.mean(coverage_array[bpe:-bpe])
                        #print("region", chromo, coverage)
                        if coverage > 0:
                            data[idx][f"{chromo}_{i}.coverage"] = coverage
                        else:
                            data[idx][f"{chromo}_{i}.coverage"] = 1 # data[idx][f"{chromo}_{i}.length"]/(region_len)

                        data[idx][f"{chromo}_{i}.fragments"] = data[idx][f"{chromo}_{i}.fragments"]/num_reads
                        data[idx][f"{chromo}_{i}.len_reg"] = (data[idx][f"{chromo}_{i}.length"]/reg_len)#/num_reads
                        data[idx][f"{chromo}_{i}.kc_len_reg"] = (data[idx][f"{chromo}_{i}.kc_len"]/reg_len)#/num_reads

                        # sum_comp = sum(comp)
                        # print(comp)
                        # for x, v in enumerate(comp):
                        #     data[idx][f"{chromo}_{i}.{x}"] = v/sum_comp
                        # data[idx][f"{chromo}_{i}.TTTAGG"] = comp[0]/sum(comp)

                if chromo in ["f", "r"]:
                    low = int(chrom_reg["chromStart"].iloc[i])
                    up = int(chrom_reg["chromEnd"].iloc[i])
                    reg_len = up-low
                    if chr_pres == True:
                        fchromo = chromo
                    if chr_pres == False:
                        fchromo = chromo.split("chr")[1]
                    aligns = bamfile.fetch(str(fchromo), low, up, multiple_iterators=True)
                    grps = itertools.groupby(sorted([x for x in aligns if x.cigartuples != None], key=lambda x: x.pos), key=lambda x: (x.pos, x.cigartuples[0][0]))
                    scqc = []
                    for pos, grp in grps:
                        scqc_grp = soft_clip_qual_corr(grp)
                        if scqc_grp != -1:
                            scqc.append(scqc_grp)
                    if math.isnan(np.mean(scqc)) == False:
                        data[idx][f"{chromo}_{i}.scqc"] = np.mean(scqc)
                    else:
                        data[idx][f"{chromo}_{i}.scqc"] = 1

                    region_len = reg_len+(2*100)
                    chromo_reg_len[chromo] += reg_len
                    data[idx][f"{chromo}_{i}.length"] = 0
                    telmer_dist[sample][f"{chromo}_{i}"] = []

                    direction = chrom_reg["direction"].iloc[i]
                    # comp = [0]*len(targets_f)
                    if chr_pres == True:
                        fchromo = chromo
                    if chr_pres == False:
                        fchromo = chromo.split("chr")[1]
                    for read in bamfile.fetch(str(fchromo), low, up, multiple_iterators=True):
                        if read.flag & 3840 > 0:
                            continue
                        else:
                            seq_len = read.query_alignment_length
                            #if seq_len < 95:
                            #    continue
                            seq_lens.add(seq_len)
                            seq = read.query_alignment_sequence
                            kc, a, blocks = count_variant_repeats(seq, targets_dict[direction], targets_to_array_dict[direction], direction)
                            # comp = [x+y for x,y in zip(comp, a)]

                            strand_dict[chromo][strandedness(chromo, read)] += 1
                            if read.is_reverse:
                                sc = "r"
                            else:
                                sc = "f"
                            # try:
                            #     if read.query_alignment_start-read.reference_start and read.reference_end-read.query_alignment_end > 0:
                            #         if read.query_alignment_length < 30:
                            #             print(read.query_sequence)
                            # except:
                            #     pass

                            num_reads_fr[chromo] += 1
                            if len(blocks) == 0:
                                blocks.append(0)
                            if kc >= 1:
                                num_reads_fr[chromo] += 1
                                data[idx][f"{chromo}_{i}.length"] += seq_len
                                coverage_array_fr[chromo] = add(read, coverage_array_fr[chromo], low, False, chromo)
                                coverage_array_strand[sc] = add(read, coverage_array_strand[sc], low, False, chromo)
                                data[idx][f"{chromo}_{i}.gc"] += (seq.count("G")+seq.count("C"))/seq_len
                                data[idx][f"{chromo}_{i}.mapq"] += read.mapq
                                for b in blocks:
                                    telmer_dist[sample][f"{chromo}_{i}"].append(b)
                                data[idx][f"{chromo}_{i}.kc_len"] += sum(blocks)
                                data[idx][f"{chromo}_{i}.fragments"] += len(blocks)

                    if chromo == "f":
                        processed_f = True
                    if chromo == "r":
                        processed_r = True

                if using_fr:
                    if processed_f and processed_r:
                        # print(f"unmapped {unmapped} \n paired {paired} \n unpaired {unpaired}")
                        #print(strand_dict)
                        data[idx]["ff"] = strand_dict["f"]["f"]
                        data[idx]["fr"] = strand_dict["f"]["r"]
                        data[idx]["rf"] = strand_dict["r"]["f"]
                        data[idx]["rr"] = strand_dict["r"]["r"]

                        ffc = strand_dict["f"]["f"]/(strand_dict["f"]["f"]+strand_dict["f"]["r"])
                        frc = strand_dict["r"]["f"]/(strand_dict["r"]["f"]+strand_dict["r"]["r"])
                        ftc = (strand_dict["f"]["r"]+strand_dict["r"]["r"])/(strand_dict["f"]["f"]+strand_dict["f"]["r"]+strand_dict["r"]["f"]+strand_dict["r"]["r"])
                        fsb = abs(ffc-frc)/ftc

                        data[idx]["ffc"] = ffc
                        data[idx]["frc"] = frc
                        data[idx]["ftc"] = ftc
                        data[idx]["fsb"] = fsb

                        rfc = strand_dict["f"]["r"]/(strand_dict["f"]["f"]+strand_dict["f"]["r"])
                        rrc = strand_dict["r"]["r"]/(strand_dict["r"]["f"]+strand_dict["r"]["r"])
                        rtc = (strand_dict["f"]["f"]+strand_dict["r"]["f"])/(strand_dict["f"]["f"]+strand_dict["f"]["r"]+strand_dict["r"]["f"]+strand_dict["r"]["r"])
                        rsb = abs(rfc-rrc)/rtc
                        data[idx]["rfc"] = rfc
                        data[idx]["rrc"] = rrc
                        data[idx]["rtc"] = rtc
                        data[idx]["rsb"] = rsb

                        #fsb = max([(rfc/frc)/rtc ,
                        #           (rrc*ffc)/rtc])
                        #rsb = max([(ffc/rrc)/ftc ,
                        #           (frc*rfc)/ftc])

                        # print(ffc-frc, rfc-rrc)
                        #print(fsb, rsb)
                        for chromo in ["f", "r"]:
                            low = int(coords[coords["chrom"] == chromo]["chromStart"].iloc[0])
                            up = int(coords[coords["chrom"] == chromo]["chromEnd"].iloc[0])
                            reg_len = up-low
                            if num_reads_fr[chromo] != 0:
                                i = 0
                                data[idx][f"{chromo}_{i}.length"] = (data[idx][f"{chromo}_{i}.length"]/max(seq_lens))*100
                                data[idx][f"{chromo}_{i}.gc"] = data[idx][f"{chromo}_{i}.gc"]/num_reads_fr[chromo]
                                data[idx][f"{chromo}_{i}.num_reads"] = num_reads_fr[chromo]
                                data[idx][f"{chromo}_{i}.mapq"] = data[idx][f"{chromo}_{i}.mapq"]/num_reads_fr[chromo]

                            # if len(telmer_dist[sample][f"{chromo}_{i}"]) != 0:
                            #     data[idx][f"{chromo}_{i}.longest"] = max(telmer_dist[sample][f"{chromo}_{i}"])
                            # if len(telmer_dist[sample][f"{chromo}_{i}"]) == 0:
                            #     data[idx][f"{chromo}_{i}.longest"] = 0

                            current_cov = 0
                            bpe = 100
                            for k in range(len(coverage_array_fr[chromo])):
                                v = coverage_array_fr[chromo][k]
                                current_cov += v
                                coverage_array_fr[chromo][k] = current_cov
                                if current_cov < 0:
                                    coverage_array_fr[chromo][k] = 0  # sanity check
                            coverage = np.mean(coverage_array_fr[chromo][bpe:-bpe])

                            current_cov_sc = 0
                            #bpe = 100
                            for k in range(len(coverage_array_strand[chromo])):
                                v = coverage_array_strand[chromo][k]
                                current_cov_sc += v
                                coverage_array_strand[chromo][k] = current_cov_sc
                                if current_cov < 0:
                                    coverage_array_strand[chromo][k] = 0  # sanity check
                            coverage_sc = np.mean(coverage_array_strand[chromo][bpe:-bpe])
                            #print(coverage_sc)
                            #print("region", chromo, coverage)
                            if coverage > 0:
                                if chromo == "f":
                                    data[idx][f"{chromo}_{i}.unadj_coverage"] = coverage
                                    data[idx][f"{chromo}_{i}.coverage"] = coverage/(ffc)
                                if chromo == "r":
                                    data[idx][f"{chromo}_{i}.unadj_coverage"] = coverage
                                    data[idx][f"{chromo}_{i}.coverage"] = coverage/(rfc)
                            else:
                                data[idx][f"{chromo}_{i}.coverage"] = 1 # data[idx][f"{chromo}_{i}.length"]/(region_len)

                            if coverage_sc > 0:
                                if chromo == "f":
                                    data[idx][f"{chromo}_{i}.coverage_sc"] = coverage_sc#/(ffc)
                                if chromo == "r":
                                    data[idx][f"{chromo}_{i}.coverage_sc"] = coverage_sc#/(rfc)
                            else:
                                data[idx][f"{chromo}_{i}.coverage_sc"] = 1 # data[idx][f"{chromo}_{i}.length"]/(region_len)

                            data[idx][f"{chromo}_{i}.fragments"] = data[idx][f"{chromo}_{i}.fragments"]/num_reads_fr[chromo]
                            data[idx][f"{chromo}_{i}.len_reg"] = (data[idx][f"{chromo}_{i}.length"]/reg_len)#/num_reads
                            data[idx][f"{chromo}_{i}.kc_len_reg"] = (data[idx][f"{chromo}_{i}.kc_len"]/reg_len)#/num_reads

                        processed_f = False
                        processed_r = False
        ## For including unmapped reads (used in previous model)
        # bamfile = pysam.AlignmentFile(os.path.join(in_dir, f), "rb", threads=n_threads)
        # for read in bamfile:#.fetch(until_eof=True):
        #     if read.is_unmapped == True:
        #         #seq = read.query_alignment_sequence
        #         cf, af, bf = count_variant_repeats(read.query_alignment_sequence.upper(), targets_f, targets_to_array_f, "forward")
        #         cr, ar, br = count_variant_repeats(read.query_alignment_sequence.upper(), targets_r, targets_to_array_r, "reverse")
        #         if cf >= cr:
        #             kc = cf
        #             a = af
        #             blocks = bf
        #         if cr > cf:
        #             kc = cr
        #             a = ar
        #             blocks = br
        #
        #         if len(blocks) == 0:
        #             blocks.append(0)
        #         ## Low threshold as trained on trimmed files
        #         if kc >= 1:
        #             seq = read.query_alignment_sequence
        #             seq_len = read.query_alignment_length
        #             data[idx]["unmapped.num_reads"] += 1
        #             data[idx]["unmapped.length"] += seq_len
        #             coverage_array = add(read, coverage_array, low, False)
        #             data[idx]["unmapped.gc"] += (seq.count("G")+seq.count("C"))/seq_len
        #             #for b in blocks:
        #             #    telmer_dist[sample][f"{chromo}_{i}"].append(b)
        #             data[idx]["unmapped.kc_len"] += sum(blocks)
        #             data[idx]["unmapped.fragments"] += len(blocks)

        if short["num_reads"] > 0:
            data[idx]["short.length"] = (short["length"]/max(seq_lens))*100
            data[idx]["short.gc"] = short["gc"]/short["num_reads"]
            data[idx]["short.num_reads"] = short["num_reads"]
            data[idx]["short.mapq"] = short["mapq"]/short["num_reads"]
            data[idx]["short.coverage"] = np.mean(short["coverage"])
            data[idx]["short.fragments"] += short["fragments"]/short["num_reads"]
            #data[idx]["short.len_reg"] += (((short["length"]/max(seq_lens))*100)/reg_len)#/num_reads
            #data[idx]["short.kc_len_reg"] += (short["kc_len"]/reg_len)#/num_reads

        for chromo in ["mapq0", "mapq1"]: #, "unmapped"]:
            data[idx][f"{chromo}.length"] = (data[idx][f"{chromo}.length"]/max(seq_lens))*100

    for chromo in ["mapq0", "mapq1"]: #, "unmapped"]:
        for idx in range(len(data)):
            if data[idx][f"{chromo}.num_reads"] != 0:
                data[idx][f"{chromo}.gc"] = data[idx][f"{chromo}.gc"]/data[idx][f"{chromo}.num_reads"]
                data[idx][f"{chromo}.fragments"] = data[idx][f"{chromo}.fragments"]/data[idx][f"{chromo}.num_reads"]
            else:
                data[idx][f"{chromo}.gc"] = 0

    table = pd.DataFrame.from_dict(data)
    len_cov_regions = table.columns.difference(["sample"])
    unique_regions = list(set([r.split(".")[0] for r in len_cov_regions]))
    for mapq in ["mapq0", "mapq1", "short", "ff", "fr", "rf", "rr",
                    "ffc", "frc", "ftc", "fsb", "rfc", "rrc", "rtc", "rsb"]: #, "unmapped"]:
        unique_regions.remove(mapq)

    print(unique_regions)

    df = table.copy()
    chrom_list = list(set([c.split("_")[0] for c in unique_regions]))

    region_names = {}
    region_count = {}
    for chrom in chrom_list:
        region_names[chrom] = []
        region_count[chrom] = 0
    for region in unique_regions:
        region_names[region.split("_")[0]].append(region)
        region_count[region.split("_")[0]] += 1

    df["avg_cov"] = 1
    # for bam_file in bam_files:
    #     sample_name = os.path.basename(bam_file).split("_")[0]
    #     df.at[df["sample"] == sample_name, "avg_cov"] = avg_coverages[sample_name]

    df["total_sc"] = df["f_0.coverage_sc"]+df["r_0.coverage_sc"]
    df["f_0.coverage_sc"] = df["f_0.coverage_sc"]/df["total_sc"]
    df["r_0.coverage_sc"] = df["r_0.coverage_sc"]/df["total_sc"]

    for mapq in ["mapq0", "mapq1", "short"]: #, "unmapped"]:
        df[mapq+".num_reads"] = df[mapq+".num_reads"]/df["avg_cov"]
        df[mapq+".length"] = df[mapq+".length"]/df["avg_cov"]
        df[mapq+".kc_len"] = df[mapq+".kc_len"]/df["avg_cov"]

    df["short.adj_cov"] = df["short.coverage"]/df["avg_cov"]
    df["short.len_reg"] = df["short.len_reg"]/df["avg_cov"]
    df["short.kc_len_reg"] = df["short.kc_len_reg"]/df["avg_cov"]

    df["how_many"] = 0
    for region in unique_regions:
        #print(region)
        df[region+".kc_perc"] = df[region+".length"]/df[region+".kc_len"]
        ## Normalised by coverage values
        df[region+".num_reads"] = df[region+".num_reads"]/df["avg_cov"]
        df[region+".length"] = df[region+".length"]/df["avg_cov"]
        df[region+".kc_len"] = df[region+".kc_len"]/df["avg_cov"]
        df[region+".len_reg"] = df[region+".len_reg"]/df["avg_cov"]
        df[region+".kc_len_reg"] = df[region+".kc_len_reg"]/df["avg_cov"]

        # df[region+".adj_length"] = df[region+".length"]-df[region+".overlap"]
        # df[region+".kc_perc"] = df[region+".kc_perc"]/df[region+".num_reads"]
        # df[region+".kc_len"] = df[region+".length"]*df[region+".kc_perc"]
        # df[region+".kc_perc"] = df[region+".kc"]/(df[region+".length"]-(6*df[region+".num_reads"]))

        df[region+".adj_cov"] = df[region+".coverage"]/df["avg_cov"]
        df = df.drop([region+".coverage"], axis=1)

        df[region+".kc_len_cov"] = df[region+".kc_len"]/df[region+".adj_cov"]
        df[region+".new"] = df[region+".length"]/(df[region+".adj_cov"]*10)
        df["tmp_count"] = np.where(df[region+".num_reads"]>0, 1, 0)
        df["how_many"] += df["tmp_count"]
        df[region+".avg_read_len"] = df[region+".length"]/df[region+".num_reads"]

    df = df.drop(["tmp_count"], axis=1)

    df["total_length"]      = df[[r+".length" for r in unique_regions+["mapq0", "mapq1", "short"]]].sum(axis=1)
    df["total_kc_length"]   = df[[r+".kc_len" for r in unique_regions+["mapq0", "mapq1", "short"]]].sum(axis=1)
    df["total_kc_perc"]     = df[[r+".kc_perc" for r in unique_regions]].sum(axis=1)/df["how_many"]
    # df["total_coverage"]    = df[[r+".adj_cov" for r in unique_regions+["short"]]].sum(axis=1)
    df["total_coverage"]    = df[[r+".adj_cov" for r in unique_regions]].sum(axis=1)
    # df["total_kc_new"] = (df["total_kc_length"]/df["total_coverage"])/df["how_many"]
    df["total_kc_cov"]      = df[[r+".kc_len_cov" for r in unique_regions]].sum(axis=1)/df["how_many"]
    df["total_adj_cov"]     = df[[r+".adj_cov" for r in unique_regions]].sum(axis=1)
    df["total_how_coverage"] = df["total_coverage"]/df["how_many"]
    df["cov_multiplier"]    = df["total_how_coverage"]/df["avg_cov"] # same as avg_adj_cov
    # df["total_adj_length"] = df[[r+".adj_length" for r in unique_regions]].sum(axis=1)
    df["total_reads"]       = df[[r+".num_reads" for r in unique_regions+["mapq0", "mapq1", "short"]]].sum(axis=1)
    df["total_avg_read_len"] = df[[r+".avg_read_len" for r in unique_regions]].sum(axis=1)/df["how_many"]
    df["total_avg_gc"]      = df[[r+".gc" for r in unique_regions]].sum(axis=1)/df["how_many"]
    df["len_cov"]           = df["total_length"]/df["total_coverage"]
    df["avg_adj_cov"]       = df["total_adj_cov"]/df["how_many"]
    df["new_cov"]           = df["total_coverage"]/df["avg_adj_cov"]
    df["total_new"]         = df[[r+".new" for r in unique_regions]].sum(axis=1)/df["how_many"]
    # df["total_new_96"]      = df[[r+".new" for r in unique_regions]].sum(axis=1)/96

    df["total_len_reg"]     = df[[r+".len_reg" for r in unique_regions]].sum(axis=1)
    df["total_kc_len_reg"]  = df[[r+".kc_len_reg" for r in unique_regions]].sum(axis=1)
    df["avg_len_reg"]       = df["total_len_reg"]/df["how_many"]
    df["avg_kc_len_reg"]    = df["total_kc_len_reg"]/df["how_many"]

    # df["new_cov"]           = (df["total_new"]/df["total_coverage"])/100
    # df["diff_new_mult_cov"] = df["new_cov"]/df["avg_adj_cov"]
    # df["diff_how_new_cov"]  = df["cov_multiplier"]-(df["avg_cov"]/df["total_coverage"])/df["total_how_coverage"]

    df["avg_tl_cov"] = df[[r+".adj_cov" for r in unique_regions]].mean(axis=1)
    df["std_tl_cov"] = df[[r+".adj_cov" for r in unique_regions]].std(axis=1)

    for i in ["f_0", "r_0", "2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0", "short"]: #, "short", "mapq0", "mapq1"]:
        df[i+".perc_cov"] = df[i+".adj_cov"]/df["total_coverage"]
    #for i in ["2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0"]:
    df["rest_0.perc_cov"] = df[[r+".perc_cov" for r in ["2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0"]]].sum(axis=1)
    #df["short.perc_cov"] = df["short.adj_cov"]/df["total_coverage"]

    # df = df.drop(["avg_cov"], axis=1)

    table.to_csv("lets_see.csv", index=False)

    return df


def feature_select(in_data, paired, model, pred_type, selected_types=["from_mod"]):
    print("Performing feature selection...")
    if paired == False:
        sample_cols = ["sample"]
        if pred_type == "regression":
            real = ["stela"]
        if pred_type == "classification":
            real = ["short"]
    if paired == True:
        sample_cols = ["sample_normal", "sample_tumour"]
        if pred_type == "regression":
            real = ["stela_normal", "stela_tumour"]
        if pred_type == "classification":
            real = ["short_normal", "short_tumour"]

    data = in_data.copy()
    data = data.replace(0,np.nan).dropna(axis=1,how="all")
    data = data.replace([np.nan, np.inf, -np.inf], 0)
    pred_cols = data.columns.difference(list(chain.from_iterable([sample_cols, real, ["how_diff"]])))

    def low_variance(data, pred_cols, sample_cols):
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit_transform(data[pred_cols])
        # pred_cols = data.columns.difference(list(chain.from_iterable([sample_cols, real])))
        return data

    def univariate(data, pred_cols, real, sample_cols, k_val=1):
        selector = SelectKBest(f_regression, k=k_val)
        sel_out = selector.fit_transform(data[pred_cols], np.ravel(data[real]))
        unifeat = data.columns[selector.get_support(indices=True)].tolist()
        # print(unifeat)
        data = data[list(chain.from_iterable([unifeat, real]))]
        # pred_cols = data.columns.difference(list(chain.from_iterable([sample_cols, real])))
        return data

    def feat_select_model(data, pred_cols, real, sample_cols, model, paired):
        copy = data.copy()
        mod_dict = {
            ## Regressors
            "lasso":    Lasso(),
            "ridge":    Ridge(),
            "rfr":      RandomForestRegressor(),
            "linear":   LinearRegression(),
            "ransac":   RANSACRegressor(),
            "elastic":  ElasticNet(),
            "mlpr":     MLPRegressor(),
            "gbmr":     LGBMRegressor(boosting_type="dart"),
            ## Classifiers
            "neighbors":    KNeighborsClassifier(3),
            "LSVC":         SVC(kernel="linear", C=0.025, probability=True),
            "SVC":          SVC(gamma=2, C=1, probability=True),
            "gp":           GaussianProcessClassifier(1.0 * RBF(1.0)),
            "tree":         DecisionTreeClassifier(max_depth=5), # depth prevent overfit
            "rfc":          RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            "mlpc":         MLPClassifier(alpha=1, max_iter=1000),
            "ada":          AdaBoostClassifier(),
            "gaus":         GaussianNB(),
            "quad":         QuadraticDiscriminantAnalysis(),
            "gbmc":         LGBMClassifier(max_depth=3, num_leaves=3, extra_trees=True, max_bin=30, boosting_type="dart")
            }
        mod = mod_dict[model]

        hyperparameter_tuning = False

        param_grid = {"num_leaves": range(3, 6),
                      "max_depth": range(3, 6),
                      "max_bin": range(30, 80, 5),
                      "extra_trees": [True, False],
                      "path_smooth": [0, 0.05, 0.1, 0.15],
                      "n_estimators": range(30, 150, 10),
                      "learning_rate": [0.1, 0.2, 0.3]
                      }

        if hyperparameter_tuning == True:
            grid = RandomizedSearchCV(mod, param_grid, refit=True, n_iter=1000, scoring="f1")
            grid.fit(copy[pred_cols], np.ravel(copy[real]))
            print(grid.best_params_)
            mod = grid.best_estimator_
        else:
            mod.fit(copy[pred_cols], np.ravel(copy[real]))

        # try:
        #     if paired == False:
        #         mod = mod.fit(copy[pred_cols], np.ravel(copy[real]))
        #     if paired == True:
        #         mod = mod.fit(copy[pred_cols], copy[real])
        # except:
        #     mod = LGBMRegressor(max_depth=3, num_leaves=3, extra_trees=True, max_bin=30, boosting_type="dart")
        #     if paired == False:
        #         mod = mod.fit(copy[pred_cols], np.ravel(copy[real]))
        #     if paired == True:
        #         mod = mod.fit(copy[pred_cols], copy[real])
        # print(clf.feature_importances_)
        model = SelectFromModel(mod, prefit=True)#max_features=50, threshold="median", prefit=True)
        sfm = model.transform(copy[pred_cols])
        sfm_cols = data.columns[model.get_support(indices=True)].tolist()
        data = data[list(chain.from_iterable([sample_cols, sfm_cols, real]))]
        # pred_cols = data.columns.difference(list(chain.from_iterable([sample_cols, real])))
        return data

    def recursive(data, pred_cols, real, sample_cols, model, paired, n_feat=None, cv=False):
        all = data.copy()
        mod_dict = {
            ## Regressors
            "lasso":    Lasso(),
            "ridge":    Ridge(),
            "rfr":      RandomForestRegressor(),
            "linear":   LinearRegression(),
            "ransac":   RANSACRegressor(),
            "elastic":  ElasticNet(),
            "mlpr":     MLPRegressor(),
            "gbmr":     LGBMRegressor(boosting_type="dart"),
            ## Classifiers
            "neighbors":    KNeighborsClassifier(3),
            "LSVC":         SVC(kernel="linear", C=0.025),
            "SVC":          SVC(gamma=2, C=1),
            "gp":           GaussianProcessClassifier(1.0 * RBF(1.0)),
            "tree":         DecisionTreeClassifier(max_depth=5), # depth prevent overfit
            "rfc":          RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            "mlpc":         MLPClassifier(alpha=1, max_iter=1000),
            "ada":          AdaBoostClassifier(),
            "gaus":         GaussianNB(),
            "quad":         QuadraticDiscriminantAnalysis(),
            "gbmc":         LGBMClassifier(max_depth=3, num_leaves=3, extra_trees=True, max_bin=30, boosting_type="dart")
            }
        mod = mod_dict[model]

        hyperparameter_tuning = False

        param_grid = {"num_leaves": range(3, 6),
                      "max_depth": range(3, 6),
                      "max_bin": range(30, 80, 5),
                      "extra_trees": [True, False],
                      "path_smooth": [0, 0.05, 0.1, 0.15],
                      "n_estimators": range(30, 150, 10),
                      "learning_rate": [0.1, 0.2, 0.3]
                      }

        if hyperparameter_tuning == True:
            grid = RandomizedSearchCV(mod, param_grid, refit=True, n_iter=1000, scoring="f1")
            grid.fit(all[pred_cols], np.ravel(all[real]))
            print(grid.best_params_)
            model = grid.best_estimator_
        #else:

        mod.fit(all[pred_cols], np.ravel(all[real]))

        if cv == False:
            rfe = RFE(estimator=mod, n_features_to_select=n_feat, step=1)
        if cv == True:
            rfe = RFECV(estimator=mod)
        if paired == False:
            new = rfe.fit_transform(data[pred_cols], np.ravel(data[real]))
        if paired == True:
            new = rfe.fit_transform(data[pred_cols], data[real])
        sel_cols = data.columns[rfe.get_support(indices=True)].tolist()
        data = data[list(chain.from_iterable([sample_cols, sel_cols, real]))]
        data = data.loc[:,~data.columns.duplicated()]
        # print(data)
        # print(rfe.n_features_)
        # print(rfe.ranking_)
        return data

    def sequential(model, data, pred_cols, real, pred_type):
        print("Sequential feature selection...")
        mod_dict = {
            ## Regressors
            "lasso":    Lasso(),
            "ridge":    Ridge(),
            "rfr":      RandomForestRegressor(),
            "linear":   LinearRegression(),
            "ransac":   RANSACRegressor(),
            "elastic":  ElasticNet(),
            "mlpr":     MLPRegressor(),
            "gbmr":     LGBMRegressor(boosting_type="dart"),
            ## Classifiers
            "neighbors":    KNeighborsClassifier(3),
            "LSVC":         SVC(kernel="linear", C=0.025),
            "SVC":          SVC(gamma=2, C=1),
            "gp":           GaussianProcessClassifier(1.0 * RBF(1.0)),
            "tree":         DecisionTreeClassifier(max_depth=5), # depth prevent overfit
            "rfc":          RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            "mlpc":         MLPClassifier(alpha=1, max_iter=1000),
            "ada":          AdaBoostClassifier(),
            "gaus":         GaussianNB(),
            "quad":         QuadraticDiscriminantAnalysis(),
            "gbmc":         LGBMClassifier(max_depth=3, num_leaves=3, extra_trees=True, max_bin=30, boosting_type="dart")
            }
        mod = mod_dict[model]

        hyperparameter_tuning = False

        param_grid = {"num_leaves": range(3, 6),
                      "max_depth": range(3, 6),
                      "max_bin": range(30, 80, 5),
                      "extra_trees": [True, False],
                      "path_smooth": [0, 0.05, 0.1, 0.15],
                      "n_estimators": range(30, 150, 10),
                      "learning_rate": [0.1, 0.2, 0.3]
                      }

        if hyperparameter_tuning == True:
            grid = RandomizedSearchCV(mod, param_grid, refit=True, n_iter=1000, scoring="f1")
            grid.fit(data[pred_cols], np.ravel(data[real]))
            print(grid.best_params_)
            mod = grid.best_estimator_
        else:
            mod.fit(data[pred_cols], np.ravel(data[real]))

        sfs = SequentialFeatureSelector(mod, n_features_to_select=30)
        sfs.fit(data[pred_cols], np.ravel(data[real]))
        remove = sfs.get_support()
        cols = [d for (d, r) in zip(pred_cols, remove) if not r]
        cols.append("stela")
        cols.append("sample")
        if pred_type == "classification":
            cols.append("short")

        return data[cols]

    def remove_correlated(data, pred_cols, sample_cols, real):
        cdata = data.iloc[:,1:-1]
        corr = cdata[pred_cols].corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        sel_pred_cols = list(cdata.columns[columns])
        sel_cols = list(chain.from_iterable([sample_cols, sel_pred_cols, real]))
        #data = data[list(chain.from_iterable([sample_cols, sel_cols, real]))]
        return data[sel_cols]


    for num, method in enumerate(selected_types):
        print(f"on: {method} ({num+1}/{len(selected_types)})", end="\r")
        if method == "low_var":
            data = low_variance(data, pred_cols, sample_cols)
            data = data.loc[:,~data.columns.duplicated()]
            pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
        if method == "univar":
            data = univariate(data, pred_cols, real, sample_cols, k_val=1)
            data = data.loc[:,~data.columns.duplicated()]
            pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
        if method == "from_mod":
            try:
                data = feat_select_model(data, pred_cols, real, sample_cols, model, paired)
                data = data.loc[:,~data.columns.duplicated()]
                pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
            except:
                print("feature selection from model not available")
                continue
        if method == "recursive":
            try:
                data = recursive(data, pred_cols, real, sample_cols, model, paired, n_feat=100)
                data = data.loc[:,~data.columns.duplicated()]
                pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
            except:
                print("recursive feature selection not available")
                continue
        if method == "sequential":
            try:
                data = sequential(model, data, pred_cols, real, pred_type)
                data = data.loc[:,~data.columns.duplicated()]
                pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
            except:
                print("sequencial feature selection not available")
                continue
        if method == "corr":
            data = remove_correlated(data, pred_cols, sample_cols, real)
            data = data.loc[:,~data.columns.duplicated()]
            pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
        if method == None:
            data = data.loc[:,~data.columns.duplicated()]
            pred_cols = data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])

    pred_cols = pred_cols.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
    print(pred_cols)
    return pred_cols




def telmer_adjacnet(seq, targets, k=6):
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



def make_adjacnet(in_file):
    targets_to_array_f = {"CCCTAA": 0, "CCCTGA": 1, "CCCGAA": 2, "CCCTAC": 3, "CCCTCA": 4, "CCCCAA": 5, "CCCTTA": 6,
                           "CCCTAT": 7, "CCCTAG": 8, "CCCAAA": 9}
    targets_to_array_f.update({"CCCTTAA": 10, "CCCACT": 11, "CCCCAT": 12, "CCCGCA": 13, "CCCGCT": 14, "CCCTCT": 15})

    targets_to_array_r = {"TTAGGG": 0, "TCAGGG": 1, "TTCGGG": 2, "GTAGGG": 3, "TGAGGG": 4, "TTGGGG": 5, "TAAGGG": 6,
                       "ATAGGG": 7, "CTAGGG": 8, "TTTGGG": 9}
    targets_to_array_r.update({"TTAAGGG": 10, "AGTGGG": 11, "ATGGGG": 12, "TGCGGG": 13, "AGCGGG": 14, "AGAGGG": 15})

    for i in targets_to_array_r:
        targets_to_array_r[i] += 16

    targets_to_array_both = {}
    targets_to_array_both.update(targets_to_array_f)
    targets_to_array_both.update(targets_to_array_r)

    targets_f = make_rotation_keys(targets_to_array_f)
    targets_r = make_rotation_keys(targets_to_array_r)
    targets_both = make_rotation_keys(targets_to_array_both)

    bam_file = pysam.AlignmentFile(os.path.join("/home/alex/Desktop/uni/PhD/TL_prediction/fastq_scan", in_file))
    G = nx.DiGraph()
    G.add_node(-1, size = 0)
    for i in set(targets_both.values()):
        G.add_node(i, size = 0)

    #print(G.nodes(data=True))

    for r in bam_file:
        t = list(telmer_adjacnet(r.query_sequence, targets_both, k=6))
        for n, i in enumerate(t[:-1]):
            if i == t[n+1]:
                G.nodes[i]["size"] += 1
            if G.has_edge(i, t[n+1]):
                G[i][t[n+1]]["weight"] += 1
                #print(G)
            else:
                G.add_edge(i, t[n+1], weight=1)

    #nx.draw(G, with_labels=True)
    #plt.show()
    #print(nx.to_numpy_matrix(G))
    return G


def dict_adjacent(in_files, coverage, short):
    graphs = {}

    bam_files = os.listdir(in_files)
    bam_files = [f for f in bam_files if f.endswith("_tel.bam")]
    bam_files = sorted(bam_files)

    avg_coverages_file = pd.read_csv(coverage)
    avg_coverages = {}
    for col in avg_coverages_file:
        avg_coverages[col] = np.median(avg_coverages_file[col]/len(avg_coverages_file.index))

    short_table = pd.read_csv(short)
    short_table['short'] = np.where(short_table['stela'] < 3.810 , True, False)
    print(short_table)

    for i in bam_files:
        fn = os.path.splitext(os.path.basename(i))[0]
        fnwe = fn.split("_")[0]
        graphs[fnwe] = dict()
        try:
            graphs[fnwe]["short"] = short_table.loc[short_table["sample"] ==  fnwe,'short'].tolist()
        except:
            pass
        graphs[fnwe]["G"] = make_adjacnet(i)
        for u in graphs[fnwe]["G"]:
            graphs[fnwe]["G"].nodes[u]["size"] = [graphs[fnwe]["G"].nodes[u]["size"]/avg_coverages[fnwe]]
            for v in graphs[fnwe]["G"][u]:
                graphs[fnwe]["G"][u][v]["weight"] = graphs[fnwe]["G"][u][v]["weight"]/avg_coverages[fnwe]

    joblib.dump(graphs, "graphs_hap1")
    print(graphs)

def model_net():
    model = nn.Sequential(
    nn.something()
    )

    optimiser = optim.SGD(model.parameters, lr = 0.01)

    loss = nn.CrossEntropyLoss()

    nb_epochs = 5
    for epochs in range(nb_epochs):
        losses = list()
        for batch in train_loader:
            x, y = batch

            b = x.size(0)
            x = x.view(b, -1)

            # forward
            l = model(x) # l: logits
            # compute objective function
            j = loss(l, y)
            # clearning gradients
            model.zero_grad()
            # accumulate partial dertivatives of j wrt parms
            j.backward()
            # opp grad
            optimiser.step()

            losses.append(j.item())
        print(f"Epoch {epoch + 1}, traim loss: {torch.tensor(losses).mean():.2f}")

        val_losses = list()
        for batch in train_loader:
            x, y = batch

            b = x.size(0)
            x = x.view(b, -1)

            # forward
            with torch.no_grad():
                l = model(x) # l: logits
            # compute objective function
            j = loss(l, y)

            val_losses.append(j.item())

        print(f"Epoch {epoch + 1}, val loss: {torch.tensor(val_losses).mean():.2f}")
