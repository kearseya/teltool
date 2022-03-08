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
import joypy
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

from teltool.cteltool import *

from pysam import AlignmentFile

import time

import progressbar

def get_n_reads(fn):
    f = AlignmentFile(fn)
    s = f.get_index_statistics()
    total = 0
    for i in s:
        total += i.mapped
        total += i.unmapped
    total += f.count(contig="*")
    return total

def time_scan_files(files, l, filter):
    filter = joblib.load(filter)
    for f in files:
        t0 = time.time()
        scan_bam(f, l, filter)
        print("TIME TAKEN:      ", time.time()-t0)

def scan_bam(in_file, l, filter):
    print("Kmers in set:    ", len(filter))
    file = pysam.AlignmentFile(in_file)
    read_pairs = dict()
    fastq_prefix = in_file.split(".bam")[0]
    total_reads = 0
    matched = 0
    n_reads_tot = 0
    n_reads_in = 0
    read_in_file = get_n_reads(in_file)
    with progressbar.ProgressBar(max_value = read_in_file, redirect_stdout=True) as bar:
        for a in file:
            total_reads += 1
            bar.update(total_reads)
            if not a.flag & 3328 and a.flag & 1:
                if a.qname not in read_pairs:
                    read_pairs[a.qname] = a
                else:
                    b = read_pairs[a.qname]
                    del read_pairs[a.qname]
                    # if "N" in a.query_alignment_sequence.upper() or b.query_alignment_sequence.upper():
                    #     n_reads_tot += 1
                    if proc_read_pari(a.query_alignment_sequence.upper(), b.query_alignment_sequence.upper(), filter, l):
                        matched += 1
                        # if "N" in a.query_alignment_sequence.upper() or b.query_alignment_sequence.upper():
                        #     n_reads_in += 1
                        # with open(fastq_prefix+"1.fq", "a") as paired1:
                        #     paired1.write(f"@{a.qname}\n{a.query_alignment_sequence}\n+\n{a.query_qualities}\n")
                        # with open(fastq_prefix+"2.fq", "a") as paired2:
                        #     paired2.write(f"@{b.qname}\n{b.query_alignment_sequence}\n+\n{b.query_qualities}\n")
    print("Total reads:     ", total_reads)
    print("Matched reads:   ", matched)
    print("total reads with n:      ", n_reads_tot)
    print("total matched with n:    ", n_reads_in)

def proc_read_pari(r1, r2, kmer_filter, kmer_len=27):
    if r1 and r2:
        # seq = r1.seq
        for i in range(len(r1) - kmer_len + 1):
            if r1[i:i + kmer_len] in kmer_filter:
                #print("r1:      ", r1)
                return True
        # seq = r2.seq
        for i in range(len(r2) - kmer_len + 1):
            if r2[i:i + kmer_len] in kmer_filter:
                #print("r2:      ", r2)
                return True
    return False


def py_collect_wanted_kmers(referece, coords, l):
    file = pysam.FastaFile(referece)
    kmer_set = set()
    for r in coords:
        region = file.fetch(r[0], r[1], r[2]).upper()
        for i in range(len(region)-l+1):
            kmer_set.add(region[i:i+l])
    dump(kmer_set, "python_kmers_32.set")
