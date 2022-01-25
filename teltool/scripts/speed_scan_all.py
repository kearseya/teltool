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
from pathlib import Path
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


bam_file_name = "/home/alex/Desktop/uni/PhD/TL_prediction/tmp/DB147_tel.bam" # /home/alex/Desktop/uni/PhD/TL_prediction/raw_data/full/DB143.bam"
bam = pysam.AlignmentFile(bam_file_name, "rb")
tot = 0
total = 0
for r in bam:#.fetch(until_eof=True):#, 5137033, 5137161):#until_eof=True):
    total += 1
    if "TTAGGG" not in r.seq and "CCCTAA" not in r.seq:
        tot += 1
    #for direction in ["forward", "reverse"]:
    #    kc, a, blocks = count_variant_repeats(r.seq, targets_dict[direction], targets_to_array_dict[direction], direction)
    #    if len(blocks) > 0:
    #        if max(blocks) >= 50:
    #            tot += 1
                #print(r.seq)
                #print(blocks, sum(blocks))
                #print(a)
                #print(r.reference_start)
print(bam_file_name)
#print("With TTAGGG CCCTAA filter")
print(tot, "/", total)
