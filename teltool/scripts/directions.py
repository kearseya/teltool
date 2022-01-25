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

from functions import *



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
coords = pd.read_csv("/home/alex/Desktop/uni/PhD/TL_prediction/bam_read/telomere_regions/hg38_all_scan_telomeres.tsv", sep="\t")
coords["direction"] = 0
print(coords)
reffile = pysam.FastaFile("hg38.fa")

for i in coords.index:
    print(i)
    chromo = coords["chrom"].iloc[i]
    low = int(coords["chromStart"].iloc[i])
    up = int(coords["chromEnd"].iloc[i])
    ref_region = reffile.fetch(chromo, low, up).upper()
    ref_kc_f, af, l = count_variant_repeats(ref_region, targets_dict["forward"], targets_to_array_dict["forward"], "forward")
    ref_kc_r, ar, l = count_variant_repeats(ref_region, targets_dict["reverse"], targets_to_array_dict["reverse"], "reverse")
    if ref_kc_f >= ref_kc_r:
        direction = "forward"
    else:
        direction = "reverse"
    coords["direction"].iloc[i] = direction

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(coords)

coords.to_csv("with_directions.tsv", sep='\t', index=False)
