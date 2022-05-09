#!/bin/python3
import sys, os
import warnings
import click
import pysam
import subprocess
import xml.etree.ElementTree as et
import json
from joblib import dump, load
import pandas as pd
import numpy as np
from numpy.random import choice
import math
from scipy import stats
from statistics import mode
from itertools import chain
import re
#from pathlib import Path
import collections
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import joypy
import seaborn as sns
# Import sci-kit learn
import sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
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
import lightgbm
from sklearn.ensemble import StackingClassifier
# Feature selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Domain adaptiation
import ot

# Threading (not used)
# import threading
# import multiprocessing
# import logging
# import concurrent.futures

# Others (not used)
# import glob
# from contextlib import contextmanager
# from mlxtend.regressor import LinearRegression
# from nltk import DecisionTreeClassifier, sys
# from collections import defaultdict
# from sknn.mlp import Classifier, Convolution, Layer

sys.path.append(str(os.path.dirname(__file__)))
from functions import *
from cteltool import *


@click.group(chain=True)
@click.option("-d", type=str, nargs=1, default=os.path.dirname(__file__),
help="Working directory (default current directory)")
@click.option("-r", "--reference", type=click.Choice(["telmer", "hg38", "hg19", "hg38c"], case_sensitive=False),
default="telmer", help="BAM reference genome build", show_default=True)
@click.option("--chr/--num", is_flag=True, flag_value=False, default=True,
help="Reference using chrN or N as chromosome names")
@click.option("--paired/--single", is_flag=True, flag_value=False, default=False,
help="Predicted output from using pairs")
@click.option("-v", is_flag=True, flag_value=True, default=False,
help="Verbose mode (progressbar for bam files)")
# @click.option("-l", type=str, nargs=1, default=None,
# help="Output log file (input name)")
@click.pass_context
class cli:
    """teltool uses ml models to predict telomere lengths from aligned files"""
    def __init__(self, ctx, d, reference, chr, paired, v): #, l):
        ctx.ensure_object(dict)
        ctx.obj["global"] = {"working_directory": d, "reference": reference, "chr": chr, "paired": paired, "verbose": v} #, "verbose": v, "output_log_file": l}









@cli.command()
@click.option("-i", default=".",
help="Input bam file")
@click.option("-o", default="trimmed",
help="Output directory")
@click.option("--kmer/--region", is_flag=True, flag_value=False, default=True,
help="Trim files by kmers (requires realignment) or region")
@click.option("-s", "--kset", default="telmers_rolling_32",
help="Set for filtering reads from file(s)")
@click.option("-r", "--ref", default=str(os.path.join(os.path.dirname(__file__), "reference", "hg38_cutout_edit.fa")),
help="Path to reference to align to")
@click.option("-k", default=False, flag_value=True,
help="Keep fastq files filtered by kmer")
@click.pass_context
class trim:
    """Trim BAM file(s) to only telomere regions"""
    def __init__(self, ctx, i, o, kmer, kset, ref, k):
        self.args = self.read_cl_arg(ctx, i, o)
        ctx.obj["args"] = self.args
        if not os.path.exists(o):
            os.makedirs(o)

        self.use_unmapped = False

        if kmer == True:
            self.coverages = collect_coverage(i, "trim")
            self.coverages = pd.DataFrame.from_dict([self.coverages])
            print(self.coverages)
            self.coverages.to_csv("coverages.csv", index=False)
            if os.path.isfile(ref) == False:
                assert os.path.isfile(ref) == True, "Please provide a valid reference path"
            ## found in cteltool.pyx
            self.read_counts = scan_files(i, o, int(kset.split("_")[-1]),
                os.path.join(str(os.path.dirname(__file__)), "telmer_set", str(kset)),
                ref, k, ctx.obj["global"]["verbose"])

        if kmer == False:
            ## collect coverage before trimming
            self.coverages = collect_coverage(i, "trim")
            self.coverages = pd.DataFrame.from_dict([self.coverages])
            print(self.coverages)
            self.coverages.to_csv("coverages.csv", index=False)
            ## load reference coordinates to trim to
            if ctx.obj["global"]["reference"] == "hg38":
                self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38.tsv"), sep="\t")
            if ctx.obj["global"]["reference"] == "hg38c":
                self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38_compat_hg19.tsv"), sep="\t")
            if ctx.obj["global"]["reference"] == "hg19":
                self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg19.csv"))
            if ctx.obj["global"]["reference"] == "telmer":
                assert ctx.obj["global"]["reference"] != "telmer", "Please specify reference (teltool -ref hg38/hg19 trim --region)"
            ## trim files to reference coordinates
            self.trim_bam(i, o , self.coords)

        self.sort_tmp_files(o)
        self.index_tmp_bam(o)


    def read_cl_arg(self, ctx, i, o):
        args = {"working_directory": ctx.obj["global"]["working_directory"], "input_data": i, "out_dir": o}
        return args

    def trim_bam(self, in_dir, out_dir, coords):

        if self.use_unmapped == True:
            ## previously used for unmapped reads
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

        if os.path.isdir(in_dir) == False:
            print("Trimming:    ", in_dir)
            bamfile = pysam.AlignmentFile(in_dir, "rb")
            tmpfile = pysam.AlignmentFile(os.path.join(out_dir, os.path.basename(in_dir).split(".")[0]+"_tel.bam"), "wb", template=bamfile)
            chr_list = list(set(sorted(coords.chrom)))
            for chromo in chr_list:
                chrom_reg = coords[coords["chrom"] == chromo]
                for i in range(len(chrom_reg["chrom"])):
                    low = int(chrom_reg["chromStart"].iloc[i])
                    up = int(chrom_reg["chromEnd"].iloc[i])
                    for read in bamfile.fetch(chromo, low, up):
                        tmpfile.write(read)

            if self.use_unmapped == True:
                ## unmapped reads (previously used in model)
                bamfile = pysam.AlignmentFile(in_dir, "rb")
                for read in bamfile: #.fetch("chr1", 0, until_eof=True):
                    if read.is_unmapped == True:
                        cf, af, bf, = count_variant_repeats(read.query_alignment_sequence.upper(), targets_f, targets_to_array_f, "forward")
                        cr, ar, br, = count_variant_repeats(read.query_alignment_sequence.upper(), targets_r, targets_to_array_r, "reverse")
                        if cf >= cr:
                            if sum(bf)/read.query_alignment_length > 0.5:
                                tmpfile.write(read)
                        if cr > cf:
                            if sum(br)/read.query_alignment_length > 0.5:
                                tmpfile.write(read)

            bamfile.close()
            tmpfile.close()

        if os.path.isdir(in_dir) == True:
            bam_files = os.listdir(in_dir)
            bam_files = [f for f in bam_files if f.endswith(".bam")]
            total_files = len(bam_files)
            for num, bam_file in enumerate(bam_files):
                print("Trimming:    ", num, "/", total_files, "     (", bam_file,")         ", end="\r")
                bamfile = pysam.AlignmentFile(os.path.join(in_dir, bam_file), "rb")
                tmpfile = pysam.AlignmentFile(os.path.join(out_dir, os.path.basename(bam_file).split(".")[0]+"_tel.bam"), "wb", template=bamfile)
                chr_list = list(set(sorted(coords.chrom)))
                for chromo in chr_list:
                    chrom_reg = coords[coords["chrom"] == chromo]
                    for i in range(len(chrom_reg["chrom"])):
                        low = int(chrom_reg["chromStart"].iloc[i])
                        up = int(chrom_reg["chromEnd"].iloc[i])
                        for read in bamfile.fetch(chromo, low, up):
                            tmpfile.write(read)

                if self.use_unmapped == True:
                    ## unmapped reads (previously used in model)
                    for read in bamfile:#bamfile.fetch("chr1", 0, until_eof=True):
                        if read.is_unmapped == True:
                            cf, af, bf, = count_variant_repeats(read.query_alignment_sequence.upper(), targets_f, targets_to_array_f, "forward")
                            cr, ar, br, = count_variant_repeats(read.query_alignment_sequence.upper(), targets_r, targets_to_array_r, "reverse")
                            if cf >= cr:
                                if sum(bf)/read.query_alignment_length > 0.5:
                                    tmpfile.write(read)
                            if cr > cf:
                                if sum(br)/read.query_alignment_length > 0.5:
                                    tmpfile.write(read)

                bamfile.close()
                tmpfile.close()
            print("")

    def sort_tmp_files(self, out_dir):
        if os.path.isdir(out_dir) == False:
            pysam.sort("-o", out_dir, out_dir)
        if os.path.isdir(out_dir) == True:
            bam_files = os.listdir(out_dir)
            bam_files = [f for f in bam_files if f.endswith(".bam")]
            total_files = len(bam_files)
            for num, bam_file in enumerate(bam_files):
                print("Sorting:     ", num+1, "/", total_files, "     (", bam_file,")         ", end="\r")
                pysam.sort("-o", os.path.join(out_dir, bam_file), os.path.join(out_dir, bam_file))
            print("")

    def index_tmp_bam(self, out_dir):
        bam_files = os.listdir(out_dir)
        bam_files = [f for f in bam_files if f.endswith(".bam")]
        total_files = len(bam_files)
        for num, bam_file in enumerate(bam_files):
            print("Indexing:    ", num+1,"/",total_files, "   (", bam_file,")         ", end="\r")
            pysam.index(os.path.join(out_dir, bam_file))
        print("")



@cli.command(hidden=True)
@click.option("-i", default="fastq_scan",
help="Input bam files")
@click.option("--threads", default=1,
help="Number of threads")
@click.option("-l", "--lengths", default="all_lengths.csv",
help="Input lentghs file")
@click.option("-c", "--coverage", default=None, #"coverages/coverage_adjusted.csv",
help="If trimmed files, whole file coverages required")
@click.option("--paired/--single", is_flag=True, flag_value=True, default=False, show_default=True,
help="Specify format of length file")
@click.option("--sample-cols", default=["sample"]) #["sample"])#default=["normal_db", "tumor_db"], help="Column name(s) for samples (if paired, normal tumour)")
@click.option("--stela-cols", default=["stela"]) #["stela"])#default=["normal_stela", "tumor_stela"], help="Column name(s) for stela (if paired, normal tumour)")
@click.option("--model", type=click.Choice(["ridge", "linear", "ransac", "rfr", "lasso", "elastic", "mlpr", "gbmr",
                                            "neighbors", "LSVC", "SVC", "gp", "tree", "rfc", "mlpc", "ada", "gaus", "quad", "gbmc", "stacking"], case_sensitive=False),
default="gbmc", help="Model type (will determin regression or classification)", show_default=True)
@click.pass_context
class train:
    """Read BAM file(s) and lengths to train model"""
    def __init__(self, ctx, i, threads, lengths, coverage, paired, sample_cols, stela_cols, model):
        if os.path.isdir(i):
            self.bam_files = os.listdir(i)
            self.bam_files = [f for f in self.bam_files if f.endswith(".bam")]
            self.bam_files = sorted(self.bam_files)
        else:
            self.bam_files = [i]

        if coverage == None:
            self.avg_coverages = collect_coverage(i)
        if coverage != None:
            self.avg_coverages_file = pd.read_csv(coverage)
            self.avg_coverages = {}
            for col in self.avg_coverages_file:
                self.avg_coverages[col] = np.median(self.avg_coverages_file[col]/len(self.avg_coverages_file.index))

        if model in ["neighbors", "LSVC", "SVC", "gp", "tree", "rfc", "mlpc", "ada", "gaus", "quad", "gbmc", "stacking"]:
            self.model_type = "classification"
        elif model in ["ridge", "linear", "ransac", "rfr", "lasso", "elastic", "mlpr", "gbmr"]:
            self.model_type = "regression"
        else:
            print("Do not know if this is classifier or regressor")
            return

        if ctx.obj["global"]["reference"] == "telmer":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv"), sep="\t")
            self.mod_build = "telmer"
        elif ctx.obj["global"]["reference"] == "hg38":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38.tsv"), sep="\t")
            self.mod_build = "hg38"
        elif ctx.obj["global"]["reference"] == "hg38c":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38_compat_hg19.tsv"), sep="\t")
            self.mod_build = "hg19"
        elif ctx.obj["global"]["reference"] == "hg19":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg19.csv"))
            self.mod_build = "hg19"
        else:
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv"), sep="\t")
            self.mod_build = "telmer"


        self.table = read_tl_bam(i, ctx, threads, self.bam_files, self.coords, self.avg_coverages, ctx.obj["global"]["chr"])#, self.std_coverages)
        self.table.to_csv("training_data.csv", index=False)
        self.lengths = self.read_lengths_data(lengths, paired, sample_cols, stela_cols)
        #print(self.lengths)
        if paired == False:
            self.all = self.combine_lengths_reads(self.table, self.lengths, paired)
        if paired == True:
            self.lengths_single = self.split_combined_pairs(self.lengths)
            self.all = self.combine_lengths_reads(self.table, self.lengths_single, paired)
            self.all = self.merge_read_pairs(self.lengths, self.all)
        self.pred_cols = self.all.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour"])

        self.all = self.filter_thresh(self.all, self.pred_cols, 1000000, paired, imp=False)
        #print(self.all)
        if self.model_type == "regression":
            mod, all_data, pred_cols, mod_stats = self.train_regression_model(self.all, model, self.pred_cols, paired, self.mod_build)
        if self.model_type == "classification":
            mod, all_data, pred_cols, mod_stats = self.train_classification_model(self.all, model, self.pred_cols, paired, self.mod_build)
        self.plot_results(None, mod, all_data, pred_cols, mod_stats, paired, self.model_type)



    def read_lengths_data(self, lengths, paired, sample_cols, stela_cols):
        lengths = pd.read_csv(lengths)
        #lengths = lengths[list(chain.from_iterable([sample_cols, stela_cols]))]
        if paired == False:
            if sample_cols != "sample":
                lengths = lengths.rename(columns={sample_cols[0]: "sample"})
            if stela_cols != "stela":
                lengths = lengths.rename(columns={stela_cols[0]: "stela"})
                lengths["stela"] = round(lengths["stela"]*1000)
        if paired == True:
            if sample_cols != ["sample_normal", "sample_tumour"]:
                lengths = lengths.rename(columns={sample_cols[0]: "sample_normal", sample_cols[1]: "sample_tumour"})
            if stela_cols != ["stela_normal", "stela_tumour"]:
                lengths = lengths.rename(columns={stela_cols[0]: "stela_normal", stela_cols[1]: "stela_tumour"})
                lengths[["stela_normal", "stela_tumour"]] = round(lengths[["stela_normal", "stela_tumour"]]*1000)
        return lengths

    def filter_thresh(self, data, pred_cols, thresh, paired, imp=False):
        #print("#" * 40)
        print("Filtering lengths by NA and threshold (", thresh,"bp ):")
        #print(data)
        if paired == False:
            total_samples = len(data["sample"])
        if paired == True:
            total_samples = len(data["sample_normal"])
        print("Total samples:           ", total_samples)
        ## For checking whole table
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(data)
        ## Impute all above thresh
        # data[tl_preds] = data[tl_preds].apply(lambda x: np.where(x > thresh, np.nan, x))
        ## Remove all above thresh
        #for c in pred_cols:
        #    data = data[data[c] < thresh]
        ## Clipping method
        #data[tl_preds] = data[tl_preds].clip(upper=thresh)
        if paired == False:
            data = data[data["stela"].notna()]
            data["stela"] = data["stela"].astype(int)
            print("remaining:   ", len(data["sample"]), "/", total_samples, "   (", round((len(data["sample"])/total_samples)*100, 1), "% )")
        if paired == True:
            data = data[data["stela_normal"].notna()]
            data = data[data["stela_tumour"].notna()]
            data[["stela_normal", "stela_tumour"]] == data[["stela_normal", "stela_tumour"]].astype(int)
            print("remaining:   ", len(data["sample_normal"]), "/", total_samples, "    (", round((len(data["sample_normal"])/total_samples)*100, 1), "% )")
        if imp == True:
            simpimp = sklearn.impute.SimpleImputer(strategy="median")
            data[pred_cols] = simpimp.fit_transform(data[pred_cols])
        #print(data)
        return data

    def split_combined_pairs(self, in_data):
        single_sample_data = pd.DataFrame()
        data = in_data.copy()
        data.columns = [s.split("_")[0] for s in data.columns]
        for c in data.columns:
            single_sample_data[c] = data[c].iloc[:,0].append(data[c].iloc[:,1])
        return single_sample_data

    def combine_lengths_reads(self, table, lengths, paired):
        merged = pd.DataFrame.merge(table, lengths, on="sample", how="left")
        ## check data
        with pd.option_context('display.max_rows', None):
            print(merged)
        # merged.to_csv("lets_see2.csv", index=False)
        # fig, ax =joypy.joyplot(merged[["how_pred", "div_pred", "other_pred", "max_pred", "stela"]])
        # plt.show()
        return merged

    def merge_read_pairs(self, pairs, data):
        pairs = pairs[["sample_normal", "sample_tumour"]]
        pairs = pairs.merge(data.add_suffix("_normal"))
        pairs = pairs.merge(data.add_suffix("_tumour"))
        return pairs

    def train_regression_model(self, all_data, model_type, pred_cols, paired, mod_build):
        mod_dict = {
            "lasso":    Lasso(),
            "ridge":    Ridge(),
            "rfr":      RandomForestRegressor(),
            "linear":   LinearRegression(),
            "ransac":   RANSACRegressor(),
            "elastic":  ElasticNet(),
            "mlpr":      MLPRegressor(hidden_layer_sizes=100, activation="identity", solver="lbfgs", learning_rate="adaptive", max_iter=10000000),
            "gbmr":      LGBMRegressor(boosting_type='dart')
        }
        mod = mod_dict[model_type]
        # all_data = all_data.drop("short.coverage", axis=1)

        pred_cols = feature_select(all_data, paired, model_type, pred_type="regression", selected_types= ["recursive", "from_mod"])

        def cross_validate_predict(model, cv_method, all_data, pred_cols, paired=False):
            cv_dict = {
                "loo": LeaveOneOut(),
                "k_fold": 3
            }
            mod_stats = {}
            train = all_data.copy().dropna()
            if paired == False:
                sample_cols = ["sample"]
                pred = ["model_pred"]
                real = ["stela"]
            if paired == True:
                sample_cols = ["sample_normal", "sample_tumour"]
                pred = ["model_pred_normal", "model_pred_tumour"]
                real = ["stela_normal", "stela_tumour"]

            hyperparameter_tuning = False

            param_grid = {'max_depth': [3, 5, 10, 12, 15],
                            'min_samples_split': [2, 5, 10, 12, 15],
                            'n_estimators': [10, 50, 100, 150, 200, 250],
                            #'max_features': ["auto", "log2", "sqrt"],
                            'bootstrap': [True, False]}

            if hyperparameter_tuning == True:
                print("Hyperparameter tuning...")
                grid = GridSearchCV(model, param_grid, refit=True)
                grid.fit(train[pred_cols], np.ravel(train[real]))
                print(grid.best_params_)
                print(grid.best_score_)
                model = grid.best_estimator_
            else:
                model.fit(train[pred_cols], np.ravel(train[real]))

            print("Cross validating model...")
            if paired == False:
                train[pred] = pd.DataFrame(cross_val_predict(model, train[pred_cols], np.ravel(train[real]), cv=cv_dict[cv_method]), columns = pred, index = train.index)
            if paired == True:
                train[pred] = pd.DataFrame(cross_val_predict(model, train[pred_cols], train[real], cv=cv_dict[cv_method]), columns = pred, index = train.index)
            train[pred] = round(train[pred], 0)
            train[pred] = train[pred].astype(int)
            if paired == False:
                train["diff"] = train["model_pred"]-train["stela"]
            if paired == True:
                train["diff_normal"] = train["model_pred_normal"]-train["stela_normal"]
                train["diff_tumour"] = train["model_pred_tumour"]-train["stela_tumour"]

            with pd.option_context('display.max_rows', None):  # more options can be specified also
                print(train)
                #print(train["stela"] - train[pred_cols])

            real_pred = train[list(chain.from_iterable([sample_cols, real, pred]))].dropna()
            preds = real_pred[list(chain.from_iterable([sample_cols, pred]))].copy()
            predictions = np.ravel(real_pred[pred])
            actual = np.ravel(real_pred[real])
            # if paired == False:
            #     actual = np.array(all_data["stela"])
            #     predictions = np.array(train["model_pred"])
            # if paired == True:
            #     actual = np.array(all_data["stela_normal"], all_data["stela_tumour"])
            #     predictions = np.array(train["model_pred_normal"], train["model_pred_tumour"])
            # if paired == False:
            #     mod_stats["Rho"] = stats.pearsonr(predictions[:,0], actual[:,0])
            # if paired == True:
            #     mod_stats["N-Rho"] = stats.pearsonr(predictions[:,0], actual[:,0])
            #     mod_stats["T-Rho"] = stats.pearsonr(predictions[:,1], actual[:,1])

            mod_stats["MAE"] = metrics.mean_absolute_error(actual, predictions)
            mod_stats["Rsq"] = metrics.r2_score(actual, predictions)
            mod_stats["rmse"] = np.sqrt(np.mean((predictions - actual) ** 2))
            mod_stats["preds"] = train[list(chain.from_iterable([sample_cols, pred_cols, real, pred, ["diff"]]))].copy()
            mod_stats["model_params"] = mod.get_params()
            mod_stats["cv_method"] = cv_method

            print(mod_stats)
            save = input("Save model? ")
            if save.lower() in ["y", "yes", "t", "true"]:
                save_name = input("Model name: ")
                dump(model, os.path.join(os.path.dirname(__file__), "models", save_name+"_"+mod_build))
                dump(pred_cols, os.path.join(os.path.dirname(__file__), "models", save_name+"_"+mod_build+".list"))
                dump(mod_stats, os.path.join(os.path.dirname(__file__), "models", save_name+"_"+mod_build+".stats"))
            ## Save trained prediction results
            # train.to_csv("train_predicted.csv", index=False)
            return mod_stats, mod, train

        mod_stats, mod, all_data = cross_validate_predict(mod, "loo", all_data, pred_cols, paired)

        return mod, all_data, pred_cols, mod_stats

    def train_classification_model(self, all_data, model_type, pred_cols, paired, mod_build):
        mod_dict = {
            "neighbors":    KNeighborsClassifier(3),
            "LSVC":         SVC(kernel="linear", C=0.025),
            "SVC":          SVC(gamma=2, C=1),
            "gp":           GaussianProcessClassifier(1.0 * RBF(1.0)),
            "tree":         DecisionTreeClassifier(max_depth=5),
            "rfc":          RandomForestClassifier(max_depth=5, n_estimators=10),#, max_features=1),
            "mlpc":         MLPClassifier(hidden_layer_sizes=500, activation="identity", learning_rate="adaptive", max_iter=25000),
            "ada":          AdaBoostClassifier(),
            "gaus":         GaussianNB(),
            "quad":         QuadraticDiscriminantAnalysis(),
            "gbmc":         LGBMClassifier(boosting_type="dart", max_depth=3), #feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
            "stacking":     StackingClassifier(estimators=[("rfc", RandomForestClassifier(max_depth=5, n_estimators=10)),
                                                           ("tre", DecisionTreeClassifier(max_depth=5),
                                                           ("ada", AdaBoostClassifier()))],
                                               final_estimator=RandomForestClassifier(max_depth=5, n_estimators=10))#MLPClassifier(hidden_layer_sizes=500, activation="identity", learning_rate="adaptive", max_iter=25000))
        }
        mod = mod_dict[model_type]

        def add_short_labels(threshold, data):
            data['short'] = np.where(data['stela'] < threshold , True, False)
            return data

        all_data = add_short_labels(3810, all_data)

        with pd.option_context('display.max_rows', None):
            print(all_data[["sample", "stela", "short"]])

        # if model_type != "stacking":
        #     pred_cols = feature_select(all_data, paired, model_type, pred_type="classification", selected_types= ["recursive", "from_mod"]) #["recursive", "from_mod"])
        # else:
        #     pred_cols = feature_select(all_data, paired, "tree", pred_type="classification", selected_types= ["recursive", "from_mod"])
        #
        # tmp_region_remove = unique_regions = ["f_0", "r_0", "2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0"] #, "short", "mapq0", "mapq1"]
        # tmp_var_remove = []
        # for i in tmp_region_remove:
        #     tmp_var_remove.append(i+".mapq")
        # for i in tmp_region_remove+["short", "mapq0", "mapq1"]:
        #     tmp_var_remove.append(i+".scqc")
        #     tmp_var_remove.append(i+".gc")
        #     tmp_var_remove.append(i+".avg_read_len")
        #     tmp_var_remove.append(i+".fragments")
        #
        #     tmp_var_remove.append(i+".num_reads")
        #     tmp_var_remove.append(i+".length")
        #     tmp_var_remove.append(i+".kc_len")
        #     tmp_var_remove.append(i+".kc_len_cov")
        #     tmp_var_remove.append(i+".kc_perc")
        #     tmp_var_remove.append(i+".new")
        #     tmp_var_remove.append(i+".len_reg")
        #     tmp_var_remove.append(i+".kc_len_reg")
        #
        # for i in ["reads", "length", "avg_read_len", "kc_len"]:
        #     tmp_var_remove.append("total_"+i)
        #
        # for i in ["new_cov"]:
        #     tmp_var_remove.append(i)

        #pred_cols = pred_cols.difference(tmp_var_remove)
        #print(pred_cols)

        pred_cols = []#["total_coverage"]
        for i in ["f_0", "r_0", "rest_0"]:#, "2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0"]: #, "short", "mapq0", "mapq1"]:
            pred_cols.append(i+".perc_cov")
            #pred_cols.append(i+".coverage_sc")
            #all_data[i+".adj_cov"] = all_data[i+".adj_cov"]/all_data["total_coverage"]

        with pd.option_context('display.max_rows', None):
            print(all_data[pred_cols])

        #pred_cols = ['13_0.avg_read_len', '13_0.new', '2_0.kc_len_reg', '2_0.num_reads', '45_0.new', '9_0.kc_len', 'total_reads']
        # pred_cols = all_data.columns.difference(["sample", "sample_normal", "sample_tumour", "stela", "stela_normal", "stela_tumour", "short", "short_normal", "short_tumour", "how_diff"])
        #compare_data = all_data[list(chain.from_iterable([["sample"], pred_cols, ["short"]]))]
        #compare_data.to_csv("bgi_feat.csv")

        def cross_validate_predict(model, cv_method, all_data, pred_cols, paired=False):
            cv_dict = {
                "loo": LeaveOneOut(),
                "k_fold": 3
            }
            mod_stats = {}

            drop_samples = False
            if drop_samples == True:
                df = all_data[all_data["short"] == False]
                drop_indices = np.random.choice(df.index, 42, replace=False)
                train_drop = all_data.drop(drop_indices)
            else:
                train_drop = all_data.copy()

            train = all_data.copy()
            train = train.replace(0, np.nan).dropna(axis=1,how="all")
            train = train.replace([np.nan, np.inf, -np.inf], 0)
            if paired == False:
                sample_cols = ["sample"]
                pred = ["model_pred"]
                real = ["short"]
            if paired == True:
                sample_cols = ["sample_normal", "sample_tumour"]
                pred = ["model_pred_normal", "model_pred_tumour"]
                real = ["short_normal", "short_tumour"]


            hyperparameter_tuning = False

            # param_grid = {'max_depth': [3, 4, 5, 6],
            #                 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            #                 'criterion': ["gini", "entropy"]
            #                 #'n_estimators': [10, 50, 100, 150, 200, 250]#,
            #                 #'max_features': ["auto", "log2", "sqrt"],
            #                 #'bootstrap': [True, False]
            #                 }

            param_grid = {"num_leaves": range(3, 6),
                          "max_depth": range(3, 6),
                          "max_bin": range(30, 80, 1),
                          "extra_trees": [True, False],
                          "path_smooth": [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
                          "n_estimators": range(30, 150, 1),
                          "learning_rate": [0.1, 0.2, 0.3]
                          #"verbose": [-1]#,
                          #"min_data_in_leaf": range(5, 40, 5)#,
                          #"feature_fraction": [0.8, 0.85, 0.9, 0.95, 1]
                          }

            if hyperparameter_tuning == True:
                print("Hyperparameter tuning...")
                grid = RandomizedSearchCV(model, param_grid, refit=True, n_iter=25000, scoring="f1")
                grid.fit(train[pred_cols], np.ravel(train[real]))
                print(grid.best_params_)
                print(grid.best_score_)
                model = grid.best_estimator_
            else:
                model.fit(train_drop[pred_cols], np.ravel(train_drop[real]))

            print("Cross validating model...")
            if paired == False:
                train[pred] = pd.DataFrame(cross_val_predict(model, train[pred_cols], np.ravel(train[real]), cv=cv_dict[cv_method]), columns = pred, index = train.index)
                try:
                    conf = pd.DataFrame(model.predict_proba(train[pred_cols]), columns=["long_conf", "short_conf"], index=train.index)
                    train["long_conf"] = conf["long_conf"]
                    train["short_conf"] = conf["short_conf"]
                    train["short"] =  np.where(train["long_conf"] <= 0.6, True, False)
                except:
                    pass
            if paired == True:
                train[pred] = pd.DataFrame(cross_val_predict(model, train[pred_cols], train[real], cv=cv_dict[cv_method]), columns = pred, index = train.index)

            df = train.copy()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(df[(df['short']==True) & (df['model_pred']==True)]["f_0.perc_cov"], df[(df['short']==True) & (df['model_pred']==True)]["r_0.perc_cov"], df[(df['short']==True) & (df['model_pred']==True)]["rest_0.perc_cov"])
            ax.scatter(df[(df['short']==False) & (df['model_pred']==True)]["f_0.perc_cov"], df[(df['short']==False) & (df['model_pred']==True)]["r_0.perc_cov"], df[(df['short']==False) & (df['model_pred']==True)]["rest_0.perc_cov"])
            ax.scatter(df[(df['short']==False) & (df['model_pred']==False)]["f_0.perc_cov"], df[(df['short']==False) & (df['model_pred']==False)]["r_0.perc_cov"], df[(df['short']==False) & (df['model_pred']==False)]["rest_0.perc_cov"])
            ax.scatter(df[(df['short']==True) & (df['model_pred']==False)]["f_0.perc_cov"], df[(df['short']==True) & (df['model_pred']==False)]["r_0.perc_cov"], df[(df['short']==True) & (df['model_pred']==False)]["rest_0.perc_cov"])
            #ax.scatter(df[df["short"].any(0)]["f_0.perc_cov"], df[df["short"].any(0)]["r_0.perc_cov"], df[df["short"].any(0)]["rest_0.perc_cov"])

            fig.legend(["tp", "fp", "tn", "fn"])

            ax.set_xlabel('f perc')
            ax.set_ylabel('r perc')
            ax.set_zlabel('o perc')

            plt.show()

            # train = train.drop(train.index[0:9])

            try:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                     print(train[["sample", "stela", "short", "model_pred", "long_conf", "short_conf", "f_0.perc_cov", "r_0.perc_cov", "rest_0.perc_cov"]])
            except:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                     print(train[["sample", "stela", "short", "model_pred"]])
            f_score = sklearn.metrics.f1_score(train["short"], train["model_pred"])
            accuracy_score = sklearn.metrics.accuracy_score(train[pred], train["short"])
            print(f"F1-score: {f_score}  (accuracy: {accuracy_score:.2%})")
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(train["short"], train["model_pred"]).ravel()
            print(F"Confusion matrix: (tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp})")
            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            print(f"Precision: {precision}, Recall: {recall}")
            specificity = tn / (tn+fp)
            print(f"Specificity: {specificity}")

            mod_stats["f1"] = f_score
            mod_stats["accuracy"] = accuracy_score
            mod_stats["precision"] = precision
            mod_stats["recall"] = recall
            mod_stats["specificity"] = specificity
            mod_stats["cm"] = (tn, fp, fn, tp)
            if model_type != "stacking":
                mod_stats["model_params"] = mod.get_params()
            mod_stats["cv_method"] = cv_method
            mod_stats["preds"] = train

            save = input("Save model? ")
            if save[0].lower() in ["y", "t"]:
                save_name = input("Model name: ")
                dump(model, os.path.join(os.path.dirname(__file__), "models", save_name+"_"+mod_build))
                dump(pred_cols, os.path.join(os.path.dirname(__file__), "models", save_name+"_"+mod_build+".list"))
                dump(mod_stats, os.path.join(os.path.dirname(__file__), "models", save_name+"_"+mod_build+".stats"))

            return mod_stats, mod, train

        mod_stats, mod, all_data = cross_validate_predict(mod, "loo", all_data, pred_cols, paired)

        return mod, all_data, pred_cols, mod_stats


    def plot_results(self, fig_dir, mod, data, pred_cols, mod_stats, paired, model_type):
        data.to_csv("normed_stats.csv", index=False)
        if hasattr(mod, "feature_importances_"):
            plot_importance = True
        else:
            plot_importance = False
        plot_joyplot = False
        plot_stdcov = False
        plot_correlation_matrix = False
        if model_type == "regression":
            plot_difference = True
            plot_confusion = False
            plot_roc_curve = False
            if paired == False:
                real = ["stela"]
                diff = data["diff"]
                model_pred = data["model_pred"]
                stela = data["stela"]
                #std_cov = data["std_cov"]
            if paired == True:
                real = ["stela_normal", "stela_tumour"]
                diff = np.ravel(data[["diff_normal", "diff_tumour"]])
                model_pred = np.ravel(data[["model_pred_normal", "model_pred_tumour"]])
                stela = np.ravel(data[["stela_normal", "stela_tumour"]])
                std_cov = np.ravel(data[["std_cov_normal", "std_cov_tumour"]])
        if model_type == "classification":
            plot_difference = False
            plot_confusion = True
            if hasattr(mod, "predict_proba"):
                plot_roc_curve = True
            else:
                plot_roc_curve = False
            if paired == False:
                real = ["short"]
                model_pred = data["model_pred"]
                stela = data["short"]
                #std_cov = data["std_cov"]
            if paired == True:
                real = ["short_normal", "short_tumour"]
                model_pred = np.ravel(data[["model_pred_normal", "model_pred_tumour"]])
                stela = np.ravel(data[["short_normal", "short_tumour"]])
                #std_cov = np.ravel(data[["std_cov_normal", "std_cov_tumour"]])

        if plot_importance == True:
            print("Plotting permutation importance...")
            #clf = RandomForestRegressor()
            if paired == False:
                mod.fit(data[pred_cols], np.ravel(data[real]))
            if paired == True:
                mod.fit(data[pred_cols], data[real])
            result = permutation_importance(mod, data[pred_cols], data[real], n_repeats=10,
                            random_state=42)
            perm_sorted_idx = result.importances_mean.argsort()

            tree_importance_sorted_idx = np.argsort(mod.feature_importances_)
            tree_indices = np.arange(0, len(mod.feature_importances_)) + 0.5

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            ax1.barh(tree_indices,
                     mod.feature_importances_[tree_importance_sorted_idx], height=0.7)
            ax1.set_yticklabels(pred_cols[tree_importance_sorted_idx])
            ax1.set_yticks(tree_indices)
            ax1.set_ylim((0, len(mod.feature_importances_)))
            ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                        labels=pred_cols[perm_sorted_idx])
            fig.tight_layout()
            plt.show()


        if plot_difference == True:
            bin_width = 100
            fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)

            ax1.hist(diff, bins = np.arange(min(diff), max(diff)+bin_width, bin_width))
            ax1.set_xlabel("Difference pred-real (bp)")
            top_of_graph1 = ax1.get_ylim()[1]
            ax1.text(max(diff)-500, top_of_graph1-1.5, "MAE {} \nSTD {}".format(round(mod_stats["MAE"]), round(np.std(diff))))

            mod_div_diff = model_pred/stela
            ax2.hist(mod_div_diff, bins = np.arange((min(mod_div_diff)//0.1)/10, max(mod_div_diff)+0.1, 0.1))
            ax2.set_xlabel("Difference pred/real")
            top_of_graph2 = ax2.get_ylim()[1]
            ax2.text(max(mod_div_diff)-0.3, top_of_graph2-1.5, "Mean {} \nSTD  {}".format(round(np.mean(mod_div_diff), 3), round(np.std(mod_div_diff), 3)))
            plt.show()

        if plot_confusion:
            confusion_matrix_out = sklearn.metrics.confusion_matrix(data["short"], data["model_pred"])
            confusion_matrix_dis = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix_out, display_labels=["long", "short"])
            confusion_matrix_dis.plot()
            plt.show()


        if plot_roc_curve == True:
            ns_probs = [0 for _ in range(len(data.index))]
            md_probs = data["short_conf"]
            ns_auc = sklearn.metrics.roc_auc_score(data["model_pred"], ns_probs)
            md_auc = sklearn.metrics.roc_auc_score(data["model_pred"], md_probs)
            print('No Skill: ROC AUC=%.3f' % (ns_auc))
            print('Model: ROC AUC=%.3f' % (md_auc))
            ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(data["model_pred"], ns_probs)
            md_fpr, md_tpr, _ = sklearn.metrics.roc_curve(data["model_pred"], md_probs)
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            plt.plot(md_fpr, md_tpr, marker='.', label='Model')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.show()

        if plot_joyplot == True:
            fix, ax = joypy.joyplot([model_pred, stela])
            plt.show()

        if plot_stdcov == True:
            plt.scatter(data["avg_cov"], abs(diff))
            plt.show()

        if plot_correlation_matrix == True:
            corr = data.corr()
            fig, ax = plt.subplots()
            ax = sns.heatmap(
                corr,
                vmin=-1, vmax=1, center=0,
                cmap=sns.diverging_palette(20,220, n=200),
                square=True
                )
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                horizontalalignment='right')
            plt.show()

            with pd.option_context("display.max_rows", None):
                print(data[data.columns[1:]].corr()["stela"][:])



@cli.command()
@click.option("-i", default="tmp",
help="Input data directory or bam file")
@click.option("-c", "--coverage", default="coverage.csv",
help="If trimmed files, whole file coverages required")
@click.option("--model", type=click.Choice([f.split("_")[0] for f in os.listdir(os.path.join(os.path.dirname(__file__), "models")) if "." not in f], case_sensitive=False),
default="gbmc", help="Model type, determins regression or classification", show_default=True)
@click.option("-o", default="predictions",
help="Output file name")
@click.option("-s", default=None,
help="Shortcut to processed bam spreadsheet")
@click.pass_context
class test:
    """Apply model to sample data for telomere length prediction"""
    def __init__(self, ctx, i, coverage, model, o):
        # test_py_code(self.coords)
        # find_telmers(i, ctx, "fastq", None, None)
        self.bam_files = os.listdir(i)
        self.bam_files = [f for f in self.bam_files if f.endswith(".bam")]
        self.bam_files = sorted(self.bam_files)

        self.comparing = True
        self.plot_gen = True

        self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv"), sep="\t")

        #self.avg_coverages_file = pd.read_csv(coverage)
        self.avg_coverages = {}
        #for col in self.avg_coverages_file:
        #    self.avg_coverages[col] = np.median(self.avg_coverages_file[col]/len(self.avg_coverages_file.index))

        # if model in ["ridge", "linear", "ransac", "rfr", "lasso", "elastic", "mlpr", "gbmr"]:
        #     self.model_type = "regression"
        # if model in ["neighbors", "LSVC", "SVC", "gp", "tree", "rfc", "mlpc", "ada", "gaus", "quad"]:
        #     self.model_type = "classification"

        if ctx.obj["global"]["reference"] == "telmer":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv"), sep="\t")
            self.mod_build = "telmer"
        elif ctx.obj["global"]["reference"] == "hg38":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38.tsv"), sep="\t")
            self.mod_build = "hg38"
        elif ctx.obj["global"]["reference"] == "hg38c":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38_compat_hg19.tsv"), sep="\t")
            self.mod_build = "hg19"
        elif ctx.obj["global"]["reference"] == "hg19":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg19.tsv"), sep="\t")
            self.mod_build = "hg19"
        else:
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv"), sep="\t")
            self.mod_build = "telmer"

        if s == None:
            self.table = read_tl_bam(i, ctx, 1, self.bam_files, self.coords, self.avg_coverages, ctx.obj["global"]["chr"])
        else:
            self.table = pd.read_csv(s)
            self.table.drop(["short", "short_conf", "long_conf"], 1)

        #model = #"correction"#"gbmc"
        # self.mod_build = "telmer"
        self.mod_file = os.path.join(os.path.dirname(__file__), "models", model+"_"+self.mod_build)
        self.model = load(open(self.mod_file, "rb"))
        self.model_params = load(open(self.mod_file+".list", "rb"))
        ## Show trees in model
        # for i in range(255):
        #     tree = lightgbm.plot_tree(self.model, tree_index=i)
        #     plt.show()
        if self.comparing == True:
            self.table["g"] = self.table["sample"].apply(lambda x: np.where(x.startswith("DB"), "b", "i"))
            ### Domain adaptation
            xs = np.array(self.table[self.table["g"] == "i"][["f_0.perc_cov", "r_0.perc_cov", "rest_0.perc_cov"]])
            xt = np.array(self.table[self.table["g"] == "b"][["f_0.perc_cov", "r_0.perc_cov", "rest_0.perc_cov"]])

            ## MappingTransport with linear kernel
            ot_mapping_linear = ot.da.MappingTransport(kernel="linear", mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
            ot_mapping_linear.fit(Xs=xs, Xt=xt)
            ## for original source samples, transform applies barycentric mapping
            transp_Xs_linear = ot_mapping_linear.transform(Xs=xs)
            #print(transp_Xs_linear)
            #print(self.table[self.table["g"] == "i"][["f_0.perc_cov", "r_0.perc_cov", "rest_0.perc_cov"]])
            self.table.loc[self.table["g"] == "i", ["f_0.perc_cov", "r_0.perc_cov", "rest_0.perc_cov"]] = transp_Xs_linear #pd.DataFrame(transp_Xs_linear)
            print(self.table[self.table["g"] == "i"][["f_0.perc_cov", "r_0.perc_cov", "rest_0.perc_cov"]])
            ###

        self.predictions = self.model.predict(self.table[self.model_params])
        self.table["short"] = self.model.predict(self.table[self.model_params])

        self.cov_cols = []#["total_coverage"]
        self.read_cols = []
        for i in ["f_0", "r_0", "2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0"]: #, "short", "mapq0", "mapq1"]:
            self.cov_cols.append(i+".prec_cov")
            self.read_cols.append(i+".num_reads")
            #all_data[i+".adj_cov"] = all_data[i+".adj_cov"]/all_data["total_coverage"]


        if self.plot_gen == True:
            #self.table["g"] = ["i"]*109+["b"]*90#np.where(self.table["sample"][:2] == "DB", True, False)
            plt.scatter(self.table[self.table["g"] == "b"]["f_0.perc_cov"], self.table[self.table["g"] == "b"]["r_0.perc_cov"])
            plt.scatter(self.table[self.table["g"] == "i"]["f_0.perc_cov"], self.table[self.table["g"] == "i"]["r_0.perc_cov"])
            plt.legend(["bgi", "ill"])
            plt.xlabel("forward coverage / strand bias")
            plt.ylabel("reverse coverage / strand bias")
            plt.show()

            plt.scatter(self.table[self.table["g"] == "b"]["f_0.perc_cov"], self.table[self.table["g"] == "b"]["rest_0.perc_cov"])
            plt.scatter(self.table[self.table["g"] == "i"]["f_0.perc_cov"], self.table[self.table["g"] == "i"]["rest_0.perc_cov"])
            plt.legend(["bgi", "ill"])
            plt.xlabel("forward coverage / strand bias")
            plt.ylabel("other coverage / strand bias")
            plt.show()

            plt.scatter(self.table[self.table["g"] == "b"]["r_0.perc_cov"], self.table[self.table["g"] == "b"]["rest_0.perc_cov"])
            plt.scatter(self.table[self.table["g"] == "i"]["r_0.perc_cov"], self.table[self.table["g"] == "i"]["rest_0.perc_cov"])
            plt.legend(["bgi", "ill"])
            plt.xlabel("reverse coverage / strand bias")
            plt.ylabel("other coverage / strand bias")
            plt.show()

            # plt.scatter(self.table[self.table["g"] == "b"]["f_0.coverage_sc"], self.table[self.table["g"] == "b"]["r_0.coverage_sc"])
            # plt.scatter(self.table[self.table["g"] == "i"]["f_0.coverage_sc"], self.table[self.table["g"] == "i"]["r_0.coverage_sc"])
            # plt.legend(["bgi", "ill"])
            # plt.xlabel("forward coverage / strand bias")
            # plt.ylabel("reverse coverage / strand bias")
            # plt.show()
            # for i in ["f_0", "r_0", "2_0", "4_0", "9_0", "11_0", "13_0", "44_0", "45_0"]:
            #     plt.hist(self.table[self.table["g"] == "b"][f"{i}.adj_cov"], alpha=0.5)
            #     plt.hist(self.table[self.table["g"] == "i"][f"{i}.adj_cov"], alpha=0.5)
            #     plt.show()
            # plt.hist(self.table[self.table["g"] == "b"]["r_0.adj_cov"], alpha=0.5)
            # plt.hist(self.table[self.table["g"] == "i"]["r_0.adj_cov"], alpha=0.5)
            # plt.show()

        # with pd.option_context('display.max_rows', None):
        #     pd.set_option('display.max_columns', None)
        #     print(self.table[["sample"]+self.cov_cols])
        #     print(self.table[["sample"]+self.read_cols])

        try:
            self.confidence = pd.DataFrame(self.model.predict_proba(self.table[self.model_params]), columns=["long_conf", "short_conf"], index=self.table.index)
            self.table["long_conf"] = self.confidence["long_conf"]
            self.table["short_conf"] = self.confidence["short_conf"]
            self.table["short"] = self.table["long_conf"].apply(lambda x: np.where(x <= 0.6, True, False))
            print(len(self.table[self.table["g"] == "i"]["short"]))
            ## show results in terminal
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.table[["sample", "short", "g", "long_conf", "short_conf"]])
        except:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.table[["sample", "short"]])

        self.table.to_csv(o+".csv")


        if self.plot_gen == True:
            df = pd.DataFrame(self.table.copy())

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(df[(df['g']=="i" and df['short'].bool())]["f_0.perc_cov"], df[(df['g']=="i" and df['short'].bool())]["r_0.perc_cov"], df[(df['g']=="i" and df['short'].bool())]["rest_0.perc_cov"])
            ax.scatter(df[(df['g']=="i" and not df['short'].bool())]["f_0.perc_cov"], df[(df['g']=="i" and not df['short'].bool())]["r_0.perc_cov"], df[(df['g']=="i" and not df['short'].bool())]["rest_0.perc_cov"])
            ax.scatter(df[(df['g']=="b" and df['short'].bool())]["f_0.perc_cov"], df[(df['g']=="b" and df['short'].bool())]["r_0.perc_cov"], df[(df['g']=="b" and df['short'].bool())]["rest_0.perc_cov"])
            ax.scatter(df[(df['g']=="b" and not df['short'].bool())]["f_0.perc_cov"], df[(df['g']=="b" and not df['short'].bool())]["r_0.perc_cov"], df[(df['g']=="b" and not df['short'].bool())]["rest_0.perc_cov"])

            fig.legend(["ill short", "ill long", "bgi short", "bgi long"])

            ax.set_xlabel('f perc')
            ax.set_ylabel('r perc')
            ax.set_zlabel('o perc')

            plt.show()



@cli.command()
@click.option("-i", default=os.getcwd(),
help="Input data directory or bam file")
@click.option("-c", "--coverage", default=None,
help="If trimmed files, whole file coverages required")
@click.option("--model", type=click.Choice([f.split("_")[0] for f in os.listdir(os.path.join(os.path.dirname(__file__), "models", "fast")) if "." not in f], case_sensitive=False),
default="gbmc", help="Model type, determins regression or classification", show_default=True)
@click.option("-o", default="fast_predictions",
help="Output file name")
@click.pass_context
class fast:
    """If aligned to hg38 a faster region based method can be used (not recommended)"""
    def __init__(self, ctx, i, coverage, model, o):
        threads = 1
        print(ctx)
        #print([f for f in os.listdir(os.path.join(os.path.dirname(__file__), "models")) if not f.endswith(".list")])
        if os.path.isdir(i):
            self.bam_files = os.listdir(i)
            self.bam_files = [f for f in self.bam_files if f.endswith(".bam")]
            self.bam_files = sorted(self.bam_files)
        else:
            self.bam_files = [i]
        if coverage == None:
            self.avg_coverages = collect_coverage(i)
        if coverage != None:
            self.avg_coverages_file = pd.read_csv(coverage)
            self.avg_coverages = {}
            for col in self.avg_coverages_file:
                self.avg_coverages[col] = np.median(self.avg_coverages_file[col]/len(self.avg_coverages_file.index))

        if model in ["ridge", "linear", "ransac", "rfr", "lasso", "elastic", "mlpr", "gbmr"]:
            self.model_type = "regression"
        if model in ["neighbors", "LSVC", "SVC", "gp", "tree", "rfc", "mlpc", "ada", "gaus", "quad"]:
            self.model_type = "classification"

        if ctx.obj["global"]["reference"] == "hg38":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38.tsv"), sep="\t")
            self.mod_build = "hg38"
        if ctx.obj["global"]["reference"] == "hg38c":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg38_compat_hg19.tsv"), sep="\t")
            self.mod_build = "hg19"
        if ctx.obj["global"]["reference"] == "hg19":
            self.coords = pd.read_csv(os.path.join(os.path.dirname(__file__), "telomere_regions", "fast", "hg19.tsv"), sep="\t")
            self.mod_build = "hg19"
        self.table = read_tl_bam(i, ctx, threads, self.bam_files, self.coords, self.avg_coverages, ctx.obj["global"]["chr"])


        self.mod_file = os.path.join(os.path.dirname(__file__), "models", "fast", model+"_"+self.mod_build)
        self.model = load(open(self.mod_file, "rb"))
        self.model_params = load(open(self.mod_file+".list", "rb"))
        ## Show trees in model
        # for i in range(255):
        #     tree = lightgbm.plot_tree(self.model, tree_index=i)
        #     plt.show()
        self.predictions = self.model.predict(self.table[self.model_params])
        self.table["short"] = self.model.predict(self.table[self.model_params])
        self.confidence = pd.DataFrame(self.model.predict_proba(self.table[self.model_params]), columns=["long_conf", "short_conf"], index=self.table.index)
        self.table["long_conf"] = self.confidence["long_conf"]
        self.table["short_conf"] = self.confidence["short_conf"]
        ## show results in terminal
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.table[["sample", "short", "long_conf", "short_conf"]])

        self.table.to_csv(o+".csv")



## HIDDEN OPTIONS (NOT NEEDED)

@cli.command(hidden=True)
@click.option("-r", "--ref", default=str(os.path.join(os.path.dirname(__file__), "reference", "hg38_cutout_edit.fa")),
help="Reference file")
@click.option("-c", "--coords", default=str(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv")),
help="Coordinates to take kmers from reference")
@click.option("-l", "--k-len", default=32,
help="Kmer length (must be evem and max 32)")
@click.option("-o", "--out", default="telmers_rolling_t2t_32",
help="Set name")
@click.pass_context
class dev:
    """For testing"""
    def __init__(self, ctx, ref, coords, k_len, out):
        self.coords = pd.read_csv(coords, sep="\t")
        coords_list = convert_coords_tuple(self.coords)
        # dict_adjacent("/home/alex/Desktop/uni/PhD/TL_prediction/fastq_scan/hap1/", "coverages/coverage_adjusted.csv", "all_lengths.csv")
        # py_collect_wanted_kmers("/home/alex/Desktop/uni/PhD/TL_prediction/reference_genomes/hg38.fa", coords_list, 32)
        # time_scan_files(["/home/alex/Desktop/uni/PhD/TL_prediction/raw_data/full/DB143.bam"], #[f for f in os.listdir("/home/alex/Desktop/uni/PhD/TL_prediction/tmp") if f.endswith(".bam")],
        # k_len,
        # "/home/alex/Desktop/uni/PhD/TL_prediction/teltool/telmer_set/python_kmers_32.set")


@cli.command(hidden=True)
@click.option("-r", "--ref", default=str(os.path.join(os.path.dirname(__file__), "reference", "hg38_cutout_edit.fa")),
help="Reference file")
@click.option("-c", "--coords", default=str(os.path.join(os.path.dirname(__file__), "telomere_regions", "hg38_cutout_edit.tsv")),
help="Coordinates to take kmers from reference")
@click.option("-l", "--k-len", default=32,
help="Kmer length (must be even and max 32)")
@click.option("-o", "--out", default="telmers_rolling_t2t_32.set",
help="Output set name")
@click.pass_context
class colset:
    """Create kmer set for filtering reads from file (already done)"""
    def __init__(self, ctx, ref, coords, k_len, out):
        assert k_len <= 32, "Max len value is 32"
        assert k_len % 2 == 0, "k-len must be even"
        if os.path.splitext(coords)[-1].lower() == ".csv":
            self.coords = pd.read_csv(coords)
        if os.path.splitext(coords)[-1].lower() == ".tsv":
            self.coords = pd.read_csv(coords, sep="\t")
        print(os.path.splitext(coords)[-1])
        self.coords_list = convert_coords_tuple(self.coords)
        collect_wanted_kmers(ref, self.coords_list, k_len, f"{out}_{k_len}")
