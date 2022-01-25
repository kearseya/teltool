import pysam
import glob
import time
from collections import defaultdict
import pickle

# arbitrary
kmerlength = 43

def get_bam_list():
    return glob.glob("/home/alex/Desktop/uni/PhD/TL_prediction/raw_data/full/DB143.bam")

def get_regions():
    return [("chr1", 10000, 10468), ("chr1", 180096, 180991), ("chr1", 248945546, 248946422)]

def is_valid_primary(a):
    if not a.flag & 3328 and a.flag & 1:  # not primary, duplicate, supplementary alignment and read-paired
        return True

def get_kmers_of_interest(kmer_len=27):
    # First get a collection of kmers of interest from telomere regions of interest
    bams = get_bam_list()
    regions = get_regions()
    print(bams)
    t0 = time.time()
    c = defaultdict(int)
    for pth in bams:
        print(pth)
        bam = pysam.AlignmentFile(pth, "rb")
        tot_reads = 0
        for region in regions:
            for a in bam.fetch(*region):
                if is_valid_primary(a):
                    tot_reads += 1
                seq = a.seq
                kmers = [seq[i:i+kmer_len] for i in range(len(seq) - kmer_len + 1)]
                for k in kmers:
                    if "N" not in k:
                        c[k] += 1
                # tel_kmers.update(kmers)
        print("Input reads", tot_reads)
    # drop high frequency kmers that dont have telomere repeats
    counts = sorted(c.items(), key=lambda x: x[1], reverse=True)
    sm = sum(c.values())
    freq = [(i, j / sm) for i, j in counts]
    print(freq[:100])
    # add to filter
    tel_kmers_set = set([])
    tot = 0
    for k, v in freq:
        if v > 0.00001:
            if "TTAGGG" not in k and "CCCTAA" not in k:
                continue
        tel_kmers_set.add(k)
        tot += 1
    # todo: add in reference kmers here for hg19 / hg38 regions
    print("done", tot, time.time() - t0)
    pickle.dump(tel_kmers_set, open("tel_kmers_set.pkl", "wb"))

#get_kmers_of_interest(kmerlength)

def proc_read_pari(r1, r2, kmer_filter, kmer_len=27):
    if r1 and r2:
        seq = r1.seq
        for i in range(len(seq) - kmer_len + 1):
            if seq[i:i + kmer_len] in kmer_filter:
                return True
        seq = r2.seq
        for i in range(len(seq) - kmer_len + 1):
            if seq[i:i + kmer_len] in kmer_filter:
                return True
    return False

def apply_filter():
    tel_kmers = pickle.load(open("tel_kmers_set.pkl", "rb"))
    # now apply to 'unseen' file (we will use the same one)
    bams = get_bam_list()
    t0 = time.time()
    read_pairs = dict()
    for pth in bams:
        bam = pysam.AlignmentFile(pth, "rb")
        found = 0
        total_seen = 0
        # test chromosome 1 for now
        for a in bam.fetch("chr1", until_eof=True):
            if is_valid_primary(a):
                total_seen += 1
                if a.qname not in read_pairs:
                    read_pairs[a.qname] = a
                else:
                    b = read_pairs[a.qname]
                    del read_pairs[a.qname]
                    if proc_read_pari(a, b, tel_kmers, kmerlength):
                        found += 2
                        # write read pair to fastq file for remapping here
        print("Reads found", found)
        print("total seen", total_seen, "ratio", found / (total_seen + 1e-6))
        # chromosome 1 took 1175
        # skipping kmer counting chr1 was read in 85 seconds! so lots of room for improvement
        # done 100918 0.16112804412841797
        # Reads found 48156
        # total seen 53329632 ratio 0.0009029876673441137
        # 1175.398910999298
    print(time.time() - t0)

apply_filter()
