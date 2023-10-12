#cython: language_level=3
from libcpp.vector cimport vector as cpp_vector
from libcpp.deque cimport deque as cpp_deque
from libcpp.pair cimport pair as cpp_pair
from libcpp.utility cimport pair
from libc.stdint cimport uint32_t, uint8_t, uint64_t, uint16_t, int32_t, int8_t
from libc.stdlib cimport malloc
from libc.string cimport strcpy, strlen
# pysam imports
from pysam.libcalignmentfile cimport AlignmentFile
from pysam.libcalignedsegment cimport AlignedSegment
from pysam.libchtslib cimport bam_get_seq, bam1_t
from pysam.libcsamfile cimport pysam_bam_get_seq

from pysam.libcfaidx cimport FastaFile
from pysam.libcfaidx cimport FastxFile
import pysam

import os
from pathlib import Path
from joblib import dump, load
import time
import datetime
from Bio.Seq import Seq

import mappy
import progressbar
import subprocess

# for conerting reference to nibble array
import numpy as np
import array
from cython.operator cimport dereference
####from teltool import rollingutils

#debugging
from libc.stdio cimport printf

from cteltool cimport hash as xxhasher
from cteltool cimport unordered_set as robin_set



cdef extern from "strings.h" nogil:
    # Returns the index, counting from 0, of the least significant set bit in `x`.
    # https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
    uint8_t ffs(uint8_t x)

cdef inline uint8_t ffs2(uint8_t x) nogil:
    return ffs(x) - 1


cdef uint8_t[16] nib = [0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]

## Takes pointer to binary sequence and performs rolling hash until in set or end of sequence
cdef bint rolling_nibble_hash_ptr(uint8_t* t, int kmer_length, int arr_len, robin_set[uint64_t]& kmers):
    cdef int i = 0
    cdef uint64_t h = 0
    cdef uint64_t mask
    cdef uint8_t v, first_2bit, second_2bit
    with nogil:
        mask = ~(h & 0) >> (64 - (kmer_length * 2))
        while i < kmer_length / 2:
            v = dereference(t)
            second_2bit = nib[v & 15]
            first_2bit = nib[(v >> 4) & 15]
            h = h << 2 | first_2bit
            h = h << 2 | second_2bit
            i += 1
            t += 1
        if kmers.find(h) != kmers.end():
            return 1
        while i < arr_len:
            v = dereference(t)
            second_2bit = nib[v & 15]
            first_2bit = nib[(v >> 4) & 15]
            h = (h << 2 | first_2bit) & mask
            if kmers.find(h) != kmers.end():
                return 1
            h = (h << 2 | second_2bit) & mask
            if kmers.find(h) != kmers.end():
                return 1
            i += 1
            t += 1
    return 0

## Same as above but takes nibble array as input and outputs array of hashed kmers
cpdef uint64_t[:] rolling_nibble_hash(uint8_t[::1] t, int kmer_length, int seq_len):
    # t is a nibble array
    # kmer_length max is 32 (for packing into int64)
    # n hashes to calculate. initialize first is faster than appending
    cdef uint64_t[:] a
    cdef int bases_remaining = seq_len
    if seq_len % 2 == 0:
        a = np.zeros(len(t) * 2 - kmer_length + 1, dtype="uint64")
    else:
        a = np.zeros(len(t) * 2 - kmer_length, dtype="uint64")
    cdef int i = 0
    cdef uint64_t h = 0
    # https://stackoverflow.com/questions/15816927/bit-manipulation-clearing-range-of-bits
    cdef uint64_t mask = ~(h & 0) >> (64 - (kmer_length * 2))
    cdef uint8_t v, first_2bit, second_2bit
    cdef int index = 1
    # with nogil:
    # fill the first block
    cdef int end = <int>(kmer_length / 2)
    while i < end:
        v = t[i]
        # not sure which is faster, maybe array indexing has slight advantage
        # second_2bit = ffs2(v & 15)  # nib[v & 15]
        # first_2bit = ffs2((v >> 4) & 15)  # nib[(v >> 4) & 15]
        second_2bit = nib[v & 15]
        first_2bit = nib[(v >> 4) & 15]
        h = h << 2 | first_2bit
        h = h << 2 | second_2bit
        i += 1
        bases_remaining -= 2
    a[0] = h
    # process the rest of the sequence
    cdef int len_t = len(t)
    while i < len_t:
        v = t[i]
        # these are encoded as nibbles
        # second = v & 15
        # first = (v >> 4) & 15
        # this converts from nibble to 2bit representation
        # could also index the nibble_2int array which might be faster using cython?
        # second_2bit = ffs2(v & 15)  # nib[v & 15]
        # first_2bit = ffs2((v >> 4) & 15)  # nib[(v >> 4) & 15]
        first_2bit = nib[(v >> 4) & 15]
        second_2bit = nib[v & 15]
        if bases_remaining == 1:
            h = (h << 2 | second_2bit) & mask
            a[index] = h
            index += 1
            break
        # break this into three steps:
        # h << 2 makes space for the next 2bit
        # h | first_2bit adds the base into the hash
        # h & mask  drops any bits outside of the kmer length, prevents overflow of the int
        h = (h << 2 | first_2bit) & mask
        a[index] = h
        index += 1
        bases_remaining -= 1
        h = (h << 2 | second_2bit) & mask
        a[index] = h
        index += 1
        bases_remaining -=1
        i += 1
    return a


## xxhasher method
cdef hash_region_binary(s, l):
    cdef const unsigned char* sub_ptr = s
    cdef uint64_t hx2
    cdef int i
    reg_kmer_set = set()
    for i in range(len(s)*2-l+1):
        hx2 = xxhasher(sub_ptr, l, 42)
        reg_kmer_set.add(hx2)
        sub_ptr += 1
    return reg_kmer_set

cdef hash_region_ascii(s, l):
    cdef bytes s_bytes = bytes(s.encode("ascii"))  # s is a python string
    cdef const unsigned char* sub_ptr = s_bytes
    cdef uint64_t hx2
    cdef int i
    reg_kmer_set = set()
    for i in range(len(s)-l+1):
        hx2 = xxhasher(sub_ptr, l, 42)
        reg_kmer_set.add(hx2)
        sub_ptr += 1
    return reg_kmer_set


def basemap():
    return np.array(['.', 'A', 'C', '.', 'G', '.', '.', '.', 'T', '.', '.', '.', '.', '.', 'N'])


def basemap_2_int():
    return {k: i for i, k in enumerate(basemap())}

def char_to_nibble_array(t):
    # encode as nibble
    test_nibble = array.array("B", [])
    t_itr = iter(t)
    for base1 in t_itr:
        v = 0
        v = v << 4 | base1
        try:
            base2 = next(t_itr)
        except StopIteration:
            pass
        else:
            v = v << 4 | base2
        # print("input", base1, bin(base1), base2, bin(base2), v, bin(v))
        # print("out", v & 15, (v >> 4) & 15)
        test_nibble.append(v)
        # A = 0001, T = 1000
        # print(base1, base2, v, bin(v))

    return np.array(test_nibble, dtype="uint8")


def seq_2_nibbles(seq):
    basemap_2_int_d = basemap_2_int()
    input_int_nibbles = [basemap_2_int_d[i] for i in seq]
    return char_to_nibble_array(input_int_nibbles)



## For collecting hashes of wanted kmers
def collect_wanted_kmers(referece, coords, l, out, k_type="binary"):
    cdef FastaFile file = FastaFile(referece)
    kmer_set = set()
    for r in coords:
        region = file.fetch(r[0], r[1], r[2]).upper()
        rev_region = str(Seq(region).reverse_complement())
        if k_type == "binary":
            f_nibble_array = seq_2_nibbles(region)#
            r_nibble_array = seq_2_nibbles(rev_region)
            ## xxhasher
            # reg_kmer_set = hash_region_binary(bytes(f_nibble_array), l)
            # reg_kmer_set_reverse = hash_region_binary(bytes(r_nibble_array), l)
            ## rolling hash method
            reg_kmer_set = rolling_nibble_hash(f_nibble_array, l, len(region))
            reg_kmer_set_reverse = rolling_nibble_hash(r_nibble_array, l, len(region))

        ## for speed testing
        if k_type == "ascii":
            reg_kmer_set = hash_region_ascii(region, l)
            reg_kmer_set_reverse = hash_region_ascii(rev_region, l)
            for i in range(len(region)-l+1):
                reg_kmer_set.add(region[i:i+l])
                reg_kmer_set_reverse.add(rev_region[i:i+l])

        for k in reg_kmer_set:
            kmer_set.add(k)
        for k in reg_kmer_set_reverse:
            kmer_set.add(k)
    print(f"KMERS IN SET (K={l}): {len(kmer_set)}")
    dump(kmer_set, os.path.join(os.path.dirname(__file__), "telmer_set", out))


def run_alignment(file_directory, file_prefix, reference, keep_fq=False):
    ## align command to run
    align_command = f"minimap2 -ax sr {reference} {file_prefix}_tel1.fq {file_prefix}_tel2.fq -o {file_prefix}_tel.sam"
    ## run command (outputs aligned SAM file)
    align = subprocess.run(align_command.split(), cwd=file_directory)
    ## remove fastq files if not wanted (trim -k)
    if keep_fq == False:
      os.remove(os.path.join(file_directory, file_prefix+"_tel1.fq"))
      os.remove(os.path.join(file_directory, file_prefix+"_tel2.fq"))
    ## convert to bam file
    con_command = f"samtools view -hbS {file_prefix}_tel.sam -o {file_prefix}_tel.bam"
    con = subprocess.run(con_command.split(), cwd=file_directory)
    # pysam.view("-hbS", os.path.join(file_directory, file_prefix+".sam"), "-o", os.path.join(file_directory, file_prefix+".bam"))
    ## remove sam file
    os.remove(os.path.join(file_directory, file_prefix+"_tel.sam"))


def scan_files(file_dir, out_dir, l, telmers, referece, keep_fq, verbose, name):
    cdef robin_set[uint64_t] kmers
    kmer_set = load(telmers)
    if os.path.isdir(file_dir) == True:
        files = sorted([os.path.join(file_dir, f) for f in os.listdir(file_dir) if os.path.splitext(f)[-1].lower() in [".bam", ".cram", ".fq", ".fastq"]])
    if os.path.isdir(file_dir) == False:
        files = [file_dir]
    print(f"Scanning files for telmers ({len(kmer_set)} in set k={l})...")
    print("Total files: ", len(files))
    reads = {}
    for h in kmer_set:
        kmers.insert(h)
    for i, f in enumerate(files):
        print(os.path.basename(f), "(",  i+1, "/", len(files), ")")
        t0 = time.time()
        if f != "-":
            if f.split(".")[-1] == "bam":
                if verbose == True:
                    reads[f] = scan_bam_bar(f, out_dir, l, kmers, referece, keep_fq)
                if verbose == False:
                    reads[f] = scan_bam_no_bar(f, out_dir, l, kmers, referece, keep_fq, "rb", name)
            if f.split(".")[-1] == "cram":
                reads[f] = scan_cram(f, out_dir, l, kmers, referece, keep_fq)
            if f.split(".")[-1].lower() in ["fq", "fastq"]:
                scan_fastx(f, l, kmers)
            print(f"TIME TAKEN ({os.path.basename(f)} {round(os.path.getsize(f)/(1024**3), 2)}GB): {str(datetime.timedelta(seconds=round(time.time()-t0)))}    (matched: {reads[f][1]}/{reads[f][0]} reads)")
            # print(os.path.basename(f)," MATCHED: ", reads[f][1], "/", reads[f][0], "(", "%.4f"%(reads[f][1]/reads[f][0]), "% )")
        else:
            reads[f] = scan_bam_no_bar(f, out_dir, l, kmers, referece, keep_fq, "r", name)
    return reads

# for progress bar on bam file
def get_n_reads(fn):
    f = AlignmentFile(fn)
    s = f.get_index_statistics()
    total = 0
    for i in s:
        total += i.mapped
        total += i.unmapped
    total += f.count(contig="*")
    return total

## with progress bar
cdef scan_bam_bar(in_file, out_dir, l, robin_set[uint64_t]& kmers, reference, keep_fq):
    cdef AlignmentFile file = AlignmentFile(in_file, "rb")
    cdef AlignedSegment a
    cdef AlignedSegment b
    cdef int total_reads = 0
    cdef int match_reads = 0
    cdef int n_reads = int(get_n_reads(in_file)*1.01)
    read_pairs = dict()
    basename = os.path.basename(in_file)
    fastq_prefix = os.path.splitext(basename)[0]
    first_file = os.path.join(out_dir, fastq_prefix+"_tel1.fq")
    second_file = os.path.join(out_dir, fastq_prefix+"_tel2.fq")
    reference_check = mappy.Aligner(os.path.join(os.path.dirname(__file__), "reference", "hg38_cutout_edit.fa"), preset="sr")
    with progressbar.ProgressBar(max_value = n_reads, redirect_stdout=True) as bar:
        for a in file:
            total_reads += 1
            bar.update(total_reads)
            if not a.flag & 3328 and a.flag & 1:
                if a.qname not in read_pairs:
                    read_pairs[a.qname] = a
                else:
                    b = read_pairs[a.qname]
                    del read_pairs[a.qname]
                    if scan_read_pair(a, b, l, kmers):
                        align = reference_check.map(a.query_sequence, b.query_sequence)
                        for hit in align:
                            match_reads += 2
                            with open(first_file, "a") as paired1:
                                paired1.write(f"@{a.qname}\n{a.query_sequence}\n+\n{''.join([chr(i+33) for i in a.query_qualities])}\n")
                            with open(second_file, "a") as paired2:
                                paired2.write(f"@{b.qname}\n{b.query_sequence}\n+\n{''.join([chr(i+33) for i in b.query_qualities])}\n")
                            break
    if reference != None:
        run_alignment(out_dir, fastq_prefix, reference, keep_fq)
    return (total_reads, match_reads)

## no progress bar
cdef scan_bam_no_bar(in_file, out_dir, l, robin_set[uint64_t]& kmers, reference, keep_fq, kind, name):
    cdef AlignmentFile file = AlignmentFile(in_file, kind)
    cdef AlignedSegment a
    cdef AlignedSegment b
    cdef int total_reads = 0
    cdef int match_reads = 0
    read_pairs = dict()
    basename = os.path.basename(in_file)
    if in_file != "-":
        fastq_prefix = os.path.splitext(basename)[0]
    else:
        fastq_prefix = name
    first_file = os.path.join(out_dir, fastq_prefix+"_tel1.fq")
    second_file = os.path.join(out_dir, fastq_prefix+"_tel2.fq")
    reference_check = mappy.Aligner(os.path.join(os.path.dirname(__file__), "reference", "hg38_cutout_edit.fa"), preset="sr")
    for a in file:
        total_reads += 1
        if not a.flag & 3328 and a.flag & 1:
            if a.qname not in read_pairs:
                read_pairs[a.qname] = a
            else:
                b = read_pairs[a.qname]
                del read_pairs[a.qname]
                if scan_read_pair(a, b, l, kmers):
                    align = reference_check.map(a.query_sequence, b.query_sequence)
                    for hit in align:
                        match_reads += 2
                        with open(first_file, "a") as paired1:
                            paired1.write(f"@{a.qname}\n{a.query_sequence}\n+\n{''.join([chr(i+33) for i in a.query_qualities])}\n")
                        with open(second_file, "a") as paired2:
                            paired2.write(f"@{b.qname}\n{b.query_sequence}\n+\n{''.join([chr(i+33) for i in b.query_qualities])}\n")
                        break
    if reference != None:
        run_alignment(out_dir, fastq_prefix, reference, keep_fq)
    return (total_reads, match_reads)

## same as scan_bam without progressbar
cdef scan_cram(in_file, out_dir, l, robin_set[uint64_t]& kmers, reference, keep_fq):
    cdef AlignmentFile file = AlignmentFile(in_file, "rc")
    cdef AlignedSegment a
    cdef AlignedSegment b
    cdef int total_reads = 0
    cdef int match_reads = 0
    read_pairs = dict()
    basename = os.path.basename(in_file)
    fastq_prefix = os.path.splitext(basename)[0]
    first_file = os.path.join(out_dir, fastq_prefix+"_tel1.fq")
    second_file = os.path.join(out_dir, fastq_prefix+"_tel2.fq")
    reference_check = mappy.Aligner(os.path.join(os.path.dirname(__file__), "reference", "hg38_cutout_edit.fa"), preset="sr")
    for a in file:
        if not a.flag & 3328 and a.flag & 1:
            total_reads += 1
            if a.qname not in read_pairs:
                read_pairs[a.qname] = a
            else:
                b = read_pairs[a.qname]
                del read_pairs[a.qname]
                if scan_read_pair(a, b, l, kmers):
                    align = reference_check.map(a.query_sequence, b.query_sequence)
                    for hit in align:
                        match_reads += 2
                        with open(first_file, "a") as paired1:
                            paired1.write(f"@{a.qname}\n{a.query_sequence}\n+\n{''.join([chr(i+33) for i in a.query_qualities])}\n")
                        with open(second_file, "a") as paired2:
                            paired2.write(f"@{b.qname}\n{b.query_sequence}\n+\n{''.join([chr(i+33) for i in b.query_qualities])}\n")
                        break
    if reference != None:
        run_alignment(out_dir, fastq_prefix, reference, keep_fq)
    return (total_reads, match_reads)


cdef scan_fastx(in_file, int l, robin_set[uint64_t]& kmers):
    cdef FastxFile file = FastxFile(in_file)
    out_file = in_file.split(".")[0]+"_tel"
    for r in file:
        if scan_read(r.query_alignment_sequence, l, kmers):
            with open(out_file+".fq", "a") as filtered_fastq:
                filtered_fastq.write(f"@{r.name}\n{r.sequence}\n+\n{r.quality}\n")

# needs fixing to rolling hash + align
cdef bint scan_read(str s, int l, robin_set[uint64_t]& kmers):
    cdef bytes s_bytes = bytes(s.upper().encode("ascii"))  # s is a python string
    cdef const unsigned char* sub_ptr = s_bytes
    cdef uint64_t hx2
    cdef int i
    for i in range(len(s)-l+1):
        hx2 = xxhasher(sub_ptr, l, 42) # m is the kmer length, 42 is a seed
        sub_ptr += 1
        if kmers.find(hx2) != kmers.end():
            return True
    return False

cdef bint scan_read_pair(AlignedSegment r1, AlignedSegment r2, int l, robin_set[uint64_t]& kmers):
    ## Rolling hash method
    cdef uint8_t* uint_ptr_rseq1 = pysam_bam_get_seq(r1._delegate)
    cdef bam1_t * src1 = r1._delegate

    if rolling_nibble_hash_ptr(uint_ptr_rseq1, l, src1.core.l_qseq, kmers):
        return 1

    cdef uint8_t* uint_ptr_rseq2 = pysam_bam_get_seq(r2._delegate)
    cdef bam1_t * src2 = r2._delegate

    if rolling_nibble_hash_ptr(uint_ptr_rseq2, l, src2.core.l_qseq, kmers):
        return 1

    ## xxhasher method
    # cdef bam1_t * src1 = r1._delegate
    # cdef bam1_t * src2 = r2._delegate
    # cdef char* char_ptr_rseq1 = <char*>bam_get_seq(r1._delegate)
    # cdef char* char_ptr_rseq2 = <char*>bam_get_seq(r2._delegate)
    # cdef uint64_t hx2
    # cdef int i
    # for i in range(src1.core.l_qseq-l+1):
    #     hx2 = xxhasher(char_ptr_rseq1, l, 42) # m is the kmer length, 42 is a seed
    #     char_ptr_rseq1 += 1
    #     if kmers.find(hx2) != kmers.end():
    #         return 1
    # for i in range(src2.core.l_qseq-l+1):
    #     hx2 = xxhasher(char_ptr_rseq2, l, 42)
    #     char_ptr_rseq2 += 1
    #     if kmers.find(hx2) != kmers.end():
    #         return 1

    return 0



def test_bam(in_file):
    basemap = np.array([ '.', 'A', 'C', '.', 'G', '.', '.', '.', 'T', '.', '.', '.', '.', '.', 'N'])
    reverse_basemap = {b: i for i, b in enumerate(basemap)}
    nibble_array = array.array("B", [])
    region = "A"*33
    print(region)
    print(len(region))
    nibble_array = seq_2_nibbles(region)
    reg = rolling_nibble_hash(nibble_array, 32, len(region))
    print(len(reg))
    # cdef AlignmentFile file = AlignmentFile(in_file)
    # cdef AlignedSegment r
    # qnames = []
    # for r in file:
    #     qnames.append(r.qname)
    #     print(r.qname)
    # print(len(qnames))
    # print(len(set(qnames)))
