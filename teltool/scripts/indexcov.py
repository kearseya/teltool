import pysam

file = "/home/alex/Desktop/uni/PhD/TL_prediction/raw_data/full/DB143.bam"#
#file = "/mnt/data/DB209.bam"
fn = str.split(file, "/")[-1]
sample = str.split(fn, ".")[0]
print(sample)

def index_stats(file, sample):
    f = pysam.AlignmentFile(file)
    s = f.get_index_statistics()
    total = 0
    hg38 = 3099734149
    ref_len = 0
    for i in s:
        total += i.mapped
        ref_len += f.get_reference_length(i.contig)
        #print(i)
    #print((total*98)/hg38)
    print(ref_len)
    cov = total*98/ref_len
    print(sample,cov)
    return cov

index_stats(file, sample)
