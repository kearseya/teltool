#!/usr/bin/env python
"""Print out start:stop locations for each reference in a BAI file.
Usage:
    bai_indexer.py /path/to/file.bam.bai > /path/to/file.bam.bai.json
"""

import struct
import sys
import numpy as np
import pysam

# -- helper functions for reading binary data from a stream --

def _unpack(stream, fmt):
    size = struct.calcsize(fmt)
    buf = stream.read(size)
    return struct.unpack(fmt, buf)[0]


def _read_int32(stream):
    return _unpack(stream, '<i')


def _read_uint32(stream):
    return _unpack(stream, '<I')


def _read_uint64(stream):
    return _unpack(stream, '<Q')


class _TellingStream(object):
    """Wrapper for a stream which adds support for tell(), e.g. to stdin."""
    def __init__(self, stream):
        self._stream = stream
        self._pos = 0

    def read(self, *args):
        data = self._stream.read(*args)
        self._pos += len(data)
        return data

    def tell(self):
        return self._pos


class InvalidBaiFileError(Exception):
    pass


def index_stream(bai_stream):
    """Generate an index of a BAM Index (BAI) file.
    Args:
        data_stream: A stream of bytes from the BAI file, as returned
            by open().  Anything with a .read() method will do.
    Returns:
        A dict with information about the BAM and BAI files. For example:
        {'minBlockIndex': 1234,

        ### THIS HAS BEEN CHANGED TO BE VOFFSET CHUNKS
         'chunks': [[8, 123456], [123456, 234567], ...]}
        The chunks are [start, stop) byte ranges in the BAI file for each
        ref. minBlockIndex is the position of the first block in the BAM
        file.
        ###

    Raises:
        InvalidBaiFileError: if the bytes do not comprise a valid BAI file.
    """
    data_stream = _TellingStream(bai_stream)

    # The logic and naming below follow the diagram on page 16 of
    # http://samtools.github.io/hts-specs/SAMv1.pdf
    magic = data_stream.read(4)
    #if magic != 'BAI\x01':
    #    raise InvalidBaiFileError('This is not a BAI file (missing magic)')

    minBlockIndex = 1000000000
    refs = []
    ##n_bins = []
    ##n_chunks = []
    ##ioffsets = []
    n_ref = _read_int32(data_stream)
    intvs = []
    chunks = []
    for i in range(0, n_ref):
        ref_start = data_stream.tell()
        n_bin = _read_int32(data_stream)
        ##n_bins.append({n_bin: []})
        ##n_chunks.append({n_bin: []})
        for j in range(0, n_bin):
            bin_id = _read_uint32(data_stream)
            ##n_bins[i][n_bin].append(bin_id)
            n_chunk = _read_int32(data_stream)
            ##n_chunks[i][n_bin].append(n_chunk)
            #chunks = [] # original
            for k in range(0, n_chunk):
                chunk_beg = _read_uint64(data_stream)
                chunk_end = _read_uint64(data_stream)
                chunks.append((chunk_beg, chunk_end))


        n_intv = _read_uint32(data_stream)
        #intvs = [] # original
        intvs.append(n_intv)
        for j in range(0, n_intv):
            ioffset = _read_uint64(data_stream)
            ##ioffsets.append(ioffsets)
            if ioffset:
                # These values are "virtual offsets". The first 48 bits are the
                # offset of the start of the compression block in the BAM file.
                # The remaining 16 bits are an offset into the inflated block.
                bi = ioffset / 65536
                if ioffset % 65536 != 0:
                    bi += 65536
                minBlockIndex = min(minBlockIndex, bi)
        ref_end = data_stream.tell()

        refs.append((ref_start, ref_end))

    # Undocumented field: # of unmapped reads
    # See https://github.com/samtools/hts-specs/pull/2/files
    try:
        num_unmapped = _read_uint64(data_stream)
    except struct.error:
        pass

    extra_bytes = data_stream.read()
    #if extra_bytes != '':
    #    raise InvalidBaiFileError(
    #            'Extra data after expected EOF (%d bytes)' % len(extra_bytes))

    return {
        'chunks': chunks,

        #'minBlockIndex': int(minBlockIndex),
        #'n_ref': n_ref,

        ## Previously called CHUNKS but actually reference chunks
        #'refs': refs,

        'intvs': intvs
        #'ioffsets': ioffsets #ellipsis
        #'n_bins': n_bins,
        #'n_chunks': n_chunks,
    }

def reference_stats(file):
    f = pysam.AlignmentFile(file)
    rs = f.get_index_statistics()
    print(rs)
    return




def run():
    #reference_stats("/home/alex/Desktop/uni/PhD/TL_prediction/raw_data/full/DB143.bam")
    TileWidth = 16384
    fn = "/home/alex/Desktop/uni/PhD/TL_prediction/raw_data/full/DB143.bam.bai"
    print(str.split(fn, "/")[-1])
    data = open(fn, 'rb')
    out = index_stream(data)
    sizes = []
    for i in out["chunks"]:
        sizes.append(i[1]-i[0])

    ## Not sure if needed if not doing by tile depth
    #n98 = np.percentile(sizes, 98)
    #for i, x in enumerate(sizes):
    #    if x > n98:
    #        sizes[i] = n98

    medianSizePerTile = np.median(sizes)
    print("Length of sizes")
    print(len(sizes))
    print("Sum intvs")
    print(sum(out["intvs"]))
    print("")
    print("Median size per tile")
    print(medianSizePerTile)
    print("")
    print("Median / Tile size")
    print(medianSizePerTile/TileWidth)
    #print(out)


if __name__ == '__main__':
    run()



"""
### GO IMPLIMENTATION OF INDEXCOV FOR REFERENCE

func getSizes(idx *bam.Index) ([][]int64, uint64, uint64) {
	var mapped, unmapped uint64
	refs := reflect.ValueOf(*idx).FieldByName("idx").FieldByName("Refs")
	ptr := unsafe.Pointer(refs.Pointer())

	ret := (*(*[1 << 28]oRefIndex)(ptr))[:refs.Len()]
	// save some memory.
	m := make([][]int64, len(ret))
	n_messages := 0
	for i, r := range ret {
		st, ok := idx.ReferenceStats(i)
		if ok {
			mapped += st.Mapped
			unmapped += st.Unmapped
		} else {
			if n_messages <= 10 {
				log.Printf("no reference stats found for %dth reference chromosome", i)
			}
			if n_messages == 10 {
				log.Printf("not reporting further chromosomes without stats. %d", i)
			}
			n_messages += 1
		}
		if len(r.Intervals) < 2 {
			m[i] = make([]int64, 0)
			continue
		}
		m[i] = make([]int64, len(r.Intervals)-1)
		for k, iv := range r.Intervals[1:] {
			m[i][k] = vOffset(iv) - vOffset(r.Intervals[k])
			if m[i][k] < 0 {
				panic("expected positive change in vOffset")
			}
		}
		r.Bins, r.Intervals = nil, nil
	}
	return m, mapped, unmapped
}


// init sets the medianSizePerTile
func (x *Index) init() {
	if x.Index != nil {
		x.sizes, x.mapped, x.unmapped = getSizes(x.Index)
		x.Index = nil
	} else if x.crai != nil {
		x.sizes = x.crai.Sizes()
		if x.sizes == nil {
			log.Fatal("bad index:", x.path)
		}
		x.crai = nil
	}

	// sizes is used to get the median.
	sizes := make([]int64, 0, 16384)
	for k := 0; k < len(x.sizes); k++ {
		sizes = append(sizes, x.sizes[k]...)
	}
	if len(sizes) < 1 {
		log.Fatalf("indexcov: no usable chromsomes in bam: %s", x.path)
	}

	sort.Slice(sizes, func(i, j int) bool { return sizes[i] < sizes[j] })
	total := int64(0)
	cumsum := make([]int64, len(sizes))

	// some bins with very high coverage can screw the normalization.
	// so we cap at the 98th percentile and then take the bin that passed then
	// median.
	n98 := sizes[int(0.98*float64(len(sizes)))]
	for i, s := range sizes {
		if s > n98 {
			s = n98
		}
		total += s
		cumsum[i] = total
	}
	var idx = sort.Search(len(cumsum), func(i int) bool { return (cumsum[i]) > (total / 2) })
	for idx >= len(sizes) {
		idx--
	}

	x.medianSizePerTile = float64(sizes[idx])
}
"""
