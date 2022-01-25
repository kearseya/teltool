import numpy as np
import array


__all__ = ["ffs", "basemap", "basemap_2_int", "char_to_nibble_array", "hash2seq", "same_as_str_split",
           "seq_2_nibbles"]


def ffs(x):
    """Returns the index, counting from 0, of the
    least significant set bit in `x`.
    """
    # there should be a c version of this function in <string.h>
    # https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
    return (x&-x).bit_length()-1


def basemap():
    return np.array(['.', 'A', 'C', '.', 'G', '.', '.', '.', 'T', '.', '.', '.', '.', '.', 'N'])


def basemap_2_int():
    return {k: i for i, k in enumerate(basemap())}

# print({k: bin(v) for k, v in basemap_2_int.items() if k in "ATCG"})
# print({k: ffs(v) for k, v in basemap_2_int.items() if k in "ATCG"})

# in 2bit format:
# A = 0 (bits = 00), C = 1 (01), G = 2 (10), T = 3 (11), N = 0 (00)
# if input is ATCG
# (00) (11) (01) (10) = 54

# in nibble format
# A = 0001, C = 0010, G = 0100, T = 1000
# ATCG = first byte = (0001) (1000)  second byte = (0010) (0100) = [24, 36]


def nibble_2bit():
    return np.array([0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")


def twobit_2_base():
    return {0: "A", 1: "C", 2: "G", 3: "T"}


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


def hash2seq(v, k):
    twobit_2_base_d = twobit_2_base()
    seq = ""
    v = int(v)  # python int
    for _ in range(k):
        seq += twobit_2_base_d[v & 3]
        v = v >> 2
    return seq[::-1]


def same_as_str_split(extracted, seq, k):
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    rolling_kmers = [hash2seq(i, k) for i in extracted]
    assert all(i == j for i, j in zip(kmers, rolling_kmers))
