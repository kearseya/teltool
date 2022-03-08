#cython: language_level=3
from libcpp.vector cimport vector as cpp_vector
from libcpp.deque cimport deque as cpp_deque
from libcpp.pair cimport pair as cpp_pair
from libcpp.utility cimport pair
from libc.stdint cimport uint32_t, uint8_t, uint64_t, uint16_t, int32_t, int8_t

# xxhash
cdef extern from "xxhash64.h" namespace "XXHash64" nogil:
    #static uint64_t hash(const void* input, uint64_t length, uint64_t seed)
    cdef uint64_t hash(void* input, uint64_t length, uint64_t seed) nogil

# robin hood set
cdef extern from "robin_hood.h" namespace "robin_hood" nogil:
    cdef cppclass unordered_set[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void insert(T&)
        void erase(T&)
        int size()
        iterator find(const T&)
        iterator begin()
        iterator end()
        void clear()
        bint empty()
