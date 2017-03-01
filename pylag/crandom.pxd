cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # uses mt19937 generator

    cdef cppclass normal_distribution[T]:
        normal_distribution() except +
        normal_distribution(T a, T b) except +
        T operator()(mt19937 generator) # uses mt19937 generator
