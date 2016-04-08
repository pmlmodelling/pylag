.. _programming_language:

Programming language
--------------------

PyLag has been written in a mixture of **Python** and **Cython**. Python is an
interpreted, dynamic programming language that includes a large standard 
library. By using the standard library for operations such as reading and 
writing output files, parsing configuration files, or writing log files one
greatly reduces the amount of code that must be implemented from scratch. PyLag 
makes use of such standard applications wherever possible.

Because Python is dynamically typed, it tends to be slower that statically typed
languages such as C, C++ or Java. For this reason, performance critical sections
of PyLag's code are writting in Cython. Cython is a programming language that 
supports optional static type declarations. Cython modules are translated into
optimised C/C++ code which is then compiled as Python extension modules. If
calls to Python's API are minimised, the resulting code has performance
characteristics comparable with those of pure C/C++ code.