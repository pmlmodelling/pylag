#!//usr/bin/env bash

set dir .

mpiexec -np 4 python -m pylag.parallel.main -c pylag.cfg

