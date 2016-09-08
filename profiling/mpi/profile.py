import subprocess
import pstats, cProfile

profile_out = "profile.prof"

n_procs = "4"

subprocess.call(["mpiexec", "-output-filename", profile_out, "-np", n_procs, "python", "-m", "cProfile", "-s", "time", "prof_main.py", "-c", "pylag.cfg"])


