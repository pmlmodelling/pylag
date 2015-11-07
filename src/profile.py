import subprocess
import pstats, cProfile

import main

profile_out = "profile.prof"

subprocess.call(["python", "-m", "cProfile", "-o", profile_out, "main.py", "-c", "pylag.cfg"])

s = pstats.Stats(profile_out)
s.strip_dirs().sort_stats("time").print_stats(20)
s.strip_dirs().sort_stats('cumulative').print_stats(20)

