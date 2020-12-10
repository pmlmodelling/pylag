import subprocess
import pstats, cProfile

profile_out = "profile.prof"

subprocess.call(["python", "-m", "cProfile", "-o", profile_out, "prof_main.py", "-c", "pylag.cfg"])

s = pstats.Stats(profile_out)
s.strip_dirs().sort_stats("time").print_stats(20)
s.strip_dirs().sort_stats('cumulative').print_stats(20)
s.strip_dirs().sort_stats("calls").print_stats(20)

