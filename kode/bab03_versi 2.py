"""Bab 3: versi Python dan pustaka yang dipakai untuk mencetak buku."""
import platform

import matplotlib
import numpy
import scipy
import sklearn

print(f"Python       {platform.python_version()}")
for nama, m in (("NumPy", numpy), ("SciPy", scipy),
                ("scikit-learn", sklearn), ("Matplotlib", matplotlib)):
    print(f"{nama:12s} {m.__version__}")
