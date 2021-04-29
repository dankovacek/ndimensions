#foo.py
import numpy as np
import pandas as pd

from skimage.util import view_as_blocks

w_tot, h_tot = 8, 8

ny, nx = 2, 2

dh = int(h_tot/ny)
dw = int(h_tot/nx)

A = np.empty((h_tot, w_tot))
n = 0
for i in range(h_tot):
    for j in range(w_tot):
        A[i, j] = n
        n += 1

print(A)

view = view_as_blocks(A, (ny, nx))

for i in range(view.shape[0]):
    for j in range(view.shape[1]):
        print(view[i, j])
        view[i, j] = int(np.mean(view[i, j]))

out = np.swapaxes(view, 1, 2).reshape(w_tot, h_tot)
print(out)

