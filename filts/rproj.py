#!/usr/bin/env python
#-- Ayan Chakrabarti <ayanc@ttic.edu>

from __future__ import print_function
import numpy as np
import sys

if len(sys.argv) < 3:
    sys.exit("USAGE: outfile.npz [ksize,stride,numf,numk]_repeated")

ofile = sys.argv[1]

wts = {}
nfilt = 0

for i in range(len(sys.argv)-2):
    args = sys.argv[i+2].split(',')
    ksize = int(args[0])
    stride = int(args[1])
    numf = int(args[2])
    numk = int(args[3])
    
    for j in range(numk):
        pfi = np.random.normal(size=(ksize,ksize,3,numf))
        pfi = np.float32(pfi)
        pfi = pfi / np.sqrt(np.sum(pfi*pfi,axis=(0,1,2)))
        wts['p%d' % nfilt] = pfi
        wts['s%d' % nfilt] = stride
        nfilt = nfilt + 1

wts['nfilt'] = nfilt
np.savez(ofile,**wts)
print("Wrote %d discriminator projections to %s." % (nfilt,ofile))
