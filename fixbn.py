#!/usr/bin/env python
#-- Ayan Chakrabarti <ayanc@ttic.edu>

from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
from skimage.io import imsave

from rpglib import utils as ut
from rpglib import genx as gen


if len(sys.argv) < 2:
    sys.exit("USAGE: fixbn.py exp [iteration]")


from importlib import import_module
p = import_module("exp." + sys.argv[1])
p.bsz = 2048 # Run in CPU mode

#########################################################################
if len(sys.argv) == 2:
    gsave = ut.ckpter(p.wts_dir + '/iter_*.gmodel.npz')
    mfile = gsave.latest
    if mfile is None:
        sys.exit("Could not find anything in " + p.wts_dir)
    niter = gsave.iter
else:
    mfile = p.wts_dir + '/iter_' + sys.argv[2] + '.gmodel.npz'
    niter = int(sys.argv[2])

ofile = p.wts_dir + '/iter_' + ('%d' % niter) + '.bgmodel.npz'
#########################################################################

# Set up Generator
Z = tf.random_uniform([p.bsz,1,1,p.zlen],-1.0,1.0)
G = gen.Gnet(p,Z,True)

#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.initialize_all_variables())

#########################################################################

print("Restoring G from " + mfile )
ut.netload(G,mfile,sess)
print("Done!")

#########################################################################

print("Running forward pass.")
_=sess.run(G.bnops)
print("Saving to %s."%ofile)
ut.netsave(G,ofile,sess)
print("Done!\n")
