#!/usr/bin/env python
#-- Ayan Chakrabarti <ayanc@ttic.edu>

from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
from skimage.io import imsave

from rpglib import utils as ut
from rpglib import gen

#########################################################################
if len(sys.argv) < 3:
    sys.exit("USAGE: sample.py exp[,seed] out.jpg [iteration]")

arg1 = sys.argv[1].split(",")
ename = arg1[0]
if len(arg1) == 1:
    seed = 0
else:
    seed = int(arg1[1])
    
    
from importlib import import_module
p = import_module("exp." + ename)
p.bsz = 150
layout = [10,15]

    
fname = sys.argv[2]

if len(sys.argv) == 3:
    gsave = ut.ckpter(p.wts_dir + '/iter_*.gmodel.npz')
    mfile = gsave.latest
else:
    mfile = p.wts_dir + '/iter_' + sys.argv[3] + '.gmodel.npz'
    
#########################################################################

# Initialize loader, generator, discriminator

Z = tf.placeholder(shape=[p.bsz,1,1,p.zlen],dtype=tf.float32)

G = gen.Gnet(p,Z)
img = G.out

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

print("Generating " + fname)
zval = np.float32(np.random.RandomState(seed).rand(p.bsz,1,1,p.zlen)*2.0-1.0)
imval = sess.run(G.out,feed_dict={Z: zval})
imval = np.uint8( (imval*0.5+0.5)*255.0)

imval = imval.reshape(layout + [p.imsz,p.imsz,3])
imval = imval.transpose([0,2,1,3,4]).copy()
imval = imval.reshape([layout[0]*p.imsz,layout[1]*p.imsz,3])
imsave(fname, imval)
