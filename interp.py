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

OLEN=150
N=11

#########################################################################
if len(sys.argv) < 4:
    sys.exit("USAGE: interp.py exp[,seed[,iteration]] out.png lid,rid lid,rid lid,rid ")

arg1 = sys.argv[1].split(",")
ename = arg1[0]
if len(arg1) == 1:
    seed = 0
else:
    seed = int(arg1[1])
if len(arg1) < 3:
    niter = None
else:
    niter = arg1[2]

from importlib import import_module
p = import_module("exp." + ename)

fname = sys.argv[2]

npair=len(sys.argv)-3
p.bsz = N*npair
layout = [npair,N]

lid = []
rid = []
for i in range(npair):
    ri,li = [int(x)-1 for x in sys.argv[i+3].split(',')]
    lid.append(li)
    rid.append(ri)

zval = np.float32(np.random.RandomState(seed).rand(OLEN,1,1,p.zlen)*2.0-1.0)
zleft = zval[lid,...]
zright = zval[rid,...]
sm = np.float32(np.linspace(0.0,1.0,N).reshape([1,N,1,1]))
zval = zleft*sm + zright*(1.0-sm)
zval = zval.reshape([p.bsz,1,1,p.zlen])

if niter is None:
    gsave = ut.ckpter(p.wts_dir + '/iter_*.bgmodel.npz')
    mfile = gsave.latest
else:
    mfile = p.wts_dir + '/iter_' + niter + '.bgmodel.npz'
    
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
imval = sess.run(G.out,feed_dict={Z: zval})
imval = np.uint8( (imval*0.5+0.5)*255.0)

imval = imval.reshape(layout + [p.imsz,p.imsz,3])
imval = imval.transpose([0,2,1,3,4]).copy()
imval = imval.reshape([layout[0]*p.imsz,layout[1]*p.imsz,3])
imsave(fname, imval)
