#!/usr/bin/env python
#-- Ayan Chakrabarti <ayanc@ttic.edu>

import sys
import os
import time
import tensorflow as tf
import numpy as np

from rpglib import utils as ut
from rpglib import real

from rpglib import gen
from rpglib import disc0 as disc

if len(sys.argv) < 2:
    sys.exit("USAGE: baseline_train.py exp")

from importlib import import_module
p = import_module("exp." + sys.argv[1])

def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()


#########################################################################

# Check for saved weights & find iter
dsave = ut.ckpter(p.wts_dir + '/iter_*.dmodel.npz')
gsave = ut.ckpter(p.wts_dir + '/iter_*.gmodel.npz')

niter = gsave.iter

#########################################################################

# Initialize loader, generator, discriminator

imgs = real.Real(p.lfile,p.bsz,p.imsz,niter,p.crop)
Z = tf.random_uniform([p.bsz,1,1,p.zlen],-1.0,1.0)

G = gen.Gnet(p,Z)
D = disc.Dnet(p)

or2r,_ = D.dloss(imgs.batch,False)
of2r,of2f = D.dloss(G.out,True)
dloss = (or2r+of2f) / 2.0
gloss = of2r

#########################################################################

# Set up optimizer steps

# For D
opt = tf.train.AdamOptimizer(2e-4,0.5)
dstep = opt.minimize(dloss,var_list=[D.weights[k] for k in D.weights.keys()])

# For G
opt = tf.train.AdamOptimizer(2e-4,0.5)
gstep = opt.minimize(gloss,var_list=[G.weights[k] for k in G.weights.keys()])


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

# Load saved weights if any
if dsave.latest is not None:
    mprint("Restoring D from " + dsave.latest )
    ut.netload(D,dsave.latest,sess)
    mprint("Done!")

if gsave.latest is not None:
    mprint("Restoring G from " + gsave.latest )
    ut.netload(G,gsave.latest,sess)
    mprint("Done!")

#########################################################################

# Main Training loop

stop=False
mprint("Starting from Iteration %d" % niter)
try:
    while niter < p.MAXITER and not stop:
            
        # Run gstep and fetch images
        f2rv = sess.run([gloss,gstep,imgs.fetchOp],feed_dict=imgs.fdict())
        glv = f2rv[0]
        # Run dstep
        dlv,_ = sess.run([dloss,dstep])
        mprint("[%09d] Adam Loss: G=%.6f,D=%.6f"
                   % (niter,glv,dlv))

        niter=niter+1
                    
        ## Save model weights if needed
        if p.SAVEFREQ > 0 and niter % p.SAVEFREQ == 0:
            dname = p.wts_dir + "/iter_%d.dmodel.npz" % niter
            gname = p.wts_dir + "/iter_%d.gmodel.npz" % niter

            ut.netsave(G,gname,sess)
            gsave.clean(every=p.SAVEFREQ,last=1)
            mprint("Saved G weights to " + gname )


except KeyboardInterrupt: # Catch ctrl+c/SIGINT
    mprint("Stopped!")
    stop = True
    pass

# Save last
if gsave.iter < niter:
    dname = p.wts_dir + "/iter_%d.dmodel.npz" % niter
    gname = p.wts_dir + "/iter_%d.gmodel.npz" % niter

    ut.netsave(D,dname,sess)
    dsave.clean(every=p.SAVEFREQ,last=1)
    mprint("Saved D weights to " + dname )

    ut.netsave(G,gname,sess)
    gsave.clean(every=p.SAVEFREQ,last=1)
    mprint("Saved G weights to " + gname )
