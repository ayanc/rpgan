#!/usr/bin/env python
#-- Ayan Chakrabarti <ayanc@ttic.edu>

import sys
import os
import time
import tensorflow as tf
import numpy as np

if len(sys.argv) < 2:
    sys.exit("USAGE: train.py exp")


from importlib import import_module
p = import_module("exp." + sys.argv[1])

from rpglib import utils as ut
from rpglib import real
from rpglib import gen
from rpglib import disc

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

# Real
imgs = real.Real(p.lfile,p.bsz,p.imsz,niter,p.crop)

# Noise
Z = tf.Variable(tf.random_uniform([p.bsz,1,1,p.zlen],-1.0,1.0))
G = gen.Gnet(p,Z)
Gz = tf.Variable(tf.zeros([p.bsz,p.imsz,p.imsz,3],dtype=tf.float32))
gfwd = Gz.assign(G.out)

# Discriminator
D = disc.Dnet(p)

or2r,_ = D.dloss(imgs.batch,False)
of2r,of2f = D.dloss(Gz,True)
dloss = (or2r+of2f) / 2.0
gloss = of2r / float(D.numd)

#########################################################################

# Set up optimizer steps

# For D
opt0 = tf.train.GradientDescentOptimizer(1.0)
gv = opt0.compute_gradients(dloss,D.v0)
dsteps = []
for i in range(D.numd):
    opt = tf.train.AdamOptimizer(2e-4,0.5)
    gvi = [(gv[j][0],D.vk[i][j]) for j in range(len(gv))]
    dsteps.append(opt.apply_gradients(gvi))

# For G
GzGrad = tf.Variable(tf.zeros([p.bsz,p.imsz,p.imsz,3],dtype=tf.float32))
gstep0 = GzGrad.initializer

opt0 = tf.train.GradientDescentOptimizer(1.0)
gv = opt0.compute_gradients(gloss,[Gz])
gstepi = GzGrad.assign_add(gv[0][0])

opt = tf.train.AdamOptimizer(2e-4,0.5)
gstepF = opt.minimize(tf.reduce_sum(GzGrad*G.out),\
                      var_list=[G.weights[k] for k in G.weights.keys()])


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

        # Run GStep
        sess.run(Z.initializer)
        sess.run([gfwd,gstep0])

        gl = 0.
        gli = []
        for i in range(D.numd):
            sess.run(D.sOps[i])
            glv,_ = sess.run([gloss,gstepi])
            gl = gl + glv
            gli.append(glv*float(D.numd))
        sess.run(gstepF)
            
        # Run DStep
        sess.run(Z.initializer)
        sess.run([gfwd,imgs.fetchOp],feed_dict=imgs.fdict())
        dl = 0.
        for i in range(D.numd):
            sess.run(D.sOps[i])
            dlv,_ = sess.run([dloss,dsteps[i]])
            dl = dl+dlv
        dl = dl/float(D.numd)
        
        mprint("[%09d] Adam Loss: G=%.6f,D=%.6f"
                   % (niter,gl,dl))

        # Display all outputs
        ostr = '[%09d]* ' % niter
        for j in range(D.numd):
            ostr = ostr + ("L%02d=%.3f," % (j,gli[j]))
        mprint(ostr[:-1])

        niter=niter+1
                    
        ## Save model weights if needed
        if p.SAVEFREQ > 0 and niter % p.SAVEFREQ == 0:
            gname = p.wts_dir + "/iter_%d.gmodel.npz" % niter

            ut.netsave(G,gname,sess)
            gsave.clean(every=p.SAVEFREQ,last=1)
            mprint("Saved G weights to " + gname )


except KeyboardInterrupt: # Catch ctrl+c/SIGINT
    mprint("Stopped!")
    stop = True
    pass

# Save last
dname = p.wts_dir + "/iter_%d.dmodel.npz" % niter
gname = p.wts_dir + "/iter_%d.gmodel.npz" % niter

if gsave.iter < niter:
    ut.netsave(G,gname,sess)
    gsave.clean(every=p.SAVEFREQ,last=1)
    mprint("Saved G weights to " + gname )

ut.netsave(D,dname,sess)
dsave.clean(every=p.SAVEFREQ,last=1)
mprint("Saved D weights to " + dname )
