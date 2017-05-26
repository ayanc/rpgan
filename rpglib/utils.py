#-- Ayan Chakrabarti <ayanc@ttic.edu>
import re
import os
from glob import glob
import numpy as np

# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()
        
    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])

## Read weights
def netload(net,fname,sess):
    wts = np.load(fname)
    for k in wts.keys():
        wvar = net.weights[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(wk))

# Save weights to an npz file
def netsave(net,fname,sess):
    wts = {}
    for k in net.weights.keys():
        wts[k] = net.weights[k].eval(sess)
    np.savez(fname,**wts)
