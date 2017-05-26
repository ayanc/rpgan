#-- Ayan Chakrabarti <ayanc@ttic.edu>
import numpy as np
import tensorflow as tf
import sys

class Dnet:

    def __init__(self,params):

        self.weights = {}

        self.bsz = params.bsz
        self.f = params.df
        imsz = params.imsz
        
        f = np.load(params.wts_dir + '/filts.npz')
        self.numd = int(f['nfilt'])
            
        # Expect all filters to be same stride/numc
        fi = f['p0']
        self.numc = fi.shape[-1]
        stride = int(f['s0'])
        self.stride = [1,stride,stride,1]
        wsz = imsz // stride
        self.numl = int(np.log2(wsz))

        f0 = fi.shape[0]
        pad = (f0-stride)//2
        if pad == 0:
            self.pad = None
        else:
            self.pad = [[0,0],[pad,pad],[pad,pad],[0,0]]
        
        self.v0 = []
        self.vk = []
        self.sOps = []
        self.filt = tf.Variable(tf.zeros(fi.shape,dtype=tf.float32))

        for i in range(self.numd):
            fi = f['p%d' % i]
            if int(f['s%d' % i]) != stride:
                sys.exit("Expect all filters to be same stride.")
            if fi.shape[-1] != self.numc:
                sys.exit("Expect all filters to have same number of channels.")
            if fi.shape[0] != f0:
                sys.exit("Expect all filters to be same size.")

            self.sOps.append([self.filt.assign(tf.constant(fi))])
            self.vk.append([])
            
                
        for j in range(self.numl):
            if j == 0:
                f = self.numc
            else:
                f = self.f*(2**(j-1))

            if j < self.numl-1:
                ksz = 4
                f1 = self.f*(2**j)
            else:
                ksz = 2
                f1 = 1

            sq = np.sqrt(3.0 / np.float32(ksz*ksz*f))
                
                            
            w0 = tf.Variable(tf.random_uniform([ksz,ksz,f,f1],\
                            minval=-sq,maxval=sq,dtype=tf.float32))
            self.v0.append(w0)
            b0 = tf.Variable(tf.constant(0,shape=[f1],dtype=tf.float32))
            self.v0.append(b0)

            for i in range(self.numd):
                w = tf.Variable(tf.random_uniform([ksz,ksz,f,f1],\
                            minval=-sq,maxval=sq,dtype=tf.float32))
                self.vk[i].append(w)
                self.sOps[i].append(w0.assign(tf.identity(w)))
                
                b = tf.Variable(tf.constant(0,shape=[f1],dtype=tf.float32))
                self.vk[i].append(b)
                self.sOps[i].append(b0.assign(tf.identity(b)))
            
                self.weights['c%d_%d_w'%(i,j)] = w
                self.weights['c%d_%d_b'%(i,j)] = b


    def dloss(self,im,floss):

        loss2=None
        if self.pad is None:
            out = im
        else:
            out = tf.pad(im,self.pad)

        out = tf.nn.conv2d(out,self.filt,self.stride,'VALID')

        idx = 0
        for j in range(self.numl):

            if j < self.numl-1:
                strides = [1,2,2,1]
                out = tf.pad(out,[[0,0],[1,1],[1,1],[0,0]])
            else:
                strides = [1,1,1,1]
                    
            out = tf.nn.conv2d(out,self.v0[idx],strides,'VALID')
            out = out + self.v0[idx+1]
            idx = idx + 2
            if j < self.numl-1:
                out = tf.maximum(0.2*out,out)

        if floss:
            loss2 = tf.reduce_mean(tf.nn.softplus(out))
        loss = tf.reduce_mean(tf.nn.softplus(-out))

        return loss,loss2
