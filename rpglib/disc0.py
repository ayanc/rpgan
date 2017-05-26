#-- Ayan Chakrabarti <ayanc@ttic.edu>
import numpy as np
import tensorflow as tf

class Dnet:

    def __init__(self,params):

        self.weights = {}

        self.bsz = params.bsz
        self.f = params.df
        imsz = params.imsz

        self.numd = 1
            
        self.numc = 3
        self.numl = int(np.log2(imsz))
                

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
            w = tf.Variable(tf.random_uniform([ksz,ksz,f,f1],\
                                minval=-sq,maxval=sq,dtype=tf.float32))
            b = tf.Variable(tf.constant(0,shape=[f1],dtype=tf.float32))

            self.weights['c%d_w'%j] = w
            self.weights['c%d_b'%j] = b


    def dloss(self,im,floss):
        out = im

        for j in range(self.numl):
            w = self.weights['c%d_w'%j]
            b = self.weights['c%d_b'%j]

            if j < self.numl-1:
                strides = [1,2,2,1]
                out = tf.pad(out,[[0,0],[1,1],[1,1],[0,0]])
            else:
                strides = [1,1,1,1]
                    
            out = tf.nn.conv2d(out,w,strides,'VALID')
            out = out + b
            if j < self.numl-1:
                out = tf.maximum(0.2*out,out)

        if floss:
            loss2 = tf.reduce_mean(tf.nn.softplus(out))
        else:
            loss2 = None
        loss = tf.reduce_mean(tf.nn.softplus(-out))

        return loss, loss2
