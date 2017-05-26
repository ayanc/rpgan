#-- Ayan Chakrabarti <ayanc@ttic.edu>
import numpy as np
import tensorflow as tf

class Gnet:

    def __init__(self,params,z):

        self.weights = {}
        
        f = params.zlen
        f1 = params.f1
        ksz = params.ksz
        bsz = params.bsz
        
        sz = ksz-ksz%2
        lnum = 1


        ####### First block is FC
        
        # Initialize
        sq = np.sqrt(3.0 / np.float32(f))
        w = tf.Variable(tf.random_uniform([1,1,f,f1*sz*sz],\
                            minval=-sq,maxval=sq,dtype=tf.float32))
        b = tf.Variable(tf.constant(0,shape=[f1],dtype=tf.float32))
        self.weights['c%d_w'%lnum] = w
        self.weights['c%d_b'%lnum] = b

        out = tf.nn.conv2d(z,w,[1,1,1,1],'VALID')
        out = tf.reshape(out,[-1,sz,sz,f1])

        om,ov = tf.nn.moments(out,[0,1,2])
        out = tf.nn.batch_normalization(out,om,ov,None,None,1e-3)
        out = out + b
        #out = tf.maximum(0.2*out,out)
        out = tf.nn.relu(out)
        
        lnum = lnum+1
        sz = sz*2
        f = f1
        f1 = f1//2

        # Subsequent blocks are deconv
        while sz <= params.imsz:
            # Initialize
            sq = np.sqrt(3.0 / np.float32(ksz*ksz*f))
            w = tf.Variable(tf.random_uniform([ksz,ksz,f1,f],\
                                minval=-sq,maxval=sq,dtype=tf.float32))
            b = tf.Variable(tf.constant(0,shape=[f1],dtype=tf.float32))

            self.weights['c%d_w'%lnum] = w
            self.weights['c%d_b'%lnum] = b
            lnum = lnum+1

            out = tf.nn.conv2d_transpose(out,w,[bsz,sz,sz,f1],[1,2,2,1],'SAME')
            if sz < params.imsz:
                om,ov = tf.nn.moments(out,[0,1,2])
                out = tf.nn.batch_normalization(out,om,ov,None,None,1e-3)
            
            out = out + b

            if sz == params.imsz:
                out = tf.nn.tanh(out)
            else:
                #out = tf.maximum(0.2*out,out)
                out = tf.nn.relu(out)

            sz = sz*2
            f = f1
            if sz == params.imsz:
                f1 = 3
            else:
                f1 = f1 //2

        self.out = out

        
