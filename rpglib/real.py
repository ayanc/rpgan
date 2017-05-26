# Ayan Chakrabarti <ayanc@ttic.edu>
import tensorflow as tf
import numpy as np

class Real:
    def graph(self,bsz,imsz,crop=False):
        self.names = []
        # Create placeholders
        for i in range(bsz):
            self.names.append(tf.placeholder(tf.string))

        batch = []
        for i in range(bsz):
            # Load image
            img = tf.read_file(self.names[i])
            code = tf.decode_raw(img,tf.uint8)[0]
            img = tf.cond(tf.equal(code,137),
                          lambda: tf.image.decode_png(img,channels=3),
                          lambda: tf.image.decode_jpeg(img,channels=3))

            if crop:
                in_s = tf.to_float(tf.shape(img)[:2])
                min_s = tf.minimum(in_s[0],in_s[1])
                new_s = tf.to_int32((float(imsz+1)/min_s)*in_s)
                img = tf.image.resize_images(img,new_s[0],new_s[1])
                img = tf.random_crop(img,[imsz,imsz,3])

            batch.append(tf.expand_dims(img,0))
            
        batch = tf.to_float(tf.concat(0,batch))*(2.0/255.0) - 1.0

        # Fetching logic
        self.batch = tf.Variable(tf.zeros([bsz,imsz,imsz,3],dtype=tf.float32),trainable=False)
        self.fetchOp = tf.assign(self.batch,batch).op

    def fdict(self):
        fd = {}

        for i in range(len(self.names)):
            idx = self.idx[self.niter % self.ndata]
            self.niter = self.niter + 1
            if self.niter % self.ndata == 0:
                self.idx = np.int32(self.rand.permutation(self.ndata))

            fd[self.names[i]] = self.files[idx]
        return fd
        
    def __init__(self,lfile,bsz,imsz,niter,crop=False):

        # Setup fetch graph
        self.graph(bsz,imsz,crop)

        # Load file list
        self.files = []
        for line in open(lfile).readlines():
            self.files.append(line.strip())
        self.ndata = len(self.files)
        
        # Setup shuffling
        self.niter = niter*bsz
        self.rand = np.random.RandomState(0)
        idx = self.rand.permutation(self.ndata)
        for i in range(niter // self.ndata):
            idx = self.rand.permutation(ndata)
        self.idx = np.int32(idx)
