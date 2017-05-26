# Generator Parameters
ksz=4
zlen = 100 # Dimensionality of z
f1 = 1024  # Features in first layer of Gen output 

# Discrimnator Parameters
df = 128   # No. of hidden features (at first layer of D)

# Training set
imsz = 64
bsz = 64
lfile='data/faces.txt'
crop=False

# Learning parameters
wts_dir='models/face24'
SAVEFREQ=1e3
MAXITER=1e5
