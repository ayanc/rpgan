# Generator Parameters
ksz=4
zlen = 200 # Dimensionality of z
f1 = 2048  # Features in first layer of Gen output 

# Discrimnator Parameters
df = 64   # No. of hidden features (at first layer of D)

# Training set
imsz = 128
bsz = 64
lfile='data/dogs.txt'
crop=True

wts_dir='models/dog12'
SAVEFREQ=1e3
MAXITER=1e5
