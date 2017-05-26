# Generating Random Projection Filters

This directory contains the different filter weight files used for training the models in the paper. The filter files are named `Knn_dm12.npz`, where nn denotes the number of projections / discriminators. Note that all filters downsample spatially by 2 and go from a 3 channel to a single channel image, through convolutions with 8x8 filters.

These filters were generated with the `rproj.py` script provided here as:
```bash
$ ./rproj.py K12_dm12.npz 8,2,1,12
$ ./rproj.py K24_dm12.npz 8,2,1,24
$ ./rproj.py K48_dm12.npz 8,2,1,48
```
The second parameter specifies kernel size, stride, number of output channels, and number of projections/discriminators.

To use these weights during training, place them as `filts.npz` in the weights directory for an experiment.
