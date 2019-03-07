"""
Visualizing .h5 files of weights
"""

import h5py
import numpy as np

filename = '/Users/sheilasagear/github/keras-training-ssagear/keras-training/train/train_3layer_binary/KERAS_check_best_model_weights.h5'
f = h5py.File(filename, 'r')
print(list(f))
#print(f['fc1_relu'])

#edited from https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path

with h5py.File(filename, 'r') as f:
    for dset in traverse_datasets(filename):
        print('Shape:', f[dset].shape)
        #print('Data type:', f[dset].dtype)
        print(f[dset][:])
