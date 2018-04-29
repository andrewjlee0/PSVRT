import numpy as np
import os, sys
import glob
import ipdb
#import experiments.regularCNN.params
from helpers import figure_helpers
from helpers.utility import find_files

if __name__ == '__main__':

    # params = params.get_params()
    # max_train_iters = params['num_max_train_imgs']/params['batch_size']
    # val_period_iters = params['num_val_period_imgs']/params['batch_size']
    max_train_iters = 20000000/50
    val_period_iters = 50000/50

    root_dir = '/home/jk/Desktop/PSVRTSD_summary'

    files = find_files([], dirs=[root_dir], contains=['.npy'])
    for i, fn in enumerate(files):
        learning_curve = np.load(fn)
        save_as = fn[:-3]+'pdf'
        figure_helpers.learningcurve_from_list(learning_curve, save_as, max_train_iters, val_period_iters)