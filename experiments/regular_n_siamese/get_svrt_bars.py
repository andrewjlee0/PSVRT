import os

import matplotlib.pyplot as plt
import numpy as np
import ipdb

if __name__ == '__main__':

    accuracies = [0.56, 0.565, 0.585,
                  0.605,0.62,0.635,
                  0.665,0.685,0.76,
                  0.865,0.885,0.89,
                  0.905,0.945,0.95,
                  0.97,1.0,1.0,
                  1.0,1.0,1.0,
                  1.0,1.0]
    colors = ['r', 'r', 'r',
              'r', 'r', 'r',
              'r', 'r', 'r',
              'b', 'b', 'b',
              'b', 'b', 'm',
              'b', 'b', 'b',
              'b', 'b', 'b',
              'b', 'b']
    problems = [20,7,21,19,1,22,5,15,16,6,17,9,13,23,8,14,4,12,10,18,3,11,2]
    load_root = '/home/jk/Desktop/'
    save_root = os.path.join(load_root, 'figsave')

    plt.figure(figsize=(14.5, 6.5))
    barlist = plt.bar(range(23), accuracies, alpha=0.6)
    for i,baritem in enumerate(barlist):
        baritem.set_color(colors[i])
    plt.legend([barlist[0],barlist[10]], ['Same-Different','Spatial Relations'], loc=2, fontsize=23)
    plt.xlabel('Problem label', fontsize=23)
    plt.ylabel('Accuracy', fontsize=23)
    xtick_markers = [str(label) for label in problems]
    ytick_vals = np.linspace(0.5, 1, 6)
    ytick_markers = [str(0.5 + float(k)*0.1) for k in range(6)]
    plt.xticks(range(23), xtick_markers)
    plt.yticks(ytick_vals, ytick_markers)
    plt.ylim(0.5, 1)
    plt.tick_params(axis='both', which='major', labelsize=23)
    save_as = 'svrt_bars.pdf'
    plt.savefig(os.path.join(save_root, save_as))
    plt.clf()
