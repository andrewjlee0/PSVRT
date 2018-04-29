import os

import matplotlib.pyplot as plt
import numpy as np

plt.ioff()
plt.style.use('ggplot')

def find_files(files, dirs=[], contains=[]):
    for d in dirs:
        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for i, part in enumerate(contains):
            files += [os.path.join(d, f) for f in onlyfiles if part in f]
        onlydirs = [os.path.join(d, dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]
        if onlydirs:
            files += find_files([], onlydirs, contains)

    return files

def plot_box_extent(box_extent_list, problem_types, item_size, num_items,
                    num_max_train_imgs, num_yticks, colors, legend,
                    load_root, save_root):

    plt.figure(figsize=(8.5, 8))
    for iprob in range(len(problem_types)):
        root_dir = os.path.join(load_root,'PSVRT'+problem_types[iprob]+'_summary')
        # OVER BOX EXTENTS
        x_list = []
        y_list = []
        for i, be in enumerate(box_extent_list):
            target_dir = os.path.join(root_dir,str(be), str(num_items), str(item_size))
            files = find_files([], dirs=[target_dir], contains=['.txt'])
            x_list.append(i)
            tta_list=[]
            for j, fn in enumerate(files):
                reader = open(fn,'r')
                line1 = reader.readline()
                line2 = reader.readline()
                line3 = reader.readline()
                tta = str(line3.split()[-1])
                if tta == 'inf':
                    tta = num_max_train_imgs
                else:
                    tta = int(tta)
                tta_list.append(tta)
            if np.min(tta_list)<num_max_train_imgs:
                #y_list.append(np.min(tta_list))
                y_list.append(None)
            else:
                y_list.append(np.min(tta_list))
                #y_list.append(None)
        plt.plot(np.array(x_list), np.array(y_list), c=colors[iprob],
                 marker='o', linestyle='--', linewidth=5.0, markersize=14, alpha=0.5, label=problem_types[iprob])
    if legend:
        plt.legend(loc=2, fontsize=23)
    plt.xlabel('Image size', fontsize=23)
    plt.ylabel('Training images (millions)', fontsize=14)
    xtick_markers = [str(be[0]) for be in box_extent_list]
    ytick_vals = np.linspace(0, num_max_train_imgs, num_yticks)
    ytick_markers = [str(int(k/1000000)) for k in ytick_vals]
    ytick_markers[-1] = 'Never'
    plt.xticks(range(len(box_extent_list)), xtick_markers)
    plt.yticks(ytick_vals, ytick_markers)
    plt.xlim((-0.2, 5.2))
    plt.ylim((-num_max_train_imgs/(2*num_yticks), num_max_train_imgs+num_max_train_imgs/(2*num_yticks)))
    plt.tick_params(axis='both', which='major', labelsize=23)
    save_as = 'mtta_img_size.pdf'
    plt.savefig(os.path.join(save_root,save_as))
    plt.clf()

def plot_item_size(item_size_list, problem_types, box_extent , num_items,
                    num_max_train_imgs, num_yticks, colors, legend,
                    load_root, save_root):
    plt.figure(figsize=(8.5, 8))
    for iprob in range(len(problem_types)):
        root_dir = os.path.join(load_root, 'PSVRT' + problem_types[iprob] + '_summary')
        # OVER ITEM SIZES
        x_list = []
        y_list = []
        for i, sz in enumerate(item_size_list):
            target_dir = os.path.join(root_dir, str(box_extent), str(num_items), str(sz))
            files = find_files([], dirs=[target_dir], contains=['.txt'])
            x_list.append(i)
            tta_list = []
            for j, fn in enumerate(files):
                reader = open(fn, 'r')
                line1 = reader.readline()
                line2 = reader.readline()
                line3 = reader.readline()
                tta = str(line3.split()[-1])
                if tta == 'inf':
                    tta = num_max_train_imgs
                else:
                    tta = int(tta)
                tta_list.append(tta)
            if np.min(tta_list)<num_max_train_imgs:
                y_list.append(np.min(tta_list))
                #y_list.append(None)
            else:
                #y_list.append(np.min(tta_list))
                y_list.append(None)
        plt.plot(np.array(x_list), np.array(y_list), c=colors[iprob],
                 marker='o', linestyle='--', linewidth=5.0, markersize=14, alpha=0.5, label=problem_types[iprob])
    if legend:
        plt.legend(loc=2, fontsize=23)
    plt.xlabel('Item size', fontsize=23)
    plt.ylabel('Training images (millions)', fontsize=23)
    xtick_markers = [str(sz[0]) for sz in item_size_list]
    ytick_vals = np.linspace(0, num_max_train_imgs, num_yticks)
    ytick_markers = [str(int(k / 1000000)) for k in ytick_vals]
    ytick_markers[-1] = 'Never'
    plt.xticks(range(len(item_size_list)), xtick_markers)
    plt.yticks(ytick_vals, ytick_markers)
    plt.ylim((-num_max_train_imgs / (2 * num_yticks), num_max_train_imgs + num_max_train_imgs / (2 * num_yticks)))
    plt.tick_params(axis='both', which='major', labelsize=23)
    save_as = 'mtta_item_size.pdf'
    plt.savefig(os.path.join(save_root, save_as))
    plt.clf()

def plot_num_items(num_items_list, problem_types, box_extent , item_size,
                    num_max_train_imgs, num_yticks, colors, legend,
                    load_root, save_root):
    plt.figure(figsize=(8.5, 8))
    for iprob in range(len(problem_types)):
        root_dir = os.path.join(load_root, 'PSVRT' + problem_types[iprob] + '_summary')
        # OVER NUM ITEMS
        x_list = []
        y_list = []
        for i, ni in enumerate(num_items_list):
            target_dir = os.path.join(root_dir, str(box_extent), str(ni), str(item_size))
            files = find_files([], dirs=[target_dir], contains=['.txt'])
            x_list.append(i)
            tta_list = []
            for j, fn in enumerate(files):
                reader = open(fn, 'r')
                line1 = reader.readline()
                line2 = reader.readline()
                line3 = reader.readline()
                tta = str(line3.split()[-1])
                if tta == 'inf':
                    tta = num_max_train_imgs
                else:
                    tta = int(tta)
                tta_list.append(tta)
            if np.min(tta_list)<num_max_train_imgs:
                #y_list.append(np.min(tta_list))
                y_list.append(None)
            else:
                y_list.append(np.min(tta_list))
                #y_list.append(None)
        plt.plot(np.array(x_list), np.array(y_list), c=colors[iprob],
                 marker='o', linestyle='--', linewidth=5.0, markersize=14, alpha=0.5, label=problem_types[iprob])
    if legend:
        plt.legend(loc=2, fontsize=23)
    plt.xlabel('Number of items', fontsize=23)
    # plt.ylabel('Training images (millions)', fontsize=14)
    xtick_markers = [str(ni) for ni in num_items_list]
    ytick_vals = np.linspace(0, num_max_train_imgs, num_yticks)
    ytick_markers = [str(int(k / 1000000)) for k in ytick_vals]
    ytick_markers[-1] = 'Never'
    plt.xticks(range(len(num_items_list)), xtick_markers)
    plt.yticks(ytick_vals, ytick_markers)
    plt.ylim((-num_max_train_imgs / (2 * num_yticks), num_max_train_imgs + num_max_train_imgs / (2 * num_yticks)))
    plt.tick_params(axis='both', which='major', labelsize=23)
    save_as = 'mtta_num_items.pdf'
    plt.savefig(os.path.join(save_root, save_as))
    plt.clf()

if __name__ == '__main__':

    box_extent_list = [[30,30],[60,60],[90,90],[120,120],[150,150],[180,180]]##
    default_be = [60,60]
    item_size_list = [[3,3],[4,4],[5,5],[6,6],[7,7]]##
    default_is = [4,4]
    num_items_list = [2,3,4,5,6] #[2] #
    default_ni = 2
    num_max_train_imgs = 20000000
    num_yticks = 5
    problem_types = ['SD'] #['SD','SR']
    colors = ['r','g','b']
    load_root = '/Users/junkyungkim/Desktop/PSVRT_Results_figs_siam'
    save_root = os.path.join(load_root, 'figsave')

    plot_box_extent(box_extent_list, problem_types, item_size=default_is, num_items=default_ni,
                    num_max_train_imgs=num_max_train_imgs, num_yticks=num_yticks, colors=colors, legend=True,
                    load_root=load_root, save_root=save_root)
    #plot_item_size(item_size_list, problem_types, box_extent=default_be, num_items=default_ni,
    #               num_max_train_imgs=num_max_train_imgs, num_yticks=num_yticks, colors=colors, legend=True,
    #                load_root=load_root, save_root=save_root)
    #plot_num_items(num_items_list, problem_types, box_extent=default_be, item_size=default_is,
    #               num_max_train_imgs=num_max_train_imgs, num_yticks=num_yticks, colors=colors, legend=False,
    #                load_root=load_root, save_root=save_root)