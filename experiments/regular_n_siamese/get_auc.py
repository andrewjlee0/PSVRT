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

    return files, len(files)

def plot_ata(curves_name_list, curves_color_list, condition_type, conditions_path_list, conditions_name_list,
             default_num_lc_samples, legend,
             load_root, save_root, type):

    plt.figure(figsize=(8.5, 8))
    plt.subplot(1,1,1)
    for i, curv in enumerate(curves_name_list):
        # OVER NUM ITEMS
        x_list = []
        median_list = []
        mean_list = []
        n_unlearned_list = []
        raw_list = []
        raw_paired_list = []
        u_list = []
        d_list = []
        for j, cond in enumerate(conditions_path_list):
            target_dir = os.path.join(load_root, curv, cond)
            files, n = find_files([], dirs=[target_dir], contains=['.npy'])
            x_list.append(j)
            ata_list = []
            ata_list_learned = []
            ata_list_unlearned = []
            print(cond)
            for k, fn in enumerate(files):
                print(fn)
                learning_curve = np.load(fn)
                ## THIS PART FOR CUTTING/FILLING
                # jitter = np.random.randint(low=-1,high=2,size=learning_curve.shape).astype(float)
                # learning_curve += jitter
                # np.save(fn,learning_curve)
                if learning_curve.shape[0] < default_num_lc_samples: #for 10 million
                    remainder = default_num_lc_samples - learning_curve.shape[0]
                    trailing_average = 0.975
                    filler = np.array([trailing_average]*remainder)
                    learning_curve = np.concatenate([learning_curve,filler])
                    ## THIS PART FOR CUTTING/FILLING
                ata = np.mean(learning_curve[:400])
                ata_list.append(ata)
                raw_paired_list.append([j, ata])
                if ata < 0.6:
                    ata_list_unlearned.append(ata)
                else:
                    ata_list_learned.append(ata)
            if type == 'segregated':
                mean_list.append(np.mean(ata_list_learned))
                u_list.append(np.sort(ata_list_learned)[-1])
                d_list.append(np.sort(ata_list_learned)[0])
                n_unlearned_list.append(len(ata_list_unlearned))
            else:
                ata_list_sorted = np.sort(ata_list)
                raw_list.append(ata_list)
                mean_list.append(np.mean(ata_list_sorted))
                median_list.append(np.median(ata_list_sorted))
                u_list.append(ata_list_sorted[int(np.floor(len(ata_list_sorted) * 0.75))])
                d_list.append(ata_list_sorted[int(np.floor(len(ata_list_sorted) * 0.25))])
        print(curves_name_list[i])
        if type =='box_plot':
            plt.boxplot(raw_list, notch=False, sym='+', vert=True, whis=1.5, bootstrap=10000)
        elif type == 'scatter':
            plt.scatter(np.array(raw_paired_list)[:, 0], np.array(raw_paired_list)[:, 1])
        elif type == 'median':
            plt.plot(np.array(x_list), np.array(median_list), c=curves_color_list[i],
                     marker='o', linestyle='--', linewidth=5.0, markersize=14, alpha=0.5, label=curves_name_list[i])
            plt.fill_between(np.array(x_list), np.array(d_list), np.array(u_list), color=curves_color_list[i],
                             alpha=0.2, edgecolor="")
        elif type == 'mean':
            plt.plot(np.array(x_list), np.array(mean_list), c=curves_color_list[i],
                     marker='o', linestyle='--', linewidth=5.0, markersize=14, alpha=0.5, label=curves_name_list[i])
            plt.fill_between(np.array(x_list), np.array(d_list), np.array(u_list), color=curves_color_list[i],
                             alpha=0.2, edgecolor="")
        elif type == 'segregated':
            fig = plt.figure(figsize=(7.5, 6))
            ax1 = fig.add_subplot(111)
            ax1.plot(np.array(x_list), np.array(mean_list), c=curves_color_list[i],
                     marker='o', linestyle='--', linewidth=5.0, markersize=14, alpha=0.8, label=curves_name_list[i])
            ax1.fill_between(np.array(x_list), np.array(d_list), np.array(u_list), color=curves_color_list[i],
                             alpha=0.2, edgecolor=curves_color_list[i], linewidth=0.5)
            ax2 = ax1.twinx()
            ax2.bar(np.array(x_list), np.array(n_unlearned_list), align='center', color='k', width=0.6, alpha=0.3)
    if type == 'segregated':
        ax1.set_xlabel(condition_type, fontsize=16)
        plt.xticks(range(len(conditions_path_list)), conditions_name_list)
        ax1.set_xlim([-0.3, len(conditions_path_list) - 1 + 0.3])
        ax1.set_ylim([0.5, 1])
        ax1.set_ylabel('Mean AULC of $\mathit{learned}$ trials', fontsize=18, color=curves_color_list[i])
        ax1.yaxis.set_tick_params(color=curves_color_list[i],
                                  labelcolor=curves_color_list[i],
                                  labelsize=16)
        ax1.xaxis.set_tick_params(color='k',
                                  labelcolor='k',
                                  labelsize=16)
        ax2.set_ylim([0, 10])
        ax2.set_ylabel('Number of $\mathit{unlearned}$ trials', fontsize=18)
        ax2.yaxis.set_tick_params(color='k',
                                  labelcolor='k',
                                  labelsize=16)

    else:
        plt.xlabel(condition_type, fontsize=23)
        plt.ylabel('Average accuracy', fontsize=23)
        plt.xticks(range(len(conditions_path_list)), conditions_name_list)
        if type == 'box_plot':
            plt.xlim((1-0.3, len(conditions_path_list)+0.3))
        else:
            plt.xlim((-0.3, len(conditions_path_list) - 1 + 0.3))
        plt.ylim((0.5,1))
        if legend:
            plt.legend(loc=3, fontsize=23)

    plt.tick_params(axis='both', which='major', labelsize=14)
    save_as = 'mauc_' + condition_type + '.pdf'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    plt.savefig(os.path.join(save_root, save_as))
    plt.clf()

if __name__ == '__main__':

    box_extent_list = [[30,30],[60,60],[90,90],[120,120],[150,150],[180,180]]##
    default_be = [60,60]
    item_size_list = [[3,3],[4,4],[5,5],[6,6],[7,7]]##
    default_is = [4,4]
    num_items_list = [2,3,4,5,6] #[2] #

    load_root = '/Users/junkyungkim/Desktop/PSVRTSD_summary'
    save_root = os.path.join(load_root, 'figsave')
    curves_name_list = ['SD-Siamese'] # ['SR-Siamese'] #['SD-Siamese'] #['SR-CNN'] # ['SD-CNN (Control)']
    curves_color_list =['tomato'] #['g'] #['tomato'] #['b'] # ['blueviolet']
    plot_type = 'segregated'

    default_num_lc_samples = 400


    condition_type = 'Item Size'
    conditions_path_list = ['[60, 60]/2/[3, 3]',
                            '[60, 60]/2/[4, 4]',
                            '[60, 60]/2/[5, 5]',
                            '[60, 60]/2/[6, 6]',
                            '[60, 60]/2/[7, 7]']
    conditions_name_list = [3, 4, 5, 6, 7]
    legend = True
    plot_ata(curves_name_list, curves_color_list, condition_type, conditions_path_list, conditions_name_list,
             default_num_lc_samples, legend,
             load_root, save_root, plot_type)

    condition_type = 'Image Size'
    conditions_path_list = ['[30, 30]/2/[4, 4]',
                            '[60, 60]/2/[4, 4]',
                            '[90, 90]/2/[4, 4]',
                            '[120, 120]/2/[4, 4]',
                            '[150, 150]/2/[4, 4]',
                            '[180, 180]/2/[4, 4]']
    conditions_name_list = [30, 60, 90, 120, 150, 180]
    legend = False
    plot_ata(curves_name_list, curves_color_list, condition_type, conditions_path_list, conditions_name_list,
             default_num_lc_samples, legend,
             load_root, save_root, plot_type)

    condition_type = 'Num. Items'
    conditions_path_list = ['[60, 60]/2/[4, 4]',
                            '[60, 60]/3/[4, 4]',
                            '[60, 60]/4/[4, 4]',
                            '[60, 60]/5/[4, 4]',
                            '[60, 60]/6/[4, 4]']
    conditions_name_list = [2, 3, 4, 5, 6]
    legend = False
    plot_ata(curves_name_list, curves_color_list, condition_type, conditions_path_list, conditions_name_list,
             default_num_lc_samples, legend,
             load_root, save_root, plot_type)