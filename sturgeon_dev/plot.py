import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sturgeon_dev.utils import get_best_checkpoint, rchop

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def plot_during_training_performance(log_file, metric, metric_dir, output_dir = None):

    train_color = 'red'
    validation_color = 'blue'

    log = pd.read_csv(log_file, header = 0, index_col = None)

    n = get_best_checkpoint(log_file, metric, metric_dir)
    for column in log.columns:
        if column.endswith('.train'):
            
            steps = log['step'].tolist()
            train_values = log[column].tolist()
            validation_values = log[column.replace('.train', '.val')].tolist()
            best_step = n['step'].tolist()[0]

            best_train = np.array(train_values)[np.array(steps) == best_step]
            best_val = np.array(validation_values)[np.array(steps) == best_step]

            plt.figure(figsize = (10, 5))
            plt.plot(steps, train_values, color = train_color, label = 'Train')
            plt.plot(steps, validation_values, color = validation_color, label = 'Validation')
            plt.axvline(best_step, color = 'black', linestyle = '--')
            plt.xlabel('Training steps')
            plt.ylabel(rchop(column, '.train'))
            plt.legend()
            plt.title('Training = {t}\n Validation = {v}'.format(t = best_train, v = best_val))
            plt.tight_layout()

            if output_dir:
                plt.savefig(
                    os.path.join(output_dir, rchop(column, '.train')+'.pdf'),
                    format = 'pdf',
                    bbox_inches = "tight"
                )

def plot_calibration_adjustment(old_scores, new_scores, bin_size, decoding_dict, color_dict, output_file = None):


    fig, axes = plt.subplots(10, 10, sharex = True, sharey = True, figsize = (15, 15))
    axes = axes.flatten()
    for idx, label in decoding_dict.items():

        axes[idx].bar(
            x = np.arange(0, 1, bin_size), 
            height = old_scores[idx, :], 
            width = bin_size, 
            align = 'edge', 
            edgecolor = 'black',
            color = 'white',
        )
        axes[idx].bar(
            x = np.arange(0, 1, bin_size), 
            height = new_scores[idx, :], 
            width = bin_size, 
            align = 'edge', 
            edgecolor = 'black', 
            color = color_dict[label],
            alpha = 0.5
        )
        axes[idx].plot([0, 1], [0, 1], color = 'black', linestyle='--')
        axes[idx].set_title(label, fontsize=6)
        
    fig.tight_layout()
    if output_file:
        fig.savefig(
            output_file,
            format = 'pdf',
            bbox_inches = "tight",
        )

def plot_calibration_change(old_scores, new_scores, bin_size, decoding_dict, color_dict, output_file = None):

    num_classes = len(decoding_dict)
    expected = np.reshape(np.tile(np.arange(0, 1, bin_size) + 0.05, num_classes), (num_classes, int(1/bin_size)))
    x = np.arange(0, num_classes, 1)
    y = np.mean(np.abs(new_scores - expected), 1) - np.mean(np.abs(old_scores - expected), 1)

    plt.figure(figsize = (20, 5))
    barlist = plt.bar(
        x,
        y,
        edgecolor = 'black',
    )
    for i, bar in enumerate(barlist):
        bar.set_color(color_dict[decoding_dict[i]])
        bar.set_edgecolor('black')

    plt.xticks(np.arange(0, num_classes, 1), list(decoding_dict.values()), rotation = 90)

    plt.ylabel('Calibrated error change after calibration\n\n(Negative is better)')
    if output_file:
        plt.savefig(
            output_file,
            format = 'pdf',
            bbox_inches = "tight"
        )
    plt.show()

def general_cm(df, encoding_dict, decoding_dict, output_file = None):

    class_labels = list(decoding_dict.values())

    cm = np.zeros((len(class_labels), len(class_labels)), dtype=float)

    encoded_diagnostic_labels = list(map(encoding_dict.get, df['Label']))
    predictions = np.array(df[class_labels]).argmax(-1)

    for t, p in zip(encoded_diagnostic_labels, predictions):
        cm[t, p] += 1
    cm_perc = np.transpose(np.transpose(cm)/np.sum(cm, axis = 1))

    fig = plt.figure(figsize = (20, 20))

    colormap = 'binary'

    arr = cm_perc
    plt.imshow(
        X = arr, 
        vmin = 0, 
        vmax = 1,
        cmap = colormap
    )

    for k in range(arr.shape[0]):
        for l in range(arr.shape[1]):
            v = arr[k, l]
            if v < 0.5:
                c = 'black'
            else:
                c = 'white'
            if v < 0.01:
                continue
            v = str(round(v, 2))
            if v != '1.0':
                v = v[1:]
            else:
                v = '1'
            plt.text(l, k, v, color = c, ha="center", va="center")


    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.xticks(np.arange(0, len(class_labels), 1), class_labels, rotation = 90)
    plt.yticks(np.arange(0, len(class_labels), 1), class_labels)
    if output_file:
        plt.savefig(output_file, format = 'pdf', bbox_inches = "tight")
    plt.show()

def diagnostics_cm(df, encoding_dict, decoding_dict, output_file = None):

    class_labels = list(decoding_dict.values())
    superlabels = list()
    for cl in class_labels:
        cl = cl.split(' - ')
        superlabels.append(cl[0])

    supersplits = list()
    prev_l = None
    for i, sl in enumerate(superlabels):
        if prev_l is None:
            prev_l = sl
        if prev_l != sl:
            prev_l = sl
            supersplits.append(i)

    cm_labels = list()
    for cl in class_labels:
        cl = cl.split(' - ')
        cm_labels.append(cl[1]+'-'+cl[2])

    labelpadding = np.zeros((len(np.unique(superlabels)),))
    for i in range(len(labelpadding)):
        if i % 2 != 0:
            labelpadding[i] = 6
        else:
            labelpadding[i] = 12

    x = np.array(supersplits + [len(class_labels)]) - np.array([0] + supersplits).tolist()
    y = np.array(supersplits + [len(class_labels)]) - np.array([0] + supersplits).tolist()
    gs_kw = dict(
        width_ratios=x, 
        height_ratios=y,
    )

    cuts = np.array([0] + supersplits+ [len(class_labels)])

    cm = np.zeros((len(class_labels), len(class_labels)), dtype=float)

    encoded_diagnostic_labels = list(map(encoding_dict.get, df['Label']))
    predictions = np.array(df[class_labels]).argmax(-1)

    for t, p in zip(encoded_diagnostic_labels, predictions):
        cm[t, p] += 1
    cm_perc = np.transpose(np.transpose(cm)/np.sum(cm, axis = 1))

    fig, axes = plt.subplots(nrows = len(supersplits) + 1, ncols = len(supersplits) + 1, sharex = False, sharey = False, gridspec_kw=gs_kw, figsize=(25, 25), dpi=300)

    for i in range(len(cuts)-1):
        for j in range(len(cuts)-1):

            colormap = 'binary'
            
            arr = cm_perc[cuts[i]:cuts[i+1], cuts[j]:cuts[j+1]]
            axes[i, j].imshow(
            X = arr, 
            vmin = 0, 
            vmax = 1,
            cmap = colormap
            )

            for k in range(arr.shape[0]):
                for l in range(arr.shape[1]):
                    v = arr[k, l]
                    if v < 0.5:
                        c = 'black'
                    else:
                        c = 'white'
                    if v < 0.01:
                        continue
                    v = str(round(v, 2))
                    if v != '1.0':
                        v = v[1:]
                    else:
                        v = '1'
                    axes[i, j].text(l, k, v, color = c, ha="center", va="center")

            if i < len(cuts)-2:
                axes[i, j].xaxis.set_visible(False)
            if j > 0:
                axes[i, j].yaxis.set_visible(False)

            axes[i, j].set_xlabel(np.unique(superlabels)[j], labelpad = labelpadding[j])
            axes[i, j].set_ylabel(np.unique(superlabels)[i], labelpad = labelpadding[i])

            if j == 0:
                lab = cm_labels[cuts[i]:cuts[i+1]]
                axes[i, j].set_yticks(np.arange(0, len(lab), 1))
                axes[i, j].set_yticklabels(lab)

            if i == len(cuts)-2:
                lab = cm_labels[cuts[j]:cuts[j+1]]
                axes[i, j].set_xticks(np.arange(0, len(lab), 1))
                axes[i, j].set_xticklabels(lab, rotation=90)

    fig.supxlabel('Prediction')
    fig.supylabel('Truth')
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, format = 'pdf', bbox_inches = "tight")
    fig.show()

def plot_metrics(test_df, decoding_dict, color_dict, output_file = None):

    starts = [0, 20000, 40000]
    ends = [20000, 40000, np.inf]
    all_metrics = list()
    for s, n in zip(starts, ends):

        df = test_df[(test_df['NSites'] >= s) & (test_df['NSites'] < n)]
        df = df.sort_values('Label')
        df = df.reset_index()

        metrics = dict()

        scores = np.array(df[list(decoding_dict.values())])
        for i, label in decoding_dict.items():
            select = np.ones((len(decoding_dict),), dtype=bool)
            select[i] = False
            non_label_score = scores[:, select].max(1)
            label_score = np.array(df[label])
            y = np.array(df['Label'] == label)

            final_score = label_score > non_label_score

            tp = np.sum((final_score.astype(int) + y.astype(int)) == 2)
            tn = np.sum((final_score.astype(int) + y.astype(int)) == 0)
            
            f = (final_score.astype(int) + y.astype(int)) == 1
            fp = np.sum((final_score.astype(int) + f.astype(int)) == 2)
            fn = np.sum((final_score.astype(int) + f.astype(int)) == 1) - tp
            
            assert tp+tn+fp+fn == len(df)
            
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*tp/(2*tp+fp+fn)

            metrics[label] = {
                "Recall":recall,
                "Precision": precision,
                "F1 score": f1
            }
        all_metrics.append(metrics)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize = (20, 10))
    markers = ['v', 'o', '^']

    for shape_i, metrics in enumerate(all_metrics):

        for x, label in enumerate(metrics.keys()):

            for j, metric in enumerate(metrics[label]):

                value = metrics[label][metric]
                axes[j].scatter(x, value, s = 70, marker=markers[shape_i], color = color_dict[label], edgecolor='black')
                axes[j].set_ylabel(metric)
            
    axes[-1].set_xticks(np.arange(0, len(decoding_dict), 1))
    axes[-1].set_xticklabels(list(decoding_dict.values()), rotation = 90)
    fig.tight_layout()
    axes[j].set_xlabel('<20k probes (down triangle), 20-40k probes (circle), >40k probes (up triangle)')
    if output_file:
        fig.savefig(output_file, format = 'pdf', bbox_inches = "tight")
    fig.show()
