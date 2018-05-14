import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from PREDICT.plotting.compute_CI import compute_confidence as CI
import os
import sys
from pandas import DataFrame
import pandas as pd


def get_Y_test(path):
    f = pd.read_hdf(path + 'svm_all_0.hdf5')
    return f['MDM2']['Y_test']


def get_Y_train(path):
    f = pd.read_hdf(path + 'svm_all_0.hdf5')
    return f['MDM2']['Y_train']


def get_Y_score(path):
    try:
        f = pd.read_hdf(path + 'Y_score.hdf5')
        lst = list()
        for i in range(len(f.keys())):
            lst.append(np.asarray(f['CV_{}'.format(i)]))
        return lst
    except:
        print('Cannot print ROC for no Y_score.hdf5 was found!')


def new_ROC(yt, ys):
    L = np.asarray(yt)
    L = np.int_(L)
    f = np.asarray(ys)
    inds = f.argsort()
    Lsorted = L[inds]
    f = f[inds]

    FP = 0
    TP = 0
    fpr = list()
    tpr = list()
    thresholds = list()
    fprev = -np.inf
    i = 0
    N = float(np.bincount(L)[0])
    P = float(np.bincount(L)[1])

    while i < len(Lsorted):
        if f[i] != fprev:
            fpr.append(1 - FP/N)
            tpr.append(1 - TP/P)
            thresholds.append(f[i])
            fprev = f[i]
        if Lsorted[i] == 1:
            TP += 1
        else:
            FP += 1
        i += 1
    return fpr[::-1], tpr[::-1], thresholds[::-1]


def ROC_thresholding(fprt, tprt, thresholds, nsamples=21):
    T = list()
    for t in thresholds:
        T.extend(t)
    T = sorted(T)
    nrocs = len(fprt)
    fpr = np.zeros((nsamples, nrocs))
    tpr = np.zeros((nsamples, nrocs))
    tsamples = np.linspace(0, len(T) - 1, nsamples)
    th = list()
    for n, tidx in enumerate(tsamples):
        tidx = int(tidx)
        th.append(T[tidx])
        for i in range(0, nrocs):
            idx = 0
            while float(thresholds[i][idx]) > float(T[tidx]) and idx < (len(thresholds[i]) - 1):
                idx += 1
            fpr[n, i] = fprt[i][idx]
            tpr[n, i] = tprt[i][idx]

    return fpr, tpr, th


def plot_CI(i, CIs_tpr, CIs_tpr_means, CIs_fpr, CIs_fpr_means):
    if CIs_tpr[i, 1] <= 1:
        ymax = CIs_tpr[i, 1]
    else:
        ymax = 1

    if CIs_tpr[i, 0] <= 0:
        ymin = 0
    else:
        ymin = CIs_tpr[i, 0]

    if CIs_tpr_means[i] <= 1:
        ymean = CIs_tpr_means[i]
    else:
        ymean = 1

    if CIs_fpr[i, 1] <= 1:
        xmax = CIs_fpr[i, 1]
    else:
        xmax = 1

    if CIs_fpr[i, 0] <= 0:
        xmin = 0
    else:
        xmin = CIs_fpr[i, 0]

    if CIs_fpr_means[i] <= 1:
        xmean = CIs_fpr_means[i]
    else:
        xmean = 1

    return ymax, ymin, ymean, xmax, xmin, xmean


def plot_ROC(Y_test, Y_score, N1, N2, path, alpha=0.95, tsamples=40, plot_conf=True):
    target_path = path + 'ROC.png'

    fprt = list()
    tprt = list()
    roc_auc = list()
    thresholds = list()

    for yt, ys in zip(Y_test, Y_score):
        fpr_temp, tpr_temp, thresholds_temp = new_ROC(yt, ys)

        #roc_auc.append(auc(fpr_temp, tpr_temp))
        roc_auc.append(roc_auc_score(yt, ys))
        fprt.append(fpr_temp)
        tprt.append(tpr_temp)
        thresholds.append(thresholds_temp)

    fpr, tpr, th = ROC_thresholding(fprt, tprt, thresholds, tsamples)

    CIs_tpr = list()
    CIs_fpr = list()
    for i in range(0, tsamples):
        cit_fpr = CI(fpr[i, :], N1, N2, alpha)
        CIs_fpr.append([cit_fpr[0], cit_fpr[1]])
        cit_tpr = CI(tpr[i, :], N1, N2, alpha)
        CIs_tpr.append([cit_tpr[0], cit_tpr[1]])

    CIs_tpr = np.asarray(CIs_tpr)
    CIs_fpr = np.asarray(CIs_fpr)
    CIs_tpr_means = np.mean(CIs_tpr, axis=1)
    CIs_fpr_means = np.mean(CIs_fpr, axis=1)

    roc_auc = CI(roc_auc, N1, N2, alpha)

    plt.figure()
    lw = 2
    # plt.plot(CIs_fpr_means, CIs_tpr_means, color='blue',
    #              lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))
    plt.plot(CIs_fpr_means, CIs_tpr_means, color='blue', lw=lw, label='ROC curve')

    if plot_conf:
        for i in range(0, len(CIs_fpr_means)):
            ymax, ymin, ymean, xmax, xmin, xmean = plot_CI(i, CIs_tpr, CIs_tpr_means, CIs_fpr, CIs_fpr_means)
            plt.plot([xmin, xmax],
                     [ymean, ymean],
                     color='black', alpha=0.15)
            plt.plot([xmean, xmean],
                     [ymin, ymax],
                     color='black', alpha=0.15)

    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', label='chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.savefig(target_path)
    print 'ROC curve saved to ' + target_path

    return CIs_fpr_means, CIs_tpr_means


def main(folder):
    #Parameters
    target_folder = '/archive/wkessels/output/{}/'.format(folder)
    Y_test = get_Y_test(target_folder)
    Y_train = get_Y_train(target_folder)
    Y_score = get_Y_score(target_folder)
    n_samples = 40
    plot_conf = False

    #Execute
    N1 = len(Y_train[0])
    N2 = len(Y_test[0])
    fpr, tpr = plot_ROC(Y_test, Y_score, N1, N2, target_folder, tsamples=n_samples, plot_conf=False)

    return fpr, tpr


if __name__ == '__main__':
    if len(sys.argv) == 2:
        fpr, tpr = main(sys.argv[1])
    elif len(sys.argv) == 1:
        raise IOError('No input argument given.')
    else:
        raise IOError('main takes exactly 1 argument, {} given'.format(len(sys.argv)-1))
