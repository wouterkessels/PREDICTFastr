from scipy.stats import mannwhitneyu
import os
import sys
import pandas
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from operator import itemgetter


def get_data(hdf_path, sem_file):
    hdf_files = glob.glob(hdf_path + '*.hdf5')
    dic = {}
    for hdf_file in hdf_files:
        if hdf_file[-14:] != 'svm_all_0.hdf5':
            if hdf_file[-12:] != 'Y_score.hdf5':
                patient = hdf_file[hdf_file.find('features_')+9:-7]
                dic[patient] = []
                hdf_data = pandas.read_hdf(hdf_file)
                dic[patient].append(hdf_data.feature_values)

                end = False
                with open(sem_file, 'r') as semantics:
                    content = semantics.readlines()
                    while not end:
                        for line in content:
                            if patient in line:
                                line = line.strip()
                                dic[patient].append(line[-1])
                                end = True

    hdf_data = pandas.read_hdf(hdf_path + 'svm_all_0.hdf5')
    feature_labels = hdf_data.MDM2.feature_labels
    for i in range(len(feature_labels)):
        if 'GLCMMS' in feature_labels[i]:
            label = feature_labels[i]
            feature_labels[i] = label.replace('GLCMMS', 'GLCM')

    #Return (1) dictionary with patient as key and features as list[0] and class as list[1] and (2) list with feature labels
    return dic, feature_labels


def features_data(dic):
    features = []
    for key in dic.keys():
        features.append(dic[key][0])
    return np.asarray(features)


def clsfs_data(dic):
    clsfs = []
    for key in dic.keys():
        clsfs.append(dic[key][1])
    return np.asarray(clsfs)


def get_group_inds(labels):
    groups = ['sf_', 'of_', 'pf_', 'semf_', 'hf_', 'Gabor', 'GLCM', 'GLRLM', 'GLSZM', 'NGTDM', 'LBP', 'vf', 'logf_', 'phasef_']
    names = ['Shape', 'Orientation', 'Patient', 'Semantic', 'Histogram', 'Gabor', 'GLCM', 'GLRLM', 'GLSZM', 'NGTDM', 'LBP', 'Vessel', 'LoG', 'Phase']
    inds = dict()
    for group in groups:
        inds[group] = []
        for label in labels:
            if group in label:
                inds[group].append(labels.index(label))
    return inds, groups, names


def plot_p(P, labels, target_folder):
    group_inds, groups, names = get_group_inds(labels)
    nums = range(1, len(groups)+1)
    colours = plt.cm.rainbow(np.linspace(0, 1, len(groups)))

    plt.figure()
    for i in range(len(P)):
        for group in groups:
            if group in labels[i]:
                plt.scatter(i, P[i], color=colours[groups.index(group)])

    #Plot dots that will define the legend
    for i in range(len(groups)):
        if names[i] != 'Patient':
            plt.scatter(nums[groups.index(group)], P[group_inds[group][0]],
                        color=colours[i], label=names[i])

    plt.plot([0, len(P)], [0.05, 0.05], '--', color='red')

    plt.xlabel('Feature group')
    #plt.yscale("log")
    plt.ylabel("p-value")
    plt.legend(loc=2)
    img_name = 'mannwhitneyu.png'
    plt.savefig(target_folder + img_name)
    print('Mann-Whitney U-test figure saved to {}'.format(target_folder + img_name))


def mannwhitney(X, Y):
    P = []
    for i in range(len(X[0])):
        x = []
        for patient_values in X:
            x.append(patient_values[i])
        x = np.asarray(x)
        stat, p = mannwhitneyu(x, Y)
        P.append(p)
    return P


def group_lengths(labels):
    group_inds, groups, names = get_group_inds(labels)
    dic = dict()
    for group in groups:
        dic[group] = 0
        for label in labels:
            if group in label:
                dic[group] += 1
    return dic


def p_stats(P, labels, threshold=0.05):
    bonferroni = float(threshold)/float(len(P))

    P = np.asarray(P)
    group_inds, groups, names = get_group_inds(labels)

    mean = np.mean(P)
    std = np.std(P)
    var = np.var(P)

    P_u_thresh = []
    labels_u_thresh = []
    for p, label in zip(P, labels):
        if p < bonferroni:
            P_u_thresh.append(p)
            labels_u_thresh.append(label)
    p_u_thresh = len(P_u_thresh)

    gr_lengths = group_lengths(labels)
    dic = dict()
    for group in groups:
        dic[group] = [0, 0, 0, 0]
        for label in labels_u_thresh:
            if group in label:
                dic[group][0] += 1
                dic[group][1] = float(dic[group][0])/float(gr_lengths[group])

    for group in groups:
        dic[group][2] = np.mean(np.asarray(P[group_inds[group]]))
        dic[group][3] = np.std(np.asarray(P[group_inds[group]]))

    inds = np.argsort(P)
    print inds
    P_sorted = P[inds]
    print P_sorted
    labels_sorted = np.asarray(labels)[inds]
    sort = zip(P_sorted, labels_sorted)

    return mean, std, var, p_u_thresh, dic, sort


def main(folder):
    #Parameters
    folder = '/archive/wkessels/output/{}/'.format(folder)
    target_folder = folder + 'mannwhitney/'
    semantics_file = '/archive/wkessels/input/Patientdata/pinfo_Lipo.txt'
    threshold = 0.05

    #Load data
    all_data, X_labels = get_data(folder, semantics_file)
    X = features_data(all_data)
    X = np.absolute(X)
    Y = clsfs_data(all_data)

    #Execute
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    P = mannwhitney(X, Y)
    plot_p(P, X_labels, target_folder)
    p_mean, p_std, p_var, p_u_thresh, group_p_scores, sort = p_stats(P, X_labels, threshold)
    print p_mean
    print p_std
    print p_var
    print p_u_thresh
    print group_p_scores
    print sort[0:10]



if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 1:
        raise IOError('No input argument given.')
    else:
        raise IOError('main takes only 1 argument, {} given'.format(len(sys.argv)))
