#!/usr/bin/env python

# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import scipy.stats as st
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sys
import compute_CI
import pandas as pd
import os

import PREDICT.genetics.genetic_processing as gp
import PREDICT.plotting.plot_ROC as Pplot

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib2tikz import save as tikz_save
except ImportError:
    print("[PREDICT Warning] Cannot use plot_ROC function, as _tkinter is not installed")

import argparse
from compute_CI import compute_confidence_logit as CIl
from compute_CI import compute_confidence as CI
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import roc_auc_score
import csv
import glob
import natsort
import json
import collections
from PREDICT.processing.fitandscore import fit_and_score
from sklearn.base import clone
import sklearn

ROC_data_folder = '/archive/wkessels/ROC_input_data/'

def write_to_txt(name, data, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + str(name) + '.txt', "w") as f:
        f.write('name = {}\ntype = {}\n\n{}'.format(name, str(type(data)), data))


def plot_single_SVM(prediction, mutation_data, label_type, show_plots=False, show_ROC=False):
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    keys = prediction.keys()
    SVMs = list()
    label = keys[0]
    SVMs = prediction[label]['classifiers']

    Y_test = prediction[label]['Y_test']
    X_test = prediction[label]['X_test']
    Y_train = prediction[label]['X_train']
    Y_score = list()

    # print(len(X_test[0][0]))
    # print(config)
    # X_train = data2['19q']['X_train']
    # Y_train = data2['19q']['Y_train']
    # mutation_data = gp.load_mutation_status(patientinfo, [[label]])
    if type(mutation_data) is not dict:
        if os.path.isfile(mutation_data):
            mutation_data = gp.load_mutation_status(mutation_data, [[label_type]])

    patient_IDs = mutation_data['patient_IDs']
    mutation_label = mutation_data['mutation_label']
    # mutation_name = mutation_data['mutation_name']

    # print(len(SVMs))
    N_iterations = float(len(SVMs))

    # mutation_label = np.asarray(mutation_label)

    sensitivity = list()
    specificity = list()
    precision = list()
    accuracy = list()
    auc = list()
    # auc_train = list()
    f1_score_list = list()

    patient_classification_list = dict()

    for i in range(0, len(Y_test)):
        # print(Y_test[i])
        # if Y_test[i].shape[1] > 1:
        #     # print(Y_test[i])
        #     y_truth = np.prod(Y_test[i][:, 0:2], axis=1)
        # else:
        #     y_truth_test = Y_test[i]
        test_patient_IDs = prediction[label]['patient_ID_test'][i]

        if 'LGG-Radiogenomics-046' in test_patient_IDs:
            wrong_index = np.where(test_patient_IDs == 'LGG-Radiogenomics-046')
            test_patient_IDs = np.delete(test_patient_IDs, wrong_index)
            X_temp = X_test[i]
            print(X_temp.shape)
            X_temp = np.delete(X_test[i], wrong_index, axis=0)
            print(X_temp.shape)

            # X_test.pop(wrong_index[0])

            # print(len(X_test))
        else:
            X_temp = X_test[i]

        test_indices = list()
        for i_ID in test_patient_IDs:
            test_indices.append(np.where(patient_IDs == i_ID)[0][0])

            if i_ID not in patient_classification_list:
                patient_classification_list[i_ID] = dict()
                patient_classification_list[i_ID]['N_test'] = 0
                patient_classification_list[i_ID]['N_correct'] = 0
                patient_classification_list[i_ID]['N_wrong'] = 0

            patient_classification_list[i_ID]['N_test'] += 1

        y_truth = [mutation_label[0][k] for k in test_indices]
        # print(y_truth)
        # print(y_truth_test)
        # print(test_patient_IDs)

        y_predict_1 = SVMs[i].predict(X_temp)

        # print(y_predict_1).shape

        y_prediction = y_predict_1
        # y_prediction = np.prod(y_prediction, axis=0)

        print "Truth: ", y_truth
        print "Prediction: ", y_prediction

        for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
            if i_truth == i_predict:
                patient_classification_list[i_test_ID]['N_correct'] += 1
            else:
                patient_classification_list[i_test_ID]['N_wrong'] += 1

        # print('bla')
        # print(y_truth)
        # print(y_prediction)

        c_mat = confusion_matrix(y_truth, y_prediction)
        TN = c_mat[0, 0]
        FN = c_mat[1, 0]
        TP = c_mat[1, 1]
        FP = c_mat[0, 1]

        if FN == 0 and TP == 0:
            sensitivity.append(0)
        else:
            sensitivity.append(float(TP)/(TP+FN))
        if FP == 0 and TN == 0:
            specificity.append(0)
        else:
            specificity.append(float(TN)/(FP+TN))
        if TP == 0 and FP == 0:
            precision.append(0)
        else:
            precision.append(float(TP)/(TP+FP))
        accuracy.append(accuracy_score(y_truth, y_prediction))
        y_score = SVMs[i].decision_function(X_temp)
        Y_score.append(y_score)
        auc.append(roc_auc_score(y_truth, y_score))
        f1_score_list.append(f1_score(y_truth, y_prediction, average='weighted'))

        # if show_ROC:
        #     ROC_target_folder = '/archive/wkessels/output/ROC_temp/'
        #     if not os.path.exists(ROC_target_folder):
        #         os.makedirs(ROC_target_folder)
        #
        #     luck = [0, 1]
        #
        #     fpr, tpr, _ = roc_curve(y_truth, y_score)
        #     plt.figure()
        #     plt.plot(fpr, tpr, color='blue', label='ROC (AUC = {})'.format(auc[-1]))
        #     plt.plot(luck, luck, '--', color='red', label='luck')
        #     plt.xlabel('1-specificity')
        #     plt.ylabel('sensitivity')
        #     plt.axis([0, 1, 0, 1])
        #     plt.legend()
        #     plt.savefig(ROC_target_folder + 'ROC_cv{}.png'.format(i))
        #     print('Saved ROC figure in {}!'.format(ROC_target_folder))

    # Adjusted according to "Inference for the Generelization error"

    accuracy_mean = np.mean(accuracy)
    S_uj = 1.0 / max((N_iterations - 1), 1) * np.sum((accuracy_mean - accuracy)**2.0)

    print Y_test
    N_1 = float(len(Y_train[0]))
    N_2 = float(len(Y_test[0]))

    print(N_1)
    print(N_2)

    accuracy_var = np.sqrt((1.0/N_iterations + N_2/N_1)*S_uj)
    print(accuracy_var)
    print(np.sqrt(1/N_iterations*S_uj))
    print(st.sem(accuracy))

    stats = dict()
    stats["Accuracy 95%:"] = str(compute_CI.compute_confidence(accuracy, N_1, N_2, 0.95))

    stats["AUC 95%:"] = str(compute_CI.compute_confidence(auc, N_1, N_2, 0.95))

    stats["F1-score 95%:"] = str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, 0.95))

    stats["Precision 95%:"] = str(compute_CI.compute_confidence(precision, N_1, N_2, 0.95))

    stats["Sensitivity 95%: "] = str(compute_CI.compute_confidence(sensitivity, N_1, N_2, 0.95))

    stats["Specificity 95%:"] = str(compute_CI.compute_confidence(specificity, N_1, N_2, 0.95))

    print("Accuracy 95%:" + str(compute_CI.compute_confidence(accuracy, N_1, N_2, 0.95)))

    print("AUC 95%:" + str(compute_CI.compute_confidence(auc, N_1, N_2, 0.95)))

    print("F1-score 95%:" + str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, 0.95)))

    print("Precision 95%:" + str(compute_CI.compute_confidence(precision, N_1, N_2, 0.95)))

    print("Sensitivity 95%: " + str(compute_CI.compute_confidence(sensitivity, N_1, N_2, 0.95)))

    print("Specificity 95%:" + str(compute_CI.compute_confidence(specificity, N_1, N_2, 0.95)))

    what_to_print = ['always', 'mostly']
    for what in what_to_print:
        if what == 'always':
            alwaysright = dict()
            alwayswrong = dict()
            for i_ID in patient_classification_list:
                percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

                # print(i_ID + ' , ' + str(patient_classification_list[i_ID]['N_test']) + ' : ' + str(percentage_right) + '\n')
                if percentage_right == 1.0:
                    label = mutation_label[0][np.where(i_ID == patient_IDs)]
                    label = label[0][0]
                    alwaysright[i_ID] = label
                    # alwaysright.append(('{} ({})').format(i_ID, label))
                    print(("Always Right: {}, label {}").format(i_ID, label))

                if percentage_right == 0:
                    label = mutation_label[0][np.where(i_ID == patient_IDs)].tolist()
                    label = label[0][0]
                    alwayswrong[i_ID] = label
                    # alwayswrong.append(('{} ({})').format(i_ID, label))
                    print(("Always Wrong: {}, label {}").format(i_ID, label))

            stats["Always right"] = alwaysright
            stats["Always wrong"] = alwayswrong
        elif what == 'mostly':
            margin = float(0.2)
            min_right = float(1-margin) #for mostly right
            max_right = float(margin) #for mostly wrong
            mostlyright = dict()
            mostlywrong = dict()

            for i_ID in patient_classification_list:
                percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

                if percentage_right > min_right:
                    label = mutation_label[0][np.where(i_ID == patient_IDs)]
                    label = label[0][0]
                    mostlyright[i_ID] = [label, "{}%".format(100*percentage_right)]
                    print(("Mostly Right: {}, label {}, percentage: {}%").format(i_ID, label, 100*percentage_right))

                if percentage_right < max_right:
                    label = mutation_label[0][np.where(i_ID == patient_IDs)].tolist()
                    label = label[0][0]
                    mostlywrong[i_ID] = [label, "{}%".format(100*percentage_right)]
                    print(("Mostly Wrong: {}, label {}, percentage: {}%").format(i_ID, label, 100*percentage_right))

            stats["Mostly right"] = mostlyright
            stats["Mostly wrong"] = mostlywrong
        else:
            raise IOError('Unknown argument given...')

    if show_plots:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.boxplot(accuracy)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Accuracy')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(auc)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('AUC')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(precision)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(sensitivity)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Sensitivity')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.boxplot(specificity)
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Specificity')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()

    # Save Y_score values
    Y_score_dict = dict()
    for j in range(len(Y_score)):
        Y_score_dict['CV_{}'.format(j)] = Y_score[j]
    Y_score = pd.DataFrame(Y_score_dict)
    Y_score.to_hdf('/archive/wkessels/output/Lipo_SVM/Y_score.hdf5', 'Y_score')

    # write_to_txt('Y_test', Y_test, ROC_data_folder)
    # write_to_txt('X_test', X_test, ROC_data_folder)
    # write_to_txt('Y_train', Y_train, ROC_data_folder)
    # write_to_txt('mutation_data', mutation_data, ROC_data_folder)
    # write_to_txt('patient_IDs', patient_IDs, ROC_data_folder)
    # write_to_txt('mutation_label',mutation_label, ROC_data_folder)
    # write_to_txt('y_truth', y_truth, ROC_data_folder)
    # write_to_txt('y_prediction', y_prediction, ROC_data_folder)
    # write_to_txt('y_score', y_score, ROC_data_folder)
    # write_to_txt('N_1', N_1, ROC_data_folder)
    # write_to_txt('N_2', N_2, ROC_data_folder)
    # write_to_txt('stats', stats, ROC_data_folder)

    return stats


def plot_multi_SVM(prediction, mutation_data, label_type, show_plots=False,
                   key=None, n_classifiers=[1], outputfolder=None):
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    keys = prediction.keys()
    SVMs = list()
    if key is None:
        label = keys[0]
    else:
        label = key
    SVMs = prediction[label]['classifiers']

    Y_test = prediction[label]['Y_test']
    X_test = prediction[label]['X_test']
    X_train = prediction[label]['X_train']
    Y_train = prediction[label]['Y_train']
    test_patient_IDs = prediction[label]['patient_ID_test']
    train_patient_IDs = prediction[label]['patient_ID_train']
    feature_labels = prediction[label]['feature_labels']

    # print(len(X_test[0][0]))
    # print(config)
    # X_train = data2['19q']['X_train']
    # Y_train = data2['19q']['Y_train']
    # mutation_data = gp.load_mutation_status(patientinfo, [[label]])
    if type(mutation_data) is not dict:
        if os.path.isfile(mutation_data):
            label_data = gp.load_mutation_status(mutation_data, [[label_type]])

    patient_IDs = label_data['patient_IDs']
    mutation_label = label_data['mutation_label']

    # print(len(SVMs))
    N_iterations = float(len(SVMs))

    # mutation_label = np.asarray(mutation_label)

    for n_class in n_classifiers:
        # output_json = os.path.join(outputfolder, ('performance_{}.json').format(str(n_class)))

        sensitivity = list()
        specificity = list()
        precision = list()
        accuracy = list()
        auc = list()
        # auc_train = list()
        f1_score_list = list()

        patient_classification_list = dict()

        trained_classifiers = list()

        y_score = list()
        y_test = list()
        pid_test = list()
        y_predict = list()

        # csvfile = os.path.join(outputfolder, ('scores_{}.csv').format(str(n_class)))
        # towrite = list()
        #
        # csvfile_plain = os.path.join(outputfolder, ('scores_plain_{}.csv').format(str(n_class)))
        # towrite_plain = list()

        empty_scores = {k: '' for k in natsort.natsorted(patient_IDs)}
        empty_scores = collections.OrderedDict(sorted(empty_scores.items()))
        # towrite.append(["Patient"] + empty_scores.keys())
        params = dict()
        for num, s in enumerate(SVMs):
            scores = empty_scores.copy()
            print("Processing {} / {}.").format(str(num + 1), str(len(SVMs)))
            trained_classifiers.append(s)

            # Extract test info
            test_patient_IDs_temp = test_patient_IDs[num]
            train_patient_IDs_temp = train_patient_IDs[num]
            X_train_temp = X_train[num]
            Y_train_temp = Y_train[num]
            X_test_temp = X_test[num]
            Y_test_temp = Y_test[num]

            # Extract sample size
            N_1 = float(len(train_patient_IDs_temp))
            N_2 = float(len(test_patient_IDs_temp))

            test_indices = list()
            for i_ID in test_patient_IDs_temp:
                test_indices.append(np.where(patient_IDs == i_ID)[0][0])

                if i_ID not in patient_classification_list:
                    patient_classification_list[i_ID] = dict()
                    patient_classification_list[i_ID]['N_test'] = 0
                    patient_classification_list[i_ID]['N_correct'] = 0
                    patient_classification_list[i_ID]['N_wrong'] = 0

                patient_classification_list[i_ID]['N_test'] += 1

            # y_truth = [mutation_label[0][k] for k in test_indices]
            # FIXME: order can be switched, need to find a smart fix
            # 1 for normal, 0 for KM
            # y_truth = [mutation_label[0][k][0] for k in test_indices]
            y_truth = Y_test_temp

            # Predict using the top N classifiers
            results = s.cv_results_['rank_test_score']
            indices = range(0, len(results))
            sortedindices = [x for _, x in sorted(zip(results, indices))]
            sortedindices = sortedindices[0:n_class]
            y_prediction = np.zeros([n_class, len(y_truth)])
            y_score = np.zeros([n_class, len(y_truth)])

            # Get some base objects required
            base_estimator = s.estimator
            y_train = Y_train_temp
            y_train_prediction = np.zeros([n_class, len(y_train)])
            scorer = s.scorer_
            train = np.asarray(range(0, len(y_train)))
            test = train # This is in order to use the full training dataset to train the model

            # Remove the NaN features
            X_notnan = X_train_temp[:]
            for pnum, (pid, x) in enumerate(zip(train_patient_IDs_temp, X_train_temp)):
                for fnum, (f, fid) in enumerate(zip(x, feature_labels)):
                    if np.isnan(f):
                        print("[PREDICT WARNING] NaN found, patient {}, label {}. Replacing with zero.").format(pid, fid)
                        # Note: X is a list of lists, hence we cannot index the element directly
                        features_notnan = x[:]
                        features_notnan[fnum] = 0
                        X_notnan[pnum] = features_notnan

            X_train_temp = X_notnan[:]
            X_train_temp = [(x, feature_labels) for x in X_train_temp]

            X_notnan = X_test_temp[:]
            for pnum, (pid, x) in enumerate(zip(test_patient_IDs_temp, X_test_temp)):
                for fnum, (f, fid) in enumerate(zip(x, feature_labels)):
                    if np.isnan(f):
                        print("[PREDICT WARNING] NaN found, patient {}, label {}. Replacing with zero.").format(pid, fid)
                        # Note: X is a list of lists, hence we cannot index the element directly
                        features_notnan = x[:]
                        features_notnan[fnum] = 0
                        X_notnan[pnum] = features_notnan

            X_test_temp = X_notnan[:]
            # X_test_temp = [(x, feature_labels) for x in X_test_temp]

            # NOTE: need to build this in the SearchCVFastr Object
            for i, index in enumerate(sortedindices):
                print("Processing number {} of {} classifiers.").format(str(i + 1), str(n_class))
                X_testtemp = X_test_temp[:]

                # Get the parameters from the index
                parameters_est = s.cv_results_['params'][index]
                parameters_all = s.cv_results_['params_all'][index]

                print parameters_all
                print s.cv_results_['mean_test_score'][index]

                # NOTE: kernel parameter can be unicode
                kernel = str(parameters_est[u'kernel'])
                del parameters_est[u'kernel']
                del parameters_all[u'kernel']
                parameters_est['kernel'] = kernel
                parameters_all['kernel'] = kernel

                # Refit a classifier using the settings given
                print("Refitting classifier with best settings.")
                # Only when using fastr this is an entry
                if 'Number' in parameters_est.keys():
                    del parameters_est['Number']

                best_estimator = clone(base_estimator).set_params(**parameters_est)

                # ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler =\
                #     fit_and_score(best_estimator, X_train, y_train, scorer,
                #                   train, test, True, parameters_all,
                #                   t.fit_params,
                #                   t.return_train_score,
                #                   True, True, True,
                #                   t.error_score)

                ret, GroupSel, VarSel, SelectModel, _, scaler =\
                    fit_and_score(estimator=best_estimator,
                                  X=X_train_temp,
                                  y=y_train,
                                  scorer=scorer,
                                  train=train, test=test,
                                  verbose=True,
                                  para=parameters_all,
                                  fit_params=s.fit_params,
                                  return_train_score=s.return_train_score,
                                  return_n_test_samples=True,
                                  return_times=True,
                                  return_parameters=True,
                                  error_score=s.error_score)

                X = [x[0] for x in X_train_temp]
                if GroupSel is not None:
                    X = GroupSel.transform(X)
                    X_testtemp = GroupSel.transform(X_testtemp)

                if SelectModel is not None:
                    X = SelectModel.transform(X)
                    X_testtemp = SelectModel.transform(X_testtemp)

                if VarSel is not None:
                    X = VarSel.transform(X)
                    X_testtemp = VarSel.transform(X_testtemp)

                if scaler is not None:
                    X = scaler.transform(X)
                    X_testtemp = scaler.transform(X_testtemp)

                try:
                    if y_train is not None:
                        best_estimator.fit(X, y_train, **s.fit_params)
                    else:
                        best_estimator.fit(X, **s.fit_params)

                    # Predict the posterios using the fitted classifier for the training set
                    print("Evaluating performance on training set.")
                    if hasattr(best_estimator, 'predict_proba'):
                        probabilities = best_estimator.predict_proba(X)
                        y_train_prediction[i, :] = probabilities[:, 1]
                    else:
                        # Regression has no probabilities
                        probabilities = best_estimator.predict(X)
                        y_train_prediction[i, :] = probabilities[:]

                    # Predict the posterios using the fitted classifier for the test set
                    print("Evaluating performance on test set.")
                    if hasattr(best_estimator, 'predict_proba'):
                        probabilities = best_estimator.predict_proba(X_testtemp)
                        y_prediction[i, :] = probabilities[:, 1]
                    else:
                        # Regression has no probabilities
                        probabilities = best_estimator.predict(X_testtemp)
                        y_prediction[i, :] = probabilities[:]

                    if type(s.estimator) == sklearn.svm.classes.SVC:
                        y_score[i, :] = best_estimator.decision_function(X_testtemp)
                    else:
                        y_score[i, :] = best_estimator.decision_function(X_testtemp)[:, 0]

                except ValueError:
                    # R2 score was set to zero previously
                    y_train_prediction[i, :] = np.asarray([0.5]*len(X))
                    y_prediction[i, :] = np.asarray([0.5]*len(X_testtemp))
                    y_score[i, :] = np.asarray([0.5]*len(X_testtemp))
                    probabilities = []

                # Add number parameter settings
                for k in parameters_all.keys():
                    if k not in params.keys():
                        params[k] = list()
                    params[k].append(parameters_all[k])

                # Save some memory
                del best_estimator, X, X_testtemp, ret, GroupSel, VarSel, SelectModel, scaler, parameters_est, parameters_all, probabilities

            # Take mean over posteriors of top n
            y_train_prediction_m = np.mean(y_train_prediction, axis=0)
            y_prediction_m = np.mean(y_prediction, axis=0)

            # NOTE: Not sure if this is best way to compute AUC
            y_score = y_prediction_m

            if type(s.estimator) == sklearn.svm.classes.SVC:
                # Look for optimal F1 performance on training set
                thresholds = np.arange(0, 1, 0.01)
                f1_scores = list()
                y_train_prediction = np.zeros(y_train_prediction_m.shape)
                for t in thresholds:
                    for ip, y in enumerate(y_train_prediction_m):
                        if y > t:
                            y_train_prediction[ip] = 1
                        else:
                            y_train_prediction[ip] = 0

                    f1_scores.append(f1_score(y_train_prediction, y_train, average='weighted'))

                # Use best threshold to determine test score
                best_index = np.argmax(f1_scores)
                best_thresh = thresholds[best_index]
                best_thresh = 0.5
                y_prediction = np.zeros(y_prediction_m.shape)
                for ip, y in enumerate(y_prediction_m):
                    if y > best_thresh:
                        y_prediction[ip] = 1
                    else:
                        y_prediction[ip] = 0

                # y_prediction = t.predict(X_temp)

                y_prediction = [min(max(y, 0), 1) for y in y_prediction]
            else:
                y_prediction = y_prediction_m
                y_prediction = [min(max(y, 0), 1) for y in y_prediction]

            # NOTE: start of old function part

            print "Truth: ", y_truth
            print "Prediction: ", y_prediction

            for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs_temp):
                if i_truth == i_predict:
                    patient_classification_list[i_test_ID]['N_correct'] += 1
                else:
                    patient_classification_list[i_test_ID]['N_wrong'] += 1

            # print('bla')
            # print(y_truth)
            # print(y_prediction)

            c_mat = confusion_matrix(y_truth, y_prediction)
            TN = c_mat[0, 0]
            FN = c_mat[1, 0]
            TP = c_mat[1, 1]
            FP = c_mat[0, 1]

            if FN == 0 and TP == 0:
                sensitivity.append(0)
            else:
                sensitivity.append(float(TP)/(TP+FN))
            if FP == 0 and TN == 0:
                specificity.append(0)
            else:
                specificity.append(float(TN)/(FP+TN))
            if TP == 0 and FP == 0:
                precision.append(0)
            else:
                precision.append(float(TP)/(TP+FP))
            accuracy.append(accuracy_score(y_truth, y_prediction))
            auc.append(roc_auc_score(y_truth, y_score))
            f1_score_list.append(f1_score(y_truth, y_prediction, average='weighted'))

            # Adjusted according to "Inference for the Generelization error"

            accuracy_mean = np.mean(accuracy)
            S_uj = 1.0 / max((N_iterations - 1), 1) * np.sum((accuracy_mean - accuracy)**2.0)

            print Y_test
            N_1 = float(len(Y_train[0]))
            N_2 = float(len(Y_test[0]))

            print(N_1)
            print(N_2)

            accuracy_var = np.sqrt((1.0/N_iterations + N_2/N_1)*S_uj)
            print(accuracy_var)
            print(np.sqrt(1/N_iterations*S_uj))
            print(st.sem(accuracy))

        stats = dict()
        stats["Accuracy 95%:"] = str(compute_CI.compute_confidence(accuracy, N_1, N_2, 0.95))

        stats["AUC 95%:"] = str(compute_CI.compute_confidence(auc, N_1, N_2, 0.95))

        stats["F1-score 95%:"] = str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, 0.95))

        stats["Precision 95%:"] = str(compute_CI.compute_confidence(precision, N_1, N_2, 0.95))

        stats["Sensitivity 95%: "] = str(compute_CI.compute_confidence(sensitivity, N_1, N_2, 0.95))

        stats["Specificity 95%:"] = str(compute_CI.compute_confidence(specificity, N_1, N_2, 0.95))

        print("Accuracy 95%:" + str(compute_CI.compute_confidence(accuracy, N_1, N_2, 0.95)))

        print("AUC 95%:" + str(compute_CI.compute_confidence(auc, N_1, N_2, 0.95)))

        print("F1-score 95%:" + str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, 0.95)))

        print("Precision 95%:" + str(compute_CI.compute_confidence(precision, N_1, N_2, 0.95)))

        print("Sensitivity 95%: " + str(compute_CI.compute_confidence(sensitivity, N_1, N_2, 0.95)))

        print("Specificity 95%:" + str(compute_CI.compute_confidence(specificity, N_1, N_2, 0.95)))

        alwaysright = dict()
        alwayswrong = dict()
        for i_ID in patient_classification_list:
            percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

            # print(i_ID + ' , ' + str(patient_classification_list[i_ID]['N_test']) + ' : ' + str(percentage_right) + '\n')
            if percentage_right == 1.0:
                label = mutation_label[0][np.where(i_ID == patient_IDs)]
                label = label[0][0]
                alwaysright[i_ID] = label
                # alwaysright.append(('{} ({})').format(i_ID, label))
                print(("Always Right: {}, label {}").format(i_ID, label))

            if percentage_right == 0:
                label = mutation_label[0][np.where(i_ID == patient_IDs)].tolist()
                label = label[0][0]
                alwayswrong[i_ID] = label
                # alwayswrong.append(('{} ({})').format(i_ID, label))
                print(("Always Wrong: {}, label {}").format(i_ID, label))

        stats["Always right"] = alwaysright
        stats["Always wrong"] = alwayswrong

        if show_plots:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.boxplot(accuracy)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Accuracy')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(auc)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('AUC')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(precision)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Precision')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(sensitivity)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Sensitivity')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(specificity)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Specificity')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

    return stats


def main():
    if len(sys.argv) == 1:
        print("TODO: Put in an example")
    elif len(sys.argv) != 3:
        raise IOError("This function accepts two arguments")
    else:
        prediction = sys.argv[1]
        patientinfo = sys.argv[2]
    plot_single_SVM(prediction, patientinfo)


if __name__ == '__main__':
    main()
