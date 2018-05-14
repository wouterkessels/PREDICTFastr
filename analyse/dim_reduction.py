from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import pandas
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.mlab import PCA as mlabPCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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
                #Feature 288 sometimes contains NaN
                del dic[patient][0][287]

                end = False
                with open(sem_file, 'r') as semantics:
                    content = semantics.readlines()
                    while not end:
                        for line in content:
                            if patient in line:
                                line = line.strip()
                                dic[patient].append(line[-1])
                                end = True

    #Return dictionary with patient as key and features as list[0] and class as list[1]
    return dic


def features_data(dic):
    features = []
    for key in dic.keys():
        features.append(dic[key][0])
    return features


def clsfs_data(dic):
    clsfs = []
    for key in dic.keys():
        clsfs.append(dic[key][1])
    return clsfs


def scaling(X, method):
    if method == 'z-score':
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    elif method == 'minmax':
        scaler = MinMaxScaler().fit(X)
        X = scaler.transform(X)
    elif method == 'None':
        pass
    return X



def do_PCA(X, Y, path, pca, plotting='linear', scaling='None'):
    pca.fit(X)
    X_2D = pca.transform(X)
    features = {}
    features['PC1'] = X_2D[:, 0]
    features['PC2'] = X_2D[:, 1]

    #Add classification
    features['PC1_0'] = [] #lipoma, no mutation
    features['PC1_1'] = [] #liposarcoma, MDM2 mutation
    features['PC2_0'] = [] #lipoma, no mutation
    features['PC2_1'] = [] #liposarcoma, MDM2 mutation
    for i in range(len(X)):
        if Y[i] == '0':
            features['PC1_0'].append(X_2D[i, 0])
            features['PC2_0'].append(X_2D[i, 1])
        else:
            features['PC1_1'].append(X_2D[i, 0])
            features['PC2_1'].append(X_2D[i, 1])

    #Plot PC1 and PC2
    plt.figure()
    if plotting == 'linear':
        plt.plot(features['PC1_0'], features['PC2_0'], 'o', color='green', label='Lipoma')
        plt.plot(features['PC1_1'], features['PC2_1'], '^', color='red', label='WDLPS')
        plt.legend()
    elif plotting == 'loglog':
        plt.loglog(features['PC1_0'], features['PC2_0'], 'o', color='green', label='Lipoma')
        plt.loglog(features['PC1_1'], features['PC2_1'], '^', color='red', label='WDLPS')
        plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if scaling == 'None':
        #plt.axis([-3e10, 0, -0.3e10, -0.15e10])
        pca_img_name = 'PCA_no_scaling.png'
    else:
        pca_img_name = 'PCA_{}_scaling.png'.format(scaling)
    plt.savefig(path + pca_img_name)

    #Extract eigenvalues
    evalues = pca.explained_variance_
    norm_evalues = evalues*(float(100)/sum(evalues))

    #Plot eigenvalues
    plt.figure()
    if scaling == 'None':
        plt.semilogy(range(1, len(norm_evalues) +1), norm_evalues, 'o')
    else:
        plt.plot(range(1, len(norm_evalues) +1), norm_evalues, 'o')
    plt.xlabel('Principal component')
    plt.ylabel('Normalized eigenvalue')
    if scaling == 'None':
        ev_img_name = 'eigenvalues_no_scaling.png'
    else:
        ev_img_name = 'eigenvalues_{}_scaling.png'.format(scaling)
    plt.savefig(path + ev_img_name)

    #Extract eigenvectors
    evectors = pca.components_

    if scaling == 'None':
        print('non-scaled PCA plots saved to {}'.format(path))
    else:
        print('{}-scaled PCA plots saved to {}'.format(scaling, path))
    return {'Eigenvalues': evalues, 'Eigenvectors': evectors, 'Normalized_eigenvalues': norm_evalues}


def do_tSNE(X, Y, tsne, path, scaling=False):
    X = np.matrix(X)
    randperm = np.random.permutation(X.shape[1])
    X_2D = tsne.fit_transform(X)

    xs = X_2D[:,0]
    ys = X_2D[:,1]

    labels = ['Lipoma', 'WDLPS']
    colors = ['g', 'r']
    shapes = ['o', '^']

    #Plot t-SNE
    plt.figure()
    for i, j, c, l in zip(range(len(xs)), range(len(ys)), range(len(Y)), range(len(Y))):
        plt.scatter(xs[i], ys[i], marker=shapes[int(round(float(Y[c])))], c=colors[int(round(float(Y[c])))], label=labels[int(round(float(Y[l])))])
    #plt.legend()

    if scaling == 'None':
        tSNE_img_name = 'tSNE_no_scaling.png'
        print('non-scaled t-SNE plots saved to {}'.format(path))
    else:
        tSNE_img_name = 'tSNE_{}_scaling.png'.format(scaling)
        print('{}-scaled t-SNE plots saved to {}'.format(scaling, path))
    plt.savefig(path + tSNE_img_name)


def main(target_folder):
    #Parameters
    semantics_file = '/archive/wkessels/input/Patientdata/pinfo_Lipo.txt'

    feature_scaling = {}
    feature_scaling['None'] = True
    feature_scaling['z-score'] = True
    feature_scaling['minmax'] = False

    pca_plotting = 'linear'
    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2)


    #Execute
    target_path = '/archive/wkessels/output/{}/dim_reduction/'.format(target_folder)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    all_data = get_data(target_path + '/../', semantics_file)
    X = features_data(all_data)
    Y = clsfs_data(all_data)
    for method in feature_scaling.keys():
        if feature_scaling[method]:
            scaled_X = scaling(X=X, method=method)
            PCA_result = do_PCA(X=scaled_X, Y=Y, pca=pca, path=target_path, plotting=pca_plotting, scaling=method)
            eigenvalues = PCA_result['Eigenvalues']
            eigenvectors = PCA_result['Eigenvectors']
            tSNE_result = do_tSNE(X=scaled_X, Y=Y, tsne=tsne, path=target_path, scaling=method)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 1:
        raise IOError('No input argument given.')
    else:
        raise IOError('main takes only 1 argument, {} given'.format(len(sys.argv)))
