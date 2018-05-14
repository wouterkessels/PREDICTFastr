#Check classes of selected features
import pandas as pd
import numpy as np
import os
import sys


def paracheck(parameters):
    output = dict()
    # print parameters

    f = parameters['semantic_features']
    total = float(len(f))
    count_semantic = sum([i == 'True' for i in f])
    ratio_semantic = count_semantic/total
    print("Semantic: " + str(ratio_semantic))
    output['semantic_features'] = ratio_semantic

    f = parameters['patient_features']
    count_patient = sum([i == 'True' for i in f])
    ratio_patient = count_patient/total
    print("patient: " + str(ratio_patient))
    output['patient_features'] = ratio_patient

    f = parameters['orientation_features']
    count_orientation = sum([i == 'True' for i in f])
    ratio_orientation = count_orientation/total
    print("orientation: " + str(ratio_orientation))
    output['orientation_features'] = ratio_orientation

    f = parameters['histogram_features']
    count_histogram = sum([i == 'True' for i in f])
    ratio_histogram = count_histogram/total
    print("histogram: " + str(ratio_histogram))
    output['histogram_features'] = ratio_histogram

    f = parameters['shape_features']
    count_shape = sum([i == 'True' for i in f])
    ratio_shape = count_shape/total
    print("shape: " + str(ratio_shape))
    output['shape_features'] = ratio_shape

    if 'coliage_features' in parameters.keys():
        f = parameters['coliage_features']
        count_coliage = sum([i == 'True' for i in f])
        ratio_coliage = count_coliage/total
        print("coliage: " + str(ratio_coliage))
        output['coliage_features'] = ratio_coliage

    if 'phase_features' in parameters.keys():
        f = parameters['phase_features']
        count_phase = sum([i == 'True' for i in f])
        ratio_phase = count_phase/total
        print("phase: " + str(ratio_phase))
        output['phase_features'] = ratio_phase

    if 'vessel_features' in parameters.keys():
        f = parameters['vessel_features']
        count_vessel = sum([i == 'True' for i in f])
        ratio_vessel = count_vessel/total
        print("vessel: " + str(ratio_vessel))
        output['vessel_features'] = ratio_vessel

    if 'log_features' in parameters.keys():
        f = parameters['log_features']
        count_log = sum([i == 'True' for i in f])
        ratio_log = count_log/total
        print("log: " + str(ratio_log))
        output['log_features'] = ratio_log

    f = parameters['texture_features']
    count_texture_all = sum([i == 'True' for i in f])
    ratio_texture_all = count_texture_all/total
    print("texture_all: " + str(ratio_texture_all))
    output['texture_all_features'] = ratio_texture_all

    count_texture_no = sum([i == 'False' for i in f])
    ratio_texture_no = count_texture_no/total
    print("texture_no: " + str(ratio_texture_no))
    output['texture_no_features'] = ratio_texture_no

    count_texture_Gabor = sum([i == 'Gabor' for i in f])
    ratio_texture_Gabor = count_texture_Gabor/total
    print("texture_Gabor: " + str(ratio_texture_Gabor))
    output['texture_Gabor_features'] = ratio_texture_Gabor

    count_texture_LBP = sum([i == 'LBP' for i in f])
    ratio_texture_LBP = count_texture_LBP/total
    print("texture_LBP: " + str(ratio_texture_LBP))
    output['texture_LBP_features'] = ratio_texture_LBP

    count_texture_GLCM = sum([i == 'GLCM' for i in f])
    ratio_texture_GLCM = count_texture_GLCM/total
    print("texture_GLCM: " + str(ratio_texture_GLCM))
    output['texture_GLCM_features'] = ratio_texture_GLCM

    count_texture_GLRLM = sum([i == 'GLRLM' for i in f])
    ratio_texture_GLRLM = count_texture_GLRLM/total
    print("texture_GLRLM: " + str(ratio_texture_GLRLM))
    output['texture_GLRLM_features'] = ratio_texture_GLRLM

    count_texture_GLSZM = sum([i == 'GLSZM' for i in f])
    ratio_texture_GLSZM = count_texture_GLSZM/total
    print("texture_GLSZM: " + str(ratio_texture_GLSZM))
    output['texture_GLSZM_features'] = ratio_texture_GLSZM

    count_texture_NGTDM = sum([i == 'NGTDM' for i in f])
    ratio_texture_NGTDM = count_texture_NGTDM/total
    print("texture_NGTDM: " + str(ratio_texture_NGTDM))
    output['texture_NGTDM_features'] = ratio_texture_NGTDM

    try:
        f = parameters['degree']
    except:
        parameters['degree'] = 1
        f = parameters['degree']
    print("Polynomial Degree: " + str(np.mean(f)))
    output['polynomial_degree'] = np.mean(f)

    return output

def write_to_file(string, file):
	with open(target_folder + txt_file, 'a') as f:
		f.write(string + '\n')
		f.close

def param1(path, label):
	t = pd.read_hdf(path)
	t = t[label]

	params = dict()
	for c in t.classifiers:
		parameters_all = c.cv_results_['params_all'][0] # First entry is best performing classifier

		for k in parameters_all.keys():
			if k not in params.keys():
				params[k] = list()
			params[k].append(parameters_all[k])

	output = paracheck(params)
	write_to_file(str(output), target_folder + txt_file)
	return output

def param10(path, label):
	t = pd.read_hdf(path)
	t = t[label]

	params = dict()
	top = 10
	for c in t.classifiers:
		for n in range(0, top):
			parameters_all = c.cv_results_['params_all'][n]

			for k in parameters_all.keys():
				if k not in params.keys():
					params[k] = list()
				params[k].append(parameters_all[k])

	output = paracheck(params)
	write_to_file(str(output), target_folder + txt_file)
	return output


def main(target_folder, hdf_file, txt_file, label):
    #Execute
    if os.path.exists(target_folder + txt_file):
    	os.remove(os.path.abspath(target_folder + txt_file))

    print('Best setting:')
    param1(target_folder + hdf_file, label)
    print('Best 10 settings:')
    param10(target_folder + hdf_file, label)
    print('file written to {}{}'.format(target_folder, txt_file))


if __name__ == '__main__':
    #Parameters
    folder_name = sys.argv[1]
    target_folder = '/archive/wkessels/output/{}/'.format(folder_name)
    hdf_file = 'svm_all_0.hdf5'
    txt_file = 'best_settings.txt'
    label = 'MDM2'

    if len(sys.argv) == 2:
        main(target_folder=target_folder, hdf_file=hdf_file, txt_file=txt_file, label=label)
    elif len(sys.argv) == 1:
        raise IOError('No input argument given.')
    else:
        raise IOError('main takes only 1 argument, {} given'.format(len(sys.argv)))
