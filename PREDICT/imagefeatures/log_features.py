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
import PREDICT.imagefeatures.histogram_features as hf
import PREDICT.helpers.image_helper as ih
# import imagefeatures.histogram_features as hf
# import helpers.image_helper as ih
import SimpleITK as sitk
import numpy as np

N_BINS = 50


def get_log_features(image, mask, parameters=dict()):
    # Alternatively, one could use the pxehancement function
    image = sitk.GetImageFromArray(image)

    im_size = image.GetSize()

    if "sigma" in parameters.keys():
        sigma = parameters["sigma"]
    else:
        sigma = [1, 5, 10]

    # Make a dummy
    LoG_features = list()
    LoG_labels = list()

    # Create LoG filter object
    LoGFilter = sitk.LaplacianRecursiveGaussianImageFilter()
    LoGFilter.SetNormalizeAcrossScale(True)
    for i_index, i_sigma in enumerate(sigma):
        boolean = False
        # Compute LoG Filter image
        for elem in im_size:
            if elem < 4:
                boolean = True
        if boolean:
            LoG_image = np.zeros((im_size[2], im_size[1], im_size[0]))
        else:
            LoGFilter.SetSigma(i_sigma)
            LoG_image = LoGFilter.Execute(image)
            LoG_image = sitk.GetArrayFromImage(LoG_image)

        # Get histogram features of LoG image for full tumor
        masked_voxels = ih.get_masked_voxels(LoG_image, mask)
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'logf_') for l in histogram_labels]
        LoG_features.extend(histogram_features)
        final_feature_names = [feature_name + '_sigma' + str(i_sigma) for feature_name in histogram_labels]
        LoG_labels.extend(final_feature_names)

    return LoG_features, LoG_labels
