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

import pandas as pd


def get_semantic_features(data, patientID):
    patient_ID = data['Patient']
    semantics_names = data.keys()

    # Get index of current patient
    index = None
    for i, s in enumerate(patient_ID):
        if s in patientID:
            index = i

    if index is None:
        raise ValueError("No semantic features found for " + patientID)

    # Extract all labels
    panda_labels = list()
    semantics_features = list()
    for name in semantics_names:
        if name != 'Patient':
            panda_labels.append('semf_' + name)
            semantics_features.append(data[name][index])

    panda_dict = dict(zip(panda_labels, semantics_features))

    semantics_dict = dict()
    semantics_dict['all'] = pd.Series(panda_dict)

    semantic_features = pd.Series(semantics_dict)

    return semantic_features
