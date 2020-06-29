# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO


import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

import joblib as jl

# load data pipeline
pipeline = jl.load('processed/data_pipe.sav')

def feat2bvh(feat_file, bvh_file):
    features = np.load(feat_file)['clips']
    print(features.shape)

    # transform the data back to it's original shape
    # note: in a real scenario this is usually done with predicted data   
    # note: some transformations (such as transforming to joint positions) are not inversible
    bvh_data=pipeline.inverse_transform([features])

    # Test to write some of it to file for visualization in blender or motion builder
    writer = BVHWriter()
    with open(bvh_file,'w') as f:
        writer.write(bvh_data[0], f)

feat2bvh("processed/NaturalTalking_002.npz", 'processed/converted2.bvh')
