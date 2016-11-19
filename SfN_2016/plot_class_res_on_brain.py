from guidata import qthelpers # needed tp fix: ValueError: API 'QString' has already been set to version 1
from mayavi import mlab
from surfer import Surface, Brain
import numpy as np
import ram_data_helpers
import matplotlib
import os
# import RAM_plotBrain
import pdb
from ptsa.data.readers.TalReader import TalReader
os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'

