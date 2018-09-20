#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import folders with the code
# At the moment, there is not other way of importing 
sys.path.insert(0, '/Users/adesant3/Documents/Kindergarten/chopyn/chopyn/src/')

from MusicalNote import *

import numpy as np 
import pandas as pd

# content of test_sample.py
def test_update_pitch_do():
	random_pitch = np.random.randint(50)
	notes_props = {'pitch':random_pitch, 'duration':100, 'intensity': 200, 'timbre': 1}
	do = Do(**notes_props)
	assert do.get_pitch() == 60

def test_update_pitch_re():
	random_pitch = np.random.randint(50)
	notes_props = {'pitch':random_pitch, 'duration':100, 'intensity': 200, 'timbre': 1}
	re = Re(**notes_props)
	assert re.get_pitch() == 62