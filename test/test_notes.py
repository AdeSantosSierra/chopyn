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
def test_get_pitch():
	random_pitch = np.random.randint(50)
	notes_props = {'pitch':random_pitch, 'duration':100, 'intensity': 200, 'timbre': 1}
	do = Do(**notes_props)
	assert (do.get_pitch() != random_pitch)

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

########################################
#
# Test get_note_from_interval_do
#
########################################

def test_get_note_from_interval_do_3M():
	assert Do().get_note_from_interval('3M').to_string() == 'Mi'

def test_get_note_from_interval_do_3m():
	assert Do().get_note_from_interval('3m').to_string() == 'Mib'

def test_get_note_from_interval_do_2M():
	assert Do().get_note_from_interval('2M').to_string() == 'Re'

def test_get_note_from_interval_do_2m():
	assert Do().get_note_from_interval('2m').to_string() == 'Reb'

def test_get_note_from_interval_do_5P():
	assert Do().get_note_from_interval('5P').to_string() == 'Sol'


########################################
#
# Test get_note_from_interval_do_sharp
#
########################################

def test_get_note_from_interval_do_sharp_3M():
	assert Do(**{'alteration':'#'}).get_note_from_interval('3M').to_string() == 'Mi#'

def test_get_note_from_interval_do_sharp_3m():
	assert Do(**{'alteration':'#'}).get_note_from_interval('3m').to_string() == 'Mi'

def test_get_note_from_interval_do_sharp_2M():
	assert Do(**{'alteration':'#'}).get_note_from_interval('2M').to_string() == 'Re#'

def test_get_note_from_interval_do_sharp_2m():
	assert Do(**{'alteration':'#'}).get_note_from_interval('2m').to_string() == 'Re'

def test_get_note_from_interval_do_sharp_5P():
	assert Do(**{'alteration':'#'}).get_note_from_interval('5P').to_string() == 'Sol#'


########################################
#
# Test get_note_from_interval_do_flat
#
########################################

def test_get_note_from_interval_do_flat_3M():
	assert Do(**{'alteration':'b'}).get_note_from_interval('3M').to_string() == 'Mib'

def test_get_note_from_interval_do_flat_3m():
	assert Do(**{'alteration':'b'}).get_note_from_interval('3m').to_string() == 'Mibb'

def test_get_note_from_interval_do_flat_2M():
	assert Do(**{'alteration':'b'}).get_note_from_interval('2M').to_string() == 'Reb'

def test_get_note_from_interval_do_flat_2m():
	assert Do(**{'alteration':'b'}).get_note_from_interval('2m').to_string() == 'Rebb'

def test_get_note_from_interval_do_flat_5P():
	assert Do(**{'alteration':'b'}).get_note_from_interval('5P').to_string() == 'Solb'




########################################
#
# Test get_note_from_interval_re
#
########################################

def test_get_note_from_interval_re_3M():
	assert Re().get_note_from_interval('3M').to_string() == 'Fa#'

def test_get_note_from_interval_re_3m():
	assert Re().get_note_from_interval('3m').to_string() == 'Fa'

def test_get_note_from_interval_re_2M():
	assert Re().get_note_from_interval('2M').to_string() == 'Mi'

def test_get_note_from_interval_re_2m():
	assert Re().get_note_from_interval('2m').to_string() == 'Mib'

def test_get_note_from_interval_re_5P():
	assert Re().get_note_from_interval('5P').to_string() == 'La'


########################################
#
# Test get_note_from_interval_re_sharp
#
########################################

def test_get_note_from_interval_re_sharp_3M():
	assert Re(**{'alteration':'#'}).get_note_from_interval('3M').to_string() == 'Fax'

def test_get_note_from_interval_re_sharp_3m():
	assert Re(**{'alteration':'#'}).get_note_from_interval('3m').to_string() == 'Fa#'

def test_get_note_from_interval_re_sharp_2M():
	assert Re(**{'alteration':'#'}).get_note_from_interval('2M').to_string() == 'Mi#'

def test_get_note_from_interval_re_sharp_2m():
	assert Re(**{'alteration':'#'}).get_note_from_interval('2m').to_string() == 'Mi'

def test_get_note_from_interval_re_sharp_5P():
	assert Re(**{'alteration':'#'}).get_note_from_interval('5P').to_string() == 'La#'


########################################
#
# Test get_note_from_interval_re_flat
#
########################################

def test_get_note_from_interval_re_flat_3M():
	assert Re(**{'alteration':'b'}).get_note_from_interval('3M').to_string() == 'Fa'

def test_get_note_from_interval_re_flat_3m():
	assert Re(**{'alteration':'b'}).get_note_from_interval('3m').to_string() == 'Fab'

def test_get_note_from_interval_re_flat_2M():
	assert Re(**{'alteration':'b'}).get_note_from_interval('2M').to_string() == 'Mib'

def test_get_note_from_interval_re_flat_2m():
	assert Re(**{'alteration':'b'}).get_note_from_interval('2m').to_string() == 'Mibb'

def test_get_note_from_interval_re_flat_5P():
	assert Re(**{'alteration':'b'}).get_note_from_interval('5P').to_string() == 'Lab'



########################################
#
# Test get_note_from_interval_mi
#
########################################

def test_get_note_from_interval_mi_3M():
	assert Mi().get_note_from_interval('3M').to_string() == 'Sol#'

def test_get_note_from_interval_mi_3m():
	assert Mi().get_note_from_interval('3m').to_string() == 'Sol'

def test_get_note_from_interval_mi_2M():
	assert Mi().get_note_from_interval('2M').to_string() == 'Fa#'

def test_get_note_from_interval_mi_2m():
	assert Mi().get_note_from_interval('2m').to_string() == 'Fa'

def test_get_note_from_interval_mi_5P():
	assert Mi().get_note_from_interval('5P').to_string() == 'Si'


########################################
#
# Test get_note_from_interval_mi_sharp
#
########################################

def test_get_note_from_interval_mi_sharp_3M():
	assert Mi(**{'alteration':'#'}).get_note_from_interval('3M').to_string() == 'Solx'

def test_get_note_from_interval_mi_sharp_3m():
	assert Mi(**{'alteration':'#'}).get_note_from_interval('3m').to_string() == 'Sol#'

def test_get_note_from_interval_mi_sharp_2M():
	assert Mi(**{'alteration':'#'}).get_note_from_interval('2M').to_string() == 'Fax'

def test_get_note_from_interval_mi_sharp_2m():
	assert Mi(**{'alteration':'#'}).get_note_from_interval('2m').to_string() == 'Fa#'

def test_get_note_from_interval_mi_sharp_5P():
	assert Mi(**{'alteration':'#'}).get_note_from_interval('5P').to_string() == 'Si#'


########################################
#
# Test get_note_from_interval_mi_flat
#
########################################

def test_get_note_from_interval_mi_flat_3M():
	assert Mi(**{'alteration':'b'}).get_note_from_interval('3M').to_string() == 'Sol'

def test_get_note_from_interval_mi_flat_3m():
	assert Mi(**{'alteration':'b'}).get_note_from_interval('3m').to_string() == 'Solb'

def test_get_note_from_interval_mi_flat_2M():
	assert Mi(**{'alteration':'b'}).get_note_from_interval('2M').to_string() == 'Fa'

def test_get_note_from_interval_mi_flat_2m():
	assert Mi(**{'alteration':'b'}).get_note_from_interval('2m').to_string() == 'Fab'

def test_get_note_from_interval_mi_flat_5P():
	assert Mi(**{'alteration':'b'}).get_note_from_interval('5P').to_string() == 'Sib'