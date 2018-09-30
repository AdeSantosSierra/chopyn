#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import folders with the code
# At the moment, there is not other way of importing 
sys.path.insert(0, '/Users/adesant3/Documents/Kindergarten/chopyn/chopyn/src/')

from MusicalNote import *

import numpy as np 
import pandas as pd


def test_get_major_scale_do():
	assert ([nota.to_string() for nota in Do().get_major_scale()] == 
	        ['Do','Re','Mi','Fa','Sol','La','Si','Do'])

def test_get_major_scale_do_sharp():
	assert ([nota.to_string() for nota in Do(**{'alteration':'#'}).get_major_scale()] == 
	        ['Do#','Re#','Mi#','Fa#','Sol#','La#','Si#','Do#'])

def test_get_major_scale_do_flat():
	assert ([nota.to_string() for nota in Do(**{'alteration':'b'}).get_major_scale()] == 
	        ['Dob','Reb','Mib','Fab','Solb','Lab','Sib','Dob'])



###############
###    RE
###############

def test_get_major_scale_re():
	assert ([nota.to_string() for nota in Re().get_major_scale()] == 
	        ['Re','Mi','Fa#','Sol','La','Si','Do#','Re'])

def test_get_major_scale_re_sharp():
	assert ([nota.to_string() for nota in Re(**{'alteration':'#'}).get_major_scale()] == 
	        ['Re#','Mi#','Fax','Sol#','La#','Si#','Dox','Re#'])

def test_get_major_scale_re_flat():
	assert ([nota.to_string() for nota in Re(**{'alteration':'b'}).get_major_scale()] == 
	        ['Reb','Mib','Fa','Solb','Lab','Sib','Do','Reb'])


###############
###    MI
###############

def test_get_major_scale_mi():
	assert ([nota.to_string() for nota in Mi().get_major_scale()] == 
	        ['Mi','Fa#','Sol#','La','Si','Do#','Re#','Mi'])

def test_get_major_scale_mi_sharp():
	assert ([nota.to_string() for nota in Mi(**{'alteration':'#'}).get_major_scale()] == 
	        ['Mi#','Fax','Solx','La#','Si#','Dox','Rex','Mi#'])

def test_get_major_scale_mi_flat():
	assert ([nota.to_string() for nota in Mi(**{'alteration':'b'}).get_major_scale()] == 
	        ['Mib','Fa','Sol','Lab','Sib','Do','Re','Mib'])


###############
###    Fa
###############

def test_get_major_scale_fa():
	assert ([nota.to_string() for nota in Fa().get_major_scale()] == 
	        ['Fa','Sol','La','Sib','Do','Re','Mi','Fa'])

def test_get_major_scale_fa_sharp():
	assert ([nota.to_string() for nota in Fa(**{'alteration':'#'}).get_major_scale()] == 
	        ['Fa#','Sol#','La#','Si','Do#','Re#','Mi#','Fa#'])

def test_get_major_scale_fa_flat():
	assert ([nota.to_string() for nota in Fa(**{'alteration':'b'}).get_major_scale()] == 
	        ['Fab','Solb','Lab','Sibb','Dob','Reb','Mib','Fab'])


###############
###    Sol
###############

def test_get_major_scale_sol():
	assert ([nota.to_string() for nota in Sol().get_major_scale()] == 
	        ['Sol','La','Si','Do','Re','Mi','Fa#','Sol'])

def test_get_major_scale_sol_sharp():
	assert ([nota.to_string() for nota in Sol(**{'alteration':'#'}).get_major_scale()] == 
	        ['Sol#','La#','Si#','Do#','Re#','Mi#','Fax','Sol#'])

def test_get_major_scale_sol_flat():
	assert ([nota.to_string() for nota in Sol(**{'alteration':'b'}).get_major_scale()] == 
	        ['Solb','Lab','Sib','Dob','Reb','Mib','Fa','Solb'])


###############
###    La
###############

def test_get_major_scale_la():
	assert ([nota.to_string() for nota in La().get_major_scale()] == 
	        ['La','Si','Do#','Re','Mi','Fa#','Sol#','La'])

def test_get_major_scale_la_sharp():
	assert ([nota.to_string() for nota in La(**{'alteration':'#'}).get_major_scale()] == 
	        ['La#','Si#','Dox','Re#','Mi#','Fax','Solx','La#'])

def test_get_major_scale_la_flat():
	assert ([nota.to_string() for nota in La(**{'alteration':'b'}).get_major_scale()] == 
	        ['Lab','Sib','Do','Reb','Mib','Fa','Sol','Lab'])


###############
###    Si
###############

def test_get_major_scale_si():
	assert ([nota.to_string() for nota in Si().get_major_scale()] == 
	        ['Si','Do#','Re#','Mi','Fa#','Sol#','La#','Si'])

def test_get_major_scale_si_sharp():
	assert ([nota.to_string() for nota in Si(**{'alteration':'#'}).get_major_scale()] == 
	        ['Si#','Dox','Rex','Mi#','Fax','Solx','Lax','Si#'])

def test_get_major_scale_si_flat():
	assert ([nota.to_string() for nota in Si(**{'alteration':'b'}).get_major_scale()] == 
	        ['Sib','Do','Re','Mib','Fa','Sol','La','Sib'])