#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import folders with the code
# At the moment, there is not other way of importing 
sys.path.insert(0, '/Users/adesant3/Documents/Kindergarten/chopyn/chopyn/src/')

from MusicalNote import *
from Tonality import *
from ReadMusic import *

import numpy as np 
import pandas as pd

from collections import Counter, OrderedDict


name_folder = '../scores/'

dict_scores = \
{'Do':'Chopin_Etude_Op_10_n_1.csv',

#'Solb':'Chopin_Etude_Op_10_n_5.csv',
'Reb':'Debussy_Claire_de_Lune.csv',
'Mib':'Schubert_Piano_Trio_2nd_Movement.csv',
#'Solb':'Chopin_Etude_6.csv', # Mib m
#'Mib':'Chopin_Etude_Opus_10_No_12.csv', #Do m
'Mi':'Beethoven_Moonlight_Sonata_third_movement.csv', #Although it is Do# m
#'Mi':'Chopin_Etude_Opus_10_No_3.csv', #Although it is Do# m
'Fa':'Schubert_S560_Schwanengesang_no7.csv', # Although it is Re m
'Sol':'Albeniz_Asturias.csv', # Although, it is Mi m
'Solb':'Schuber_Impromptu_D_899_No_3.csv',
'Si':'Chopin_Etude_Op25_No_6.csv' #Although it is Sol# m
}

def test_tonality_Do():
	assert Read(name_folder+dict_scores['Do']).get_tonality() == 'Do'

def test_tonality_Sol():
	assert Read(name_folder+dict_scores['Sol']).get_tonality() == 'Sol'

def test_tonality_Solb():
	assert Read(name_folder+dict_scores['Solb']).get_tonality() == 'Solb'

def test_tonality_Fa():
	assert Read(name_folder+dict_scores['Fa']).get_tonality() == 'Fa'

def test_tonality_Reb():
	assert Read(name_folder+dict_scores['Reb']).get_tonality() == 'Reb'

def test_tonality_Mib_num1():
	assert Read(name_folder+dict_scores['Mib']).get_tonality() == 'Mib'

def test_tonality_Mib_num2():
	assert Read(name_folder+'Chopin_Etude_6.csv').get_tonality() == 'Solb'

def test_tonality_Mib_num3():
	assert Read(name_folder+'Chopin_Etude_Opus_10_No_12.csv').get_tonality() == 'Mib'

def test_tonality_Mi_num1():
	assert Read(name_folder+dict_scores['Mi']).get_tonality() == 'Mi'

def test_tonality_Mi_num2():
	assert Read(name_folder+'Chopin_Etude_Opus_10_No_3.csv').get_tonality() == 'Mi'

def test_tonality_Si():
	assert Read(name_folder+dict_scores['Si']).get_tonality() == 'Si'