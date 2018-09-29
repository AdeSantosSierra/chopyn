#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Play import CSVtoMIDI
from Tonality import *

import numpy as np


class Score(object):

	pass


class Read(Score):

	def __init__(self, name_file_midi):

		
		midi_file = open(name_file_midi, 'r')

		print(midi_file)


if __name__ == "__main__":
	name_file_midi = '../../scores/Chopn_Etude_Op_10_n_1.csv'
	Read(name_file_midi)


