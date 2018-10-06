#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Play import CSVtoMIDI
from Tonality import *

import numpy as np
import pandas as pd

from collections import Counter, OrderedDict

import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

logger.warning('Protocol problem: %s', 'connection reset')


class Score(object):

	pass


class Read(Score):

	def __init__(self, name_file_midi):

		logger.info('INFO: %s', 'Creating class '+self.__class__.__name__)
		# Read midi file
		self.music_df = pd.read_csv(name_file_midi)
		# Calculate attribute name_note
		self.calculate_name_note()
		self.divide_music_with_most_granular_tick()

	def get_music_data(self):
		return self.music_df

	def calculate_name_note(self):

		# Names can be F3#, A2#, then return F# and A#
		# Names can be F3, A2, then return F and A
		self.music_df['name_note'] = \
		(self.music_df
		 .apply(lambda x:
		        _get_note_name_without_octave(x['fullNoteOctave']), axis=1)
		 )


	def get_most_common_note(self):

		return (self
		      .music_df
		      # Group by name of the note (A, F#, ...)
		      .groupby(['name_note'])
		      # Take the total length that note has been played
		      .agg({'dur_ms':sum})
		      .rename(columns={'dur_ms': 'cum_duration'})
		      # Order in descending manner
		      .sort_values(['cum_duration'],ascending=False)
		      .reset_index()
		      # Take the first element and the corresponding column
		      .head(1)['name_note'][0]
		      )


	def get_tonality(self):
		tonalities = \
					 ({'Do':['Do','Re','Mi','Fa','Sol','La','Si'],
		               'Re':['Do#','Re','Mi','Fa#','Sol','La','Si'],
		               #'Reb':['Do','Reb','Mib','Fa','Solb','Lab','Sib'],
		               'Reb':['Do','Do#','Re#','Fa','Fa#','Sol#','La#'],
		               'Mi':['Do#','Re#','Mi','Fa#','Sol#','La','Si'],
		               #'Mib':['Do','Re','Mib','Fa','Sol','Lab','Sib'],
		               'Mib':['Do','Re','Re#','Fa','Sol','Sol#','La#'],
		               'Fa':['Do','Re','Mi','Fa','Sol','La','La#'],
		               # Fa# is the same as Solb
		               #'Fa#':['Do#','Re#','Fa','Fa#','Sol#','La#','Si'],
		               'Sol':['Do','Re','Mi','Fa#','Sol','La','Si'],
		               #'Solb':['Dob','Reb','Mib','Fa','Solb','Lab','Sib'],
		               'Solb':['Si','Do#','Re#','Fa','Fa#','Sol#','La#'],
		               'La':['Do#','Re','Mi','Fa#','Sol#','La','Si'],
		               #'Lab':['Do','Reb','Mib','Fa','Sol','Lab','Sib'],
		               'Lab':['Do','Do#','Re#','Fa','Sol','Sol#','La#'],
		               'Si':['Do#','Re#','Mi','Fa#','Sol#','La#','Si'],
		               #'Sib':['Do','Re','Mib','Fa','Sol','La','Sib'],
		               'Sib':['Do','Re','Re#','Fa','Sol','La','La#'],
		              })

		note_histogram = \
			(self
		      .music_df
		      # Group by name of the note (A, F#, ...)
		      .groupby(['name_note'])
		      # Take the total length that note has been played
		      .agg({'dur_ms':sum})
		      .rename(columns={'dur_ms': 'cum_duration'})
		      # Order in descending manner
		      .sort_values(['cum_duration'],ascending=False)
		      .reset_index()
		      # Take the first element and the corresponding column
		      #.head(10)['name_note'][0:10]
		      ).set_index('name_note')['cum_duration'].to_dict()

		tonality_candidates = {}
		for tonality_name, tonality_scale in tonalities.iteritems():
			tonality_candidates[tonality_name] = np.sum([note_histogram.get(iter_tonality_scale,0) 
			                                            for iter_tonality_scale in tonality_scale])

		return next(iter(OrderedDict(sorted(tonality_candidates.items(), key=lambda t: -t[1]))))	


	def divide_music_with_most_granular_tick(self):

		# Obtain the histograms of ticks
		# Counter({80.0: 946, 0.0: 642, 40.0: 15, 240.0: 8, 60.0: 6, 20.0: 3, 480.0: 2, 120.0: 2, nan: 1, 160.0: 1, 300.0: 1, 360.0: 1})
		hist_ticks = Counter(self.music_df['start_ticks'].diff())
		# Take the minimum_tick different from 0
		# [  0.  20.  40.  60.  80. 120. 160. 240. 300. 360. 480.  nan]
		# In this case, it is 20.
		minimum_tick = int([ticks for ticks in np.sort(hist_ticks.keys()) if ticks > 0][0])

		self.minimum_tick = minimum_tick

		# Divide music with minimum_tick

		self.music_df['num_minimum_ticks'] = \
		((self.music_df['dur_ticks']+1) / minimum_tick)

		granular_music_list = list()

		for index, iter_note in self.music_df.iterrows():
			for iter_num_minimum_ticks in range(int(iter_note['num_minimum_ticks'])):
				granular_music_list.append(([iter_note['start_ticks']+(iter_num_minimum_ticks)*minimum_tick,
				                          iter_note['pitch'],iter_note['velocity'],
				                          iter_note['part'],iter_note['fullNoteOctave'],
				                          iter_note['name_note']]
				                          ))

		self.granular_music_df = (pd.DataFrame(granular_music_list,
		                                  columns = ['start_ticks','pitch','velocity','part',
		                                  'fullNoteOctave','name_note']))



	def get_chord_from_tick(self):
		# Given a sequence of notes such as:
		# 0,0,79,85,90,F6#,96,1
		# 0,0,119,128,66,F4#,96,2
		# 0,0,119,128,61,C4#,96,2
		# 0,0,119,128,58,A3#,96,2
		# 0,0,119,128,54,F3#,96,2
		# 80,86,79,85,94,A6#,96,1
		# 160,172,79,85,85,C6#,96,1
		# 240,258,79,85,90,F6#,96,1
		# 240,258,119,128,66,F4#,96,2
		# 240,258,119,128,63,D4#,96,2
		# 240,258,119,128,59,B3,96,2
		# 240,258,119,128,54,F3#,96,2

		# This method will extract the chords for 0, 80, 160, 240 and so forth.

		return (self.granular_music_df
		      .groupby('start_ticks')
		      .agg({'fullNoteOctave':lambda x: Counter(x),
		           # 'name_note':lambda x: tuple(dict(Counter(x)).keys())}
		           'name_note':lambda x: Counter(x)}
		           )
		      .reset_index()
		      )
		# Do not forget the aggregated time pero granular chord.

	def aggregate_chord_from_tick(self):
		# Given a sequence of notes such as:
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F4#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F4#': 2, u'F3#': 1}

		# The result must be
		# {u'F2#': 5, u'F1#': 5, u'F3#': 5}
		# {u'F2#': 2, u'F4#': 3, u'F3#': 2}

		# First of all calculate chord per ticks
		chord_per_ticks = self.get_chord_from_tick()

		aggregation_criteria = 'fullNoteOctave'

		# See the changes in chord per ticks
		changes_in_chords = chord_per_ticks[aggregation_criteria].diff()
		# Since the first element is Nan, force it to be {}
		# It is {} since we are working with dicts
		changes_in_chords[0] = {}

		# Store the final column into id_aggregation_criteria
		chord_per_ticks['id_aggregation_criteria'] = np.cumsum([len(element)>0 for element in changes_in_chords])



		aggregated_chord_per_ticks = (chord_per_ticks
		      .groupby(['id_aggregation_criteria',
		               chord_per_ticks[aggregation_criteria].map(tuple)])
		      .agg({'start_ticks':['min','max']
		           }
		           )
		      .reset_index()
		      #.sort_values('start_ticks',ascending=False)
		      )

		# Rename the columns
		aggregated_chord_per_ticks.columns = ['_'.join(col) for col in aggregated_chord_per_ticks.columns]

		# Number of notes of the chord according to aggregation_criteria
		aggregated_chord_per_ticks['num_elements_chord'] = \
		aggregated_chord_per_ticks[aggregation_criteria+'_'].apply(len)

		# Length in time (ticks) of the chord
		aggregated_chord_per_ticks['time_length_chord'] = \
		aggregated_chord_per_ticks['start_ticks_max']-aggregated_chord_per_ticks['start_ticks_min']+self.minimum_tick


		# Analyze those notes that are not individual notes
		tonic_chord_candidates = \
		(aggregated_chord_per_ticks
		      #.loc[aggregated_chord_per_ticks['num_elements_chord']==4]
		      .groupby(aggregation_criteria+'_')
		      .agg({'time_length_chord':'sum'})
		      .filter([aggregation_criteria+'_','time_length_chord'])
		      .sort_values('time_length_chord',ascending=False)
		      .reset_index()
		      )
		
		tonic_chord_candidates['n_elmnts_chord'] = tonic_chord_candidates[aggregation_criteria+'_'].apply(len)
		tonic_chord_candidates['imp'] = tonic_chord_candidates['n_elmnts_chord']*100+tonic_chord_candidates['time_length_chord']

		tonic_chord_candidates.columns = ['notes','ticks','num_el','imp']

		# print(tonic_chord_candidates
		#       .groupby('num_el')
		#       .size()
		#       )

		# print(tonic_chord_candidates
		#       .groupby('ticks')
		#       .size()
		#       )

		# print(tonic_chord_candidates
		#       .sort_values(['imp'],ascending=False)
		#       .filter(['notes','num_el','imp','ticks'])
		#       .head(10)
		#       )

		
def _get_note_name_without_octave(fullNoteOctave):
	# Function to get the name, regardless the octave

	notes_dict = {'A':'La','B':'Si','C':'Do','D':'Re','E':'Mi','F':'Fa','G':'Sol'}

	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[2]
	else:
		# Names can be F3, A2, then return F and A
		return notes_dict[fullNoteOctave[0]]

if __name__ == "__main__":
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_1.csv'
	name_file_midi = '../../scores/Albeniz_Asturias.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_5.csv'
	name_file_midi = '../../scores/Debussy_Claire_de_Lune.csv'
	name_file_midi = '../../scores/Schuber_Impromptu_D_899_No_3.csv'
	name_file_midi = '../../scores/Schubert_S560_Schwanengesang_no7.csv'
	name_file_midi = '../../scores/Schubert_Piano_Trio_2nd_Movement.csv'
	name_file_midi = '../../scores/Beethoven_Moonlight_Sonata_third_movement.csv'
	name_file_midi = '../../scores/Bach-Partita_No1_in_Bb_BWV825_7Gigue.csv'
	name_file_midi = '../../scores/Brahms_symphony_2_2.csv' # Si M
	name_file_midi = '../../scores/Brahms_symphony_2_1.csv'
	
	chopin = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	#print(chopin.get_chord_from_tick().filter(['fullNoteOctave']))
	#print(chopin.aggregate_chord_from_tick())
	print(chopin.get_most_common_note())
	print('La tonalidad es: '+chopin.get_tonality())


