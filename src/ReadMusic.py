#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Polyphony import *
from Play import CSVtoMIDI
from Tonality import *

import numpy as np
import pandas as pd

from collections import Counter, OrderedDict
import itertools

import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

logger.warning('Protocol problem: %s', 'connection reset')


import sys
# import folders with the code
# At the moment, there is not other way of importing 

# Importing files from c-rnn-gan to read music
# sys.path.insert(0, '/Users/adesant3/Documents/Kindergarten/chopyn/c-rnn-gan')
# from music_data_utils import *



class Score(object):

	pass


class Read(Score):

	def __init__(self, name_file_midi):

		logger.info('INFO: %s', 'Creating class '+self.__class__.__name__)

		self.map_tonic_with_scale = \
					 ({'Do':['Do','Re','Mi','Fa','Sol','La','Si'],
		               'Re':['Do#','Re','Mi','Fa#','Sol','La','Si'],
		               'Reb':['Reb','Mib','Fa','Solb','Lab','Sib','Do'],
		               'Mi':['Mi','Fa#','Sol#','La','Si','Do#','Re#'],
		               'Mib':['Mib','Fa','Sol','Lab','Sib','Do','Re'],
		               'Fa':['Fa','Sol','La','Sib','Do','Re','Mi'],
		               # Fa# is the same as Solb
		               'Sol':['Sol','La','Si','Do','Re','Mi','Fa#'],
		               'Solb':['Solb','Lab','Sib','Dob','Reb','Mib','Fa'],
		               'La':['La','Si','Do#','Re','Mi','Fa#','Sol#'],
		               'Lab':['Lab','Sib','Do','Reb','Mib','Fa','Sol'],
		               'Si':['Si','Do#','Re#','Mi','Fa#','Sol#','La#'],
		               'Sib':['Sib','Do','Re','Mib','Fa','Sol','La'],
		              })

		self.map_note_with_alias = \
						{'Do#':'Reb', 'Reb':'Do#',
						 'Re#':'Mib', 'Mib':'Re#',
						 'Fa':'Mi#', 'Mi#':'Fa',
						 'Fa#':'Solb', 'Solb':'Fa#',
						 'Sol#':'Lab',  'Lab':'Sol#',
						 'La#':'Sib',  'Sib':'La#',
						 'Si#':'Do',  'Do':'Si#',
						}

		self.map_grades_with_scale_position = \
						{'I':0,
						 'II':1,
						 'III':2,
						 'IV':3,
						 'V':4,
						 'VI':5,
						 'VII':6
						}


		# Read midi file
		self.music_df = pd.read_csv(name_file_midi)
		# Calculate attribute name_note
		self.calculate_name_note()
		self.divide_music_with_most_granular_tick()

	def get_map_tonic_with_scale(self):
		return self.map_tonic_with_scale

	def get_map_note_with_alias(self):
		return self.map_note_with_alias

	def get_map_grades_with_scale_position(self):
		return self.map_grades_with_scale_position

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

		# Names can be F3#, A2#, then return F# and A#
		# Names can be F3, A2, then return F and A
		self.music_df['octave_name_note'] = \
		(self.music_df
		 .apply(lambda x:
		        _get_note_name_with_octave(x['fullNoteOctave']), axis=1)
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

	def apply_tonality(self):

		map_tonic_with_scale = self.get_map_tonic_with_scale()

		map_note_with_alias = self.get_map_note_with_alias()

		grades = ['I','II','III','IV','V','VI','VII']

		# Return the aggregates of the chord and their sequence
		agg_criteria = 'octave_name_note'
		#agg_criteria = 'name_note'
		chord_df = self.aggregate_chord_from_tick(aggregation_criteria = agg_criteria)
		
		all_notes = list(np.unique(list(itertools.chain(*chord_df[agg_criteria]))))

		tonic = self.get_tonality()



		# Find the intersection between notes in the scale and notes in the piece of music
		tonic_scale_notes = map_tonic_with_scale[tonic]
		if agg_criteria == 'octave_name_note':
			# In the case, there are, for instance, values such as F4# or G5#
			common_notes = list(set(tonic_scale_notes) & set([iter_notas[:-1] for iter_notas in all_notes]))
		else:
			common_notes = list(set(tonic_scale_notes) & set(all_notes))

		missing_notes_in_scale = list(set(tonic_scale_notes) - set(common_notes))
		# print(common_notes)
		# print(missing_notes_in_scale)

		# Convert notes in music to the closest in tonic scale
		renamed_missing_notes = \
							([map_note_with_alias[renamed_notes] 
		                     for renamed_notes in missing_notes_in_scale 
		                     if renamed_notes in map_note_with_alias.keys()]
		                     )


		# Apply the mapping transformation to the remaining notes
		if agg_criteria == 'octave_name_note':
			chord_df['chord'] = \
			(chord_df[agg_criteria]
			 .apply(lambda tuple_x: 
			        tuple([map_note_with_alias[renamed_notes[:-1]]+renamed_notes[-1]
			              if renamed_notes[:-1] in renamed_missing_notes 
			              else renamed_notes
			              for renamed_notes in tuple_x
			              ])))
		else:
			chord_df['chord'] = \
			(chord_df[agg_criteria]
			 .apply(lambda tuple_x: 
			        tuple([map_note_with_alias[renamed_notes] 
			              if renamed_notes in renamed_missing_notes 
			              else renamed_notes
			              for renamed_notes in tuple_x
			              ])))

		# Convert chord into grades
		if agg_criteria == 'octave_name_note':
			chord_df['grades'] = \
			(chord_df['chord']
			 .apply(lambda tuple_x:
			        tuple([grades[map_tonic_with_scale[tonic]
			              .index(chord_element[:-1])]+chord_element[-1]
			              if chord_element[:-1] in tonic_scale_notes 
			              else 'X'
			              for chord_element in tuple_x
			              ])))
		else:
			chord_df['grades'] = \
			(chord_df['chord']
			 .apply(lambda tuple_x:
			        tuple([grades[map_tonic_with_scale[tonic]
			              .index(chord_element)]
			              if chord_element in tonic_scale_notes 
			              else 'X'
			              for chord_element in tuple_x
			              ])))

		chord_df['dur'] = chord_df['max_tick']-chord_df['min_tick']+20


		return chord_df[['chord','grades','dur']]

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
				                          iter_note['name_note'],iter_note['octave_name_note']]
				                          ))

		self.granular_music_df = (pd.DataFrame(granular_music_list,
		                                  columns = ['start_ticks','pitch','velocity','part',
		                                  'fullNoteOctave','name_note','octave_name_note']))

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
		           'name_note':lambda x: Counter(x),
		           'octave_name_note':lambda x: Counter(x),
		           }
		           )
		      .reset_index()
		      )
		# Do not forget the aggregated time pero granular chord.

	def aggregate_chord_from_tick(self, aggregation_criteria = 'name_note', dataframe = None):

		# The other option for aggregation_criteria is:
		# aggregation_criteria = 'fullNoteOctave'

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

		# See the changes in chord per ticks
		changes_in_chords = chord_per_ticks[aggregation_criteria].diff()
		# Since the first element is Nan, force it to be {}
		# It is {} since we are working with dicts
		changes_in_chords[0] = {}

		# Store the final column into id_aggregation_criteria
		chord_per_ticks['id_aggregation_criteria'] = np.cumsum([len(element)>0 
		                                                       for element in changes_in_chords])



		aggregated_chord_per_ticks = (chord_per_ticks
		      .groupby(['id_aggregation_criteria',
		               chord_per_ticks[aggregation_criteria].map(tuple)])
		      .agg({'start_ticks':['min','max']
		           }
		           )
		      .reset_index()
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

		tonic_chord_candidates.columns = ['notes','ticks','num_el']

		aggregated_chord_per_ticks.columns = \
		['seq_id', aggregation_criteria, 'min_tick','max_tick', 'len', 'time']

		return aggregated_chord_per_ticks

	def convert_grades_sequence_to_notes(self,grades_sequence, tonic):

		tonality_scale = self.get_map_tonic_with_scale()[tonic]
		grade_mapping = self.get_map_grades_with_scale_position()

		notes_sequence = list()

		for chord in grades_sequence:
			notes_chord = list()
			for note in chord:
				if note != 'X':
					# Extract octave
					octave = int(note[-1])
					note_name = note[:-1]
					converted_note = tonality_scale[grade_mapping[note_name]]
					if converted_note[-1] == 'b' or converted_note[-1] == '#':
						alteration = converted_note[-1]
						name_note  = converted_note[:-1]
					else:
						alteration = ''
						name_note  = converted_note

				else:
					# In case of X, then use tonic
					converted_note = tonic
					octave = 4
					if tonic[-1] == 'b' or tonic[-1] == '#':
						alteration = tonic[-1]
						name_note  = tonic[:-1]
					else:
						alteration = ''
						name_note  = tonic

				notes_props = {'duration':200, 'intensity':70, 'timbre':1,
						   'alteration':alteration, 'octave':octave}		
				notes_chord.append(globals()[name_note](**notes_props))

			notes_sequence.append(notes_chord)

			# notes_props = {'duration':200, 'intensity':70, 'timbre':1,
			# 			   'alteration':alteration, 'octave':octave}

			# notes_sequence.append(globals()[name_note](**notes_props))
				
		return notes_sequence



		
def _get_note_name_without_octave(fullNoteOctave):
	# Function to get the name, regardless the octave

	notes_dict = {'A':'La','B':'Si','C':'Do','D':'Re','E':'Mi','F':'Fa','G':'Sol'}

	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[2]
	else:
		# Names can be F3, A2, then return F and A
		return notes_dict[fullNoteOctave[0]]

def _get_note_name_with_octave(fullNoteOctave):
	# Function to get the name INCLUDING the octave

	notes_dict = {'A':'La','B':'Si','C':'Do','D':'Re','E':'Mi','F':'Fa','G':'Sol'}

	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[2]+fullNoteOctave[1]
	else:
		# Names can be F3, A2, then return F and A
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[1]

def _convert_note_into_grade(chord_tuple):
	# Function to convert a chord such as (F4#, C5#, F5, A4) into 
	# grade functionalities (II,III,I,IV) for instance.
	# In this way, music is normalized and harmonic structure may be learned.

	tonality = self.get_tonality()

if __name__ == "__main__":

	name_file_midi = '../../scores/Schubert_S560_Schwanengesang_no7.csv'
	name_file_midi = '../../scores/Brahms_symphony_2_2.csv' # Si M
	name_file_midi = '../../scores/Brahms_symphony_2_1.csv'
	name_file_midi = '../../scores/Bach-Partita_No1_in_Bb_BWV825_7Gigue.csv'
	name_file_midi = '../../scores/Albeniz_Asturias.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_5.csv'
	name_file_midi = '../../scores/Schuber_Impromptu_D_899_No_3.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_1.csv'
	name_file_midi = '../../scores/Debussy_Claire_de_Lune.csv'
	#name_file_midi = '../../scores/Beethoven_Moonlight_Sonata_third_movement.csv'
	#name_file_midi = '../../scores/Schubert_Piano_Trio_2nd_Movement.csv'
	
	chopin = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	#print(chopin.get_chord_from_tick().filter(['fullNoteOctave']))
	print('La tonalidad es: '+chopin.get_tonality())
	# grades_chords = chopin.apply_tonality()
	# grades_chords.to_csv('../tmp/'+name_file_midi[13:-4]+'_grades_chords.csv',
	#                      header=True,
	#                      index_label=None)

	# print(grades_chords)

	grades_sequence = [('V4', 'III5', 'III4', 'I5'),
('IV2', 'I3', 'III5', 'I5'),
('I3', 'I4', 'VI3', 'VI4', 'IV2', 'IV3', 'IV4', 'II5'),
('I3', 'I4', 'III5', 'VI3', 'VI4', 'IV2', 'IV3', 'IV4'),
('I3', 'I4', 'VI3', 'VI4', 'IV2', 'IV3', 'IV4', 'II5'),
('I4', 'VI3', 'VI4', 'IV3', 'IV4', 'II5'),
('III2', 'I3', 'II5'),
('III2', 'I3', 'V4', 'V3', 'I4', 'I5', 'III3'),
('III2', 'I3', 'V4', 'V3', 'III3', 'II4', 'II5'),
('III2', 'I3', 'V4', 'V5', 'III4', 'I5', 'I4'),
('V3', 'I5', 'I4', 'III4', 'III5'),
('IV2', 'I3', 'III5'),
('I3', 'I4', 'VI3', 'VI4', 'IV2', 'IV3', 'IV4', 'II5'),
('I3', 'I4', 'III5', 'VI3', 'VI4', 'IV2', 'IV3', 'IV4'),
('I3', 'I4', 'VI3', 'VI4', 'IV2', 'IV3', 'IV4', 'II5'),
('I4', 'I5', 'VI3', 'VI4', 'IV3', 'IV4'),
('V2', 'I5'),
('X', 'III3', 'V4', 'V2', 'I4', 'I5'),
('X', 'III3', 'V4', 'V2', 'II4', 'II5'),
('X', 'V2', 'I4', 'III5', 'VI4', 'VI5', 'III4'),
('X', 'V4', 'V5', 'V2', 'I4', 'III5', 'III4'),
('V4', 'V5', 'V2', 'III5'),
('V2', 'III4', 'III5'),
('VI2',),
('III3', 'III4', 'VI2', 'VI3', 'VI4', 'II4', 'II5'),
('III3', 'III4', 'III5', 'VI2', 'VI3', 'VI4', 'II4'),
('III3', 'III4', 'VI2', 'VI3', 'VI4', 'II4', 'II5'),
('III3', 'I4', 'I5', 'VI3', 'VI4', 'III4'),
('III3', 'VI3', 'V3', 'II4'),
('III3', 'VI3', 'V3', 'I4'),
('II2', 'II1'),
('III6', 'III4', 'III5', 'VI4', 'VI5', 'II2', 'II1', 'IV4'),
('VI4', 'VI5', 'IV4', 'II6', 'II4', 'II5'),
('II4', 'VI4', 'VI5', 'II2', 'II1', 'II6', 'IV4', 'II5'),
('I6', 'I4', 'I5', 'VI4', 'VI5', 'IV4'),
('VI4', 'VI5', 'VII4', 'IV5', 'VII6', 'IV4', 'VII5'),
('I6', 'I4', 'I5', 'VI4', 'VI5', 'IV4'),
('I4', 'VI3', 'VI4', 'VI5', 'IV4', 'IV5'),
('II2', 'VI2', 'II1'),
('III6', 'III4', 'III5', 'VI2', 'VI4', 'VI5', 'II2', 'II1', 'IV4'),
('II5', 'VI2', 'VI4', 'VI5', 'II2', 'II1', 'IV6', 'IV4', 'IV5'),
('III6', 'III4', 'III5', 'VI2', 'VI4', 'VI5', 'II2', 'II1', 'IV4'),
('VI4', 'VI5', 'IV4', 'II6', 'II4', 'II5'),
('III6', 'III4', 'III5', 'VI4', 'VI5', 'IV4'),
('VI4', 'VI5', 'IV4', 'II6', 'II4', 'II5'),
('I6', 'I4', 'I5', 'VI4', 'VI5', 'IV4'),
('VI4', 'VI5', 'IV4', 'II6', 'II4', 'II5'),
('I6', 'I4', 'I5', 'VI4', 'VI5', 'IV4'),
('VI4', 'VI5', 'IV4', 'VII6', 'VII4', 'VII5'),
('II6',),
('VI4', 'VI5', 'IV4', 'VII6', 'VII4', 'VII5'),
('I6', 'I4', 'I5', 'VI4', 'VI5', 'IV4'),
('VI3', 'VI4', 'VI5', 'IV4', 'II4', 'IV5'),
('V2', 'V1', 'VI4', 'VI5', 'IV5'),
('IV3', 'IV5', 'V2', 'IV4', 'V1'),
('V4', 'V5', 'V2', 'V3', 'V1', 'VII4', 'II4', 'II5'),
('VII6', 'VII5', 'V2', 'V1'),
('V2', 'V1', 'II4', 'VII6', 'VII4', 'VII5'),
('VI3', 'VI4', 'VI5', 'IV4', 'II4', 'IV5'),
('IV3', 'IV4', 'IV5'),
('IV3', 'II4', 'IV4', 'IV5'),
('I6', 'IV6', 'X'),
('X', 'IV6', 'IV4'),
('IV6', 'IV4', 'III4'),
('X', 'IV6', 'III4'),
('IV6', 'X', 'I5'),
('IV6', 'II4', 'IV5'),
('I6', 'IV6', 'X'),
('X', 'IV6', 'III4'),
('X', 'X', 'I4'),
('X', 'X', 'I4'),
('X', 'I4', 'I5'),
('X', 'X', 'I4'),
('I7', 'I4'),
('I6', 'I7', 'I4'),
('X', 'IV5'),
('X', 'II4', 'IV5'),
('X', 'IV4', 'IV5'),
('X', 'I5', 'IV5'),
('V5', 'VII6', 'VII5'),
('X', 'X', 'X'),
('X', 'I4', 'IV5'),
('X', 'IV4', 'IV5'),
('X', 'IV4', 'IV5'),
('X', 'IV4', 'IV5'),
('X', 'V5', 'X', 'X'),
('V5', 'V3', 'VII5'),
('V4', 'I4', 'X'),
('V4', 'I4', 'I5'),
('III3', 'III4'),
('III3', 'VII5'),
('I3', 'X', 'I5'),
('X', 'X', 'II5'),
('IV3', 'X', 'II5'),
('X', 'X', 'I5'),
('X', 'I4', 'I5'),
('V4', 'X', 'X'),
('V4', 'I4', 'X'),
('X', 'IV4', 'X'),
('X', 'IV3', 'IV4')]


	grades_sequence_chopin = [('I2', 'I1', 'I4'),
('I2', 'I1', 'V3'),
('I2', 'I1', 'III4'),
('I2', 'V4', 'I1'),
('I2', 'I1', 'I5'),
('I2', 'V4', 'I1'),
('I2', 'I1', 'III5'),
('I2', 'I1', 'V5'),
('I2', 'I1', 'I6'),
('I2', 'I1', 'V5'),
('I2', 'I1', 'III6'),
('I2', 'V6', 'I1'),
('I2', 'I1', 'I7'),
('I2', 'V4', 'I1'),
('V2', 'X', 'V1'),
('X', 'X', 'X'),
('IV2', 'IV1', 'I5'),
('IV2', 'IV1', 'III5'),
('IV2', 'IV1', 'I4'),
('IV2', 'IV1', 'III6'),
('IV2', 'IV1', 'I6'),
('X', 'X', 'X'),
('IV2', 'IV1', 'I5'),
('II3', 'V2', 'V1'),
('III3', 'III2', 'VII4'),
('X', 'X', 'VII4'),
('V4', 'V2', 'V1'),
('II3', 'V2', 'V1'),
('III3', 'III2', 'VII4'),
('I6', 'VI1', 'VI2'),
('I2', 'I3'),
('I2', 'I3'),
('I2', 'I1', 'V3'),
('I2', 'I1', 'I4'),
('I2', 'I1', 'III5'),
('I2', 'I1', 'VII4'),
('I2', 'V4', 'I1'),
('I2', 'I1', 'I4'),
('I2', 'V4', 'I1'),
('I2', 'I4'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V5'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I4'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V5'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X'),
('I2', 'I3', 'III6'),
('I2', 'I3', 'I6'),
('I2', 'I3', 'V4'),
('I2', 'I3', 'X')]

	chords_notes = chopin.convert_grades_sequence_to_notes(grades_sequence_chopin, chopin.get_tonality())

	polyphony = SequenceChordPolyphony(chords_notes)
	CSVtoMIDI(polyphony.convert_to_midi(),'my_first_polyphony')





