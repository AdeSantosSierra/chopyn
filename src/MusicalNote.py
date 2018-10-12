#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# Create Musical Notes (Do, Re, Mi, ...)
class MusicalNote(object):

	# Duration (lon/short notes)
	# Pitch Frequency (high/low pitch)
	# Intensity (louder/soft)
	# Timbre (instrument you use to play)

	global half_tone, tone, number_notes, diatonic_scale, distance_major_scale, dic_alterations

	# Config main properties of music
	half_tone      = 1
	tone           = 2
	number_notes   = 7
	diatonic_scale = ['Do','Re','Mi','Fa','Sol','La','Si']
	distance_major_scale = [tone, tone, half_tone, tone, tone, tone, half_tone]
	# dic_alterations = {'bb':-2,'b':-1,'n':0,'#':1,'x':2}
	dic_alterations = {'-2':'bb','-1':'b','0':'','1':'#','2':'x'}

	def __init__(self, **notes_props):

		# Only pitch is present
		# if (len(notes_props) == 1) & (notes_props.keys() == ['pitch']):
		# 	self.pitch = notes_props['pitch']
		# else:
		# 	self.notes_props = notes_props
		# 	self.duration = notes_props['duration']
		# 	self.pitch = notes_props['pitch']
		# 	self.intensity = notes_props['intensity']
		# 	self.timbre = notes_props['timbre']
		
		self.alteration = ''
		# Why 4? Because 4 is the default octave
		self.octave = 4

		# As it is used by Melody class
		self.notes_props = notes_props
		
		if 'pitch' in notes_props:
			self.pitch = notes_props['pitch']

		if 'duration' in notes_props:
			self.duration = notes_props['duration']

		if 'intensity' in notes_props:
			self.intensity = notes_props['intensity']

		if 'timbre' in notes_props:
			self.timbre = notes_props['timbre']

		# Check whether there is alteration on the properties of the notes
		if 'alteration' in notes_props:
			self.alteration = notes_props['alteration']
			self.pitch = self.pitch + int(dic_alterations.keys()[dic_alterations.values().index(self.alteration)])	
			self.notes_props['pitch'] = self.pitch

		if 'octave' in notes_props:
			
			octave_increment = notes_props['octave']-self.octave
			# Why 12? Because an octaves are separate 12 numbers
			self.pitch = self.pitch+12*octave_increment
			# Change octave in the end so that self.octave default is kept as 3
			self.octave = notes_props['octave']
			self.notes_props['pitch'] = self.pitch
			self.notes_props['octave'] = self.octave

	def get_pitch(self):
		return self.pitch

	def get_duration(self):
		return self.duration

	def get_octave(self):
		return self.octave

	def get_props(self):
		return self.notes_props

	def get_play_props(self):
		# Only those props useful to play music	
		#print(self.notes_props['duration'])	
		# print(dict((k, self.notes_props[k]) for k in ('pitch','duration','intensity','timbre') if k in self.notes_props))
		return dict((k, self.notes_props[k]) for k in ('pitch','duration','intensity','timbre') if k in self.notes_props)

	def update_props(self,new_notes_props):

		if 'pitch' in new_notes_props:
			self.notes_props.update({'pitch':new_notes_props['pitch']})
			self.pitch = self.notes_props['pitch']

		if 'duration' in new_notes_props:
			self.notes_props.update({'duration':new_notes_props['duration']})
			self.duration = self.notes_props['duration']

		if 'intensity' in new_notes_props:
			self.notes_props.update({'intensity':new_notes_props['intensity']})
			self.intensity = self.notes_props['intensity']

		if 'timbre' in new_notes_props:
			self.notes_props.update({'timbre':new_notes_props['timbre']})
			self.timbre = self.notes_props['timbre']


	def get_note_from_interval(self,interval):

		#print('get_note_from_interval')
		#print(self.alteration)

		base_note = self.__class__.__name__
		
		position_base_note = diatonic_scale.index(self.__class__.__name__)

		# Dictionary with intervals definition
		dic_interval_definition = ({'3m':[tone, half_tone],'3M':[tone, tone], 
		                           '5P':[tone, tone, half_tone, tone], 
		                           '2M':[tone], '2m':[half_tone]})

		# Estimate length of the interval (Ex. if '3m', length_interval = 2)
		length_interval = len(dic_interval_definition[interval])

		# Sum up all the tones within the wanted interval (Ex. if '3m', sum_interval = 3)
		sum_interval = np.sum(dic_interval_definition[interval])

		# Rotate distance_major_scale accordingly
		rotated_distances = distance_major_scale[position_base_note:] + distance_major_scale[:position_base_note]
		rotated_diatonic_scale = diatonic_scale[(position_base_note+length_interval) % number_notes:] + diatonic_scale[:(position_base_note+length_interval) % number_notes]
		sum_tones_interval = np.cumsum(rotated_distances)[length_interval % number_notes -1]

		# Step 1: Obtain distance between base_note and the chord_note
		chord_note = rotated_diatonic_scale[0]


		correction = int(dic_alterations.keys()[dic_alterations.values().index(self.alteration)])
		# print(base_note+self.alteration+' - '+interval+' - '+chord_note+str(dic_alterations[str(sum_interval-sum_tones_interval+correction)]))
		update_note_props = {'alteration':str(dic_alterations[str(sum_interval-sum_tones_interval+correction)])}
							 #'pitch':self.pitch+correction}
		return globals()[chord_note](**update_note_props)
		


	def to_string(self):
		# Combine the name of the note (Do, Re, ...) with their corresponding alteration if any
		return self.__class__.__name__+self.alteration

	###################	
	# Chords
	###################

	def get_major_chord(self):
		# Arreglar esto para que se pueda hacer un acorde mayor apartir de una nota alterada
		return [self, self.get_note_from_interval('3M'), self.get_note_from_interval('5P')]

	def get_minor_chord(self):
		return [self, self.get_note_from_interval('3m'), self.get_note_from_interval('5P')]

	def get_dis_chord(self):
		return self.pitch + np.cumsum([0, half_tone+tone, half_tone+tone])


	###################	
	# Scales
	###################

	def get_major_scale(self):

		tonic_scale = [self]

		# Iterate distance_major_scale
		for idx, distance in enumerate(distance_major_scale):
			if distance == tone:
				tonic_scale.append(tonic_scale[idx]
				                   .get_note_from_interval('2M'))
			else:
				tonic_scale.append(tonic_scale[idx]
				                   .get_note_from_interval('2m'))
		return tonic_scale



class Do(MusicalNote):
	# Create Musical Note - Do
	def __init__(self,**notes_props):
		# According to MIDI, Central Do has a specific pitch (60)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':60})
		super(self.__class__, self).__init__(**notes_props)

class Re(MusicalNote):
	# Create Musical Note - Re
	def __init__(self,**notes_props):
		# According to MIDI, Central Re has a specific pitch (62)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':62})
		super(self.__class__, self).__init__(**notes_props)

class Mi(MusicalNote):
	# Create Musical Note - Mi
	def __init__(self,**notes_props):
		# According to MIDI, Central Mi has a specific pitch (64)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':64})
		super(self.__class__, self).__init__(**notes_props)

class Fa(MusicalNote):
	# Create Musical Note - Fa
	def __init__(self,**notes_props):
		# According to MIDI, Central Fa has a specific pitch (65)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':65})
		super(self.__class__, self).__init__(**notes_props)

class Sol(MusicalNote):
	# Create Musical Note - Sol
	def __init__(self,**notes_props):
		# According to MIDI, Central Sol has a specific pitch (67)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':67})
		super(self.__class__, self).__init__(**notes_props)

class La(MusicalNote):
	# Create Musical Note - La
	def __init__(self,**notes_props):
		# According to MIDI, Central La has a specific pitch (69)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':69})
		super(self.__class__, self).__init__(**notes_props)


class Si(MusicalNote):
	# Create Musical Note - Si
	def __init__(self,**notes_props):
		# According to MIDI, Central Si has a specific pitch (71)
		# Focus only on one single Octave
		# Sharp (#) and flat (b)
		notes_props.update({'pitch':71})
		super(self.__class__, self).__init__(**notes_props)

	
