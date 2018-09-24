#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# Create Musical Notes (Do, Re, Mi, ...)
class MusicalNote(object):

	# Duration (lon/short notes)
	# Pitch Frequency (high/low pitch)
	# Intensity (louder/soft)
	# Timbre (instrument you use to play)

	

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

	def get_pitch(self):
		return self.pitch

	def get_duration(self):
		return self.duration

	def get_props(self):
		return self.notes_props

	
		


	def to_string(self):
		# Combine the name of the note (Do, Re, ...) with their corresponding alteration if any
		return self.__class__.__name__+self.alteration

	def get_major_chord(self):
		tonic_note = globals()[self.__class__.__name__]()
		return [tonic_note, tonic_note.get_note_from_interval('3M'), tonic_note.get_note_from_interval('5P')]

	def get_minor_chord(self):
		tonic_note = globals()[self.__class__.__name__]()
		return [tonic_note, tonic_note.get_note_from_interval('3m'), tonic_note.get_note_from_interval('5P')]

	def get_dis_chord(self):
		return self.pitch + np.cumsum([0, half_tone+tone, half_tone+tone])


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


if __name__ == '__main__':
	do_props = {'duration':100, 'intensity':50, 'timbre':1}
	print(Si(**{'alteration':'b'}).get_note_from_interval('5P').get_pitch())
	print('test')
	print(Do().get_note_from_interval('3m'))
	print(Do().get_note_from_interval('3m').get_pitch())
	print(Do().get_major_chord()[2].get_pitch())
	
