#!/usr/bin/env python
# -*- coding: utf-8 -*-



# Create Musical Notes (Do, Re, Mi, ...)

class MusicalNote(object):

	# Duration (lon/short notes)
	# Pitch Frequency (high/low pitch)
	# Intensity (louder/soft)
	# Timbre (instrument you use to play)

	def __init__(self, **notes_props):
		self.notes_props = notes_props
		self.duration = notes_props['duration']
		self.pitch = notes_props['pitch']
		self.intensity = notes_props['intensity']
		self.timbre = notes_props['timbre']

	def get_pitch(self):
		return self.pitch

	def get_duration(self):
		return self.duration

	def get_props(self):
		return self.notes_props

	# MusicalNote({parameters:})

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
	notes_props = {'pitch':12,'duration':100,'timbre':2,'intensity':4}
	standardMusicalNote = MusicalNote(**notes_props)
	do = Do(**notes_props)
	re = Re(**notes_props)

	print(standardMusicalNote.get_pitch())
	print(do.get_pitch())
	print(re.get_pitch())
