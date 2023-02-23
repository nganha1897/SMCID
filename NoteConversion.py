''' Reference: 
    https://github.com/ianvonseggern1/note-prediction
    https://stackoverflow.com/questions/32373996/pydub-raw-audio-data
    https://stackoverflow.com/questions/53308674/audio-frequencies-in-python
 '''
import librosa
import array
from collections import Counter
import numpy as np
import scipy
from pydub.utils import get_array_type
from Levenshtein import distance
from pydub import AudioSegment
import argparse
import pydub.scipy_effects

class NoteConversion:
    audio_file = ""

    def __init__(self, audio_file):
        self.audio_file = audio_file
    
    def find_note_starts(self, song):
        SEGMENT_MS = 50
        VOLUME_THRESHOLD = -35
        EDGE_THRESHOLD = 5
        MIN_MS_BETWEEN = 100

        song = song.high_pass_filter(librosa.note_to_hz('C2'), order=4)
        volume = [segment.dBFS for segment in song[::SEGMENT_MS]]

        predicted_starts = []
        for i in range(1, len(volume)):
            if volume[i] > VOLUME_THRESHOLD and volume[i] - volume[i - 1] > EDGE_THRESHOLD:
                ms = i * SEGMENT_MS
                if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                    predicted_starts.append(ms)

        return predicted_starts

    def find_frequency(self, sample, max_frequency=800):
        bit_depth = sample.sample_width * 8
        array_type = get_array_type(bit_depth)
        raw_audio_data = array.array(array_type, sample._data)
        n = len(raw_audio_data)

        freq_array = np.arange(n) * (float(sample.frame_rate) / n)
        freq_array = freq_array[: (n // 2)]

        raw_audio_data = raw_audio_data - np.average(raw_audio_data)
        freq_magnitude = scipy.fft.fft(raw_audio_data)
        freq_magnitude = freq_magnitude[: (n // 2)]

        if max_frequency:
            max_index = int(max_frequency * n / sample.frame_rate) + 1
            freq_array = freq_array[:max_index]
            freq_magnitude = freq_magnitude[:max_index]

        freq_magnitude = abs(freq_magnitude)
        freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
        return freq_array, freq_magnitude

    def classify_note_attempt(self, freq_array, freq_magnitude):
        min_freq = librosa.note_to_hz('C2')
        note_counter = Counter()
        for i in range(len(freq_magnitude)):
            if freq_magnitude[i] < 0.01:
                continue

            for freq_multiplier, credit_multiplier in [
                (1, 1),
                (1 / 3, 3 / 4),
                (1 / 5, 1 / 2),
                (1 / 6, 1 / 2),
                (1 / 7, 1 / 2),
            ]:
                freq = freq_array[i] * freq_multiplier

                if freq < min_freq:
                    continue
                note = librosa.hz_to_note(freq)
                if note:
                    note_counter[note] += freq_magnitude[i] * credit_multiplier

        return note_counter.most_common(1)[0][0]

    def convert_notes(self):
        song = AudioSegment.from_file(self.audio_file)
        song = song.high_pass_filter(80, order=4)
        starts = self.find_note_starts(song)
        converted_notes = []
        for i, start in enumerate(starts):
            sample_from = start + 50
            sample_to = start + 550
            if i < len(starts) - 1:
                sample_to = min(starts[i + 1], sample_to)
            segment = song[sample_from:sample_to]
            freqs, freq_magnitudes = self.find_frequency(segment)

            converted = self.classify_note_attempt(freqs, freq_magnitudes)

            converted_notes.append(converted or "U")
        print(converted_notes)
        return converted_notes
    
    