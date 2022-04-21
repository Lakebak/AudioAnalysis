import librosa as lb
import numpy as np
import soundfile as sf
from modules.pyAudioAnalysis import audioBasicIO


class Audio:
    def __init__(self, path: str, sr: int, mono: bool = True):
        self.signal = None
        self.path = path
        self.sr = sr
        self.mono = mono

    def load(self):
        sr, self.signal = audioBasicIO.read_audio_file(self.path)

        #  Convertire a float prima di ricampionare o cercare ricampionamento type indipendente
        '''if sr != self.sr:
            self.signal = lb.resample(self.signal, sr, self.sr)'''

        if self.mono:
            self.signal = audioBasicIO.stereo_to_mono(self.signal)
            self.signal = np.array(self.signal, dtype=np.int16)

        return self.signal

    def stream(self, frame_length, hop_length, block_length=1):
        stream = lb.stream(self.path,
                           frame_length=frame_length,
                           hop_length=hop_length,
                           mono=self.mono,
                           block_length=block_length,
                           fill_value=0
                           )

        return stream

    def __save_file(self, out_path: str, filtered_signal: np.ndarray):
        sf.write(file=out_path,
                 data=filtered_signal,
                 samplerate=self.sr,
                 subtype='PCM_16')
