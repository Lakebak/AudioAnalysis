import os
import librosa as lb
import noisereduce as nr
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

    def __save_file(self, out_path: str, filtered_signal: np.ndarray):
        sf.write(file=out_path,
                 data=filtered_signal,
                 samplerate=self.sr,
                 subtype='PCM_16')

    def denoise(self,
                signal: np.ndarray,
                save: bool = False,
                out_path: str = 'modules/tmp',
                prop_decrease: float = 1.0,
                time_constant_s: float = 2.0,
                freq_mask_smooth_hz: int = 500,
                time_mask_smooth_ms: int = 50,
                thresh_n_mult_nonstationary: int = 1,
                sigmoid_slope_nonstationary: int = 10,
                chunk_size: int = 6000,
                n_ftt: int = 512):

        filtered = nr.reduce_noise(y=signal, sr=self.sr,
                                   prop_decrease=prop_decrease,
                                   time_constant_s=time_constant_s,
                                   freq_mask_smooth_hz=freq_mask_smooth_hz,
                                   time_mask_smooth_ms=time_mask_smooth_ms,
                                   thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
                                   sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
                                   chunk_size=chunk_size,
                                   n_fft=n_ftt,
                                   use_tqdm=False)

        out_file = os.path.basename(self.path)
        out_path = os.path.join(out_path, out_file)

        if save:
            Audio.__save_file(self, out_path=out_path, filtered_signal=filtered)

        return filtered
