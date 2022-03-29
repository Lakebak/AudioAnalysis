import librosa as lb
import noisereduce as nr
from pyAudioAnalysis import audioBasicIO


class Audio:
    def __init__(self, path: str, sr: int):
        self.path = path
        self.sr = sr

    def load(self):
        self.signal, sr = audioBasicIO.read_audio_file(self.path)
        if sr != self.sr:
            self.signal = lb.resample(self.signal, sr, self.sr)
            
        return self.signal

    def denoise(self,
    prop_decrease: float=1.0,
    freq_mask_smooth_hz: int=500,
    time_mask_smooth_ms: int=50,
    n_std_thresh_stationary: float=1.5,
    chunk_size: int=6000,
    n_ftt: int=1024):  
        filtered = nr.reduce_noise(y=self.signal, sr=self.sr,
        stationary=True,
        prop_decrease=prop_decrease,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
        n_std_thresh_stationary=n_std_thresh_stationary,
        chunk_size=chunk_size,
        n_fft=n_ftt)

        return filtered