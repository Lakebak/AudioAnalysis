from re import S
import pandas as pd
from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures
from Audio import Audio

class Feature:
    def __init__(self, 
    low_lvl_wnd: float,
    low_lvl_step: float,
    mid_lvl_wnd: float,
    mid_lvl_step: float):
    # TO-DO: implementare possibilità di scegliere le feature da un file di configurazione
        self.mw = mid_lvl_wnd
        self.ms = mid_lvl_step
        self.sw = low_lvl_wnd
        self.ss = low_lvl_step
    
    def __call__(self, signal, sr):

        def __pyaudio_analysis(self, signal, sr: int=44100, 
    low_lvl: bool=False,
    mid_lvl: bool=True):

            window = sr * self.sw
            step = sr * self.ss
            mid_window = sr * self.mw
            mid_step = sr * self.ms
            
            def low_level():
                if low_lvl:
                    low_feat, low_feat_names = \
                        ShortTermFeatures.feature_extraction(signal=signal,
                        sampling_rate=sr,
                        window=window,
                        step=step)
                    df_low = pd.DataFrame(low_feat.transpose(),
                    columns=low_feat_names)
                    return df_low
                else:
                    df_low = pd.DataFrame()

            def mid_level():
                if mid_lvl:
                    mid_feat, _, mid_feat_names = \
                        MidTermFeatures.mid_feature_extraction(signal=signal,
                        sampling_rate=sr,
                        mid_window=mid_window,
                        mid_step=mid_step,
                        short_window=window,
                        short_step=step)
                    
                    df_mid = pd.DataFrame(mid_feat.transpose(),
                    columns=mid_feat_names)
                    return df_mid
                else:
                    df_mid = pd.DataFrame()
        
            return low_level(), mid_level()
        
        return __pyaudio_analysis(self, signal=signal, sr=sr) 