from email.mime import audio
import pandas as pd

from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures
from Audio import Audio


class AudioFeature:

    def __init__(self) -> None:
        self.turn_taking = self.TurnTaking()
        self.prosody = self.Prosody()

    class TurnTaking:

        def __init__(self) -> None:
            pass

    class Prosody:             # TO-DO: implementare possibilità di scegliere le feature da un file di configurazione

        def __init__(self,
            low_lvl_wnd: float,
            low_lvl_step: float,
            mid_lvl_wnd: float,
            mid_lvl_step: float):

            self.mw = mid_lvl_wnd
            self.ms = mid_lvl_step
            self.sw = low_lvl_wnd
            self.ss = low_lvl_step

    
        def __call__(self, path, sr=44100):

            def __load_audio():
                audio = Audio(path=path, sr=sr)
                return audio.load()

            def __pyaudio_analysis(sampling_rate: int=sr,
                                low_lvl: bool = False,
                                mid_lvl: bool = True):

                window = sampling_rate * self.sw
                step = sampling_rate * self.ss
                mid_window = sampling_rate * self.mw
                mid_step = sampling_rate * self.ms

                signal = __load_audio()

                def low_level():
                    if low_lvl:
                        low_feat, low_feat_names = \
                            ShortTermFeatures.feature_extraction(signal=signal,
                                                                sampling_rate=sampling_rate,
                                                                window=window,
                                                                step=step)
                        df_low = pd.DataFrame(low_feat.transpose(),
                                            columns=low_feat_names)

                    else:
                        df_low = pd.DataFrame()
                    return df_low

                def mid_level():
                    if mid_lvl:
                        mid_feat, _, mid_feat_names = \
                            MidTermFeatures.mid_feature_extraction(signal=signal,
                                                                sampling_rate=sampling_rate,
                                                                mid_window=mid_window,
                                                                mid_step=mid_step,
                                                                short_window=window,
                                                                short_step=step)

                        df_mid = pd.DataFrame(mid_feat.transpose(),
                                            columns=mid_feat_names)

                    else:
                        df_mid = pd.DataFrame()
                    return df_mid

                return low_level(), mid_level()

            return __pyaudio_analysis()
