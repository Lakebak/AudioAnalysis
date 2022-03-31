from operator import index
import os
import numpy as np
import pandas as pd
from modules import Audio
from modules.pyAudioAnalysis.MidTermFeatures import mid_feature_extraction
from modules.pyAudioAnalysis.ShortTermFeatures import feature_extraction
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection


class AudioFeature:

    def __init__(self) -> None:
        self.turn_taking = self.TurnTaking()
        self.prosody = self.Prosody()

    class TurnTaking:

        def __init__(self, 
                    vad_onset=0.648,
                    vad_offset=0.577,
                    vad_min_dur_on=0.181,
                    vad_min_dur_off=0.037,
                    ovd_onset=0.448,
                    ovd_offset=0.362,
                    ovd_min_dur_on=0.116,
                    ovd_min_dur_off=0.187):
            self.vad_onset = vad_onset
            self.vad_offset = vad_offset
            self.vad_min_dur_on = vad_min_dur_on
            self.vad_min_dur_off = vad_min_dur_off
            self.ovd_onset = ovd_onset
            self.ovd_offset = ovd_offset
            self.ovd_min_dur_on = ovd_min_dur_on
            self.ovd_min_dur_off = ovd_min_dur_off

        def __make_df(path, iterable_track):
            file_name = os.path.splitext(os.path.basename(path))[0]

            data = {}
            
            for tracks, person, _ in iterable_track.itertracks(yield_label=True):
                #data['File'] = file_name
                data['Person'] = person
                data['Start'] = tracks.start
                data['End'] = tracks.end
                data['Duration'] = tracks.duration
                data['Middle'] = tracks.middle

            df = pd.DataFrame([data], index=[file_name])
            

            if df.empty:
                data = {'Person': np.nan,
                'Start': np.nan,
                'End': np.nan,
                'Duration': np.nan,
                'Middle': np.nan
                }
                df = pd.DataFrame([data], index=[file_name])
                
            #df.set_index('File', inplace=True)
            return df

        def __call__(self, path):
                        
            def voice_activity(self, path):
                pipeline = VoiceActivityDetection(segmentation='pyannote/segmentation')
                hyper_parameters = {
                    "onset": self.vad_onset,
                    "offset": self.vad_offset,
                    "min_duration_on": self.vad_min_dur_on,
                    "min_duration_off": self.vad_min_dur_off
                }

                pipeline.instantiate(hyper_parameters)

                return pipeline(path)

            def overlapped_speech(self, path):
                pipeline = OverlappedSpeechDetection(segmentation='pyannote/segmentation')
                hyper_parameters = {
                    "onset": self.ovd_onset,
                    "offset": self.ovd_offset,
                    "min_duration_on": self.ovd_min_dur_on,
                    "min_duration_off": self.ovd_min_dur_off
                }

                pipeline.instantiate(hyper_parameters)

                return pipeline(path)

            vad = voice_activity(self, path)
            ovd = overlapped_speech(self, path)
            df_vad = AudioFeature.TurnTaking.__make_df(path, vad)
            df_ovd = AudioFeature.TurnTaking.__make_df(path, ovd)

            return df_vad, df_ovd

        def make_dataset(self, df_vad, df_ovd):
            dataset = df_vad.join(df_ovd,
            lsuffix='_vad',
            rsuffix='_ovd',
            sort=True)

            return dataset


    class Prosody:  # TO-DO: implementare possibilità di scegliere le feature da un file di configurazione

        def __init__(self,
                     low_lvl_wnd: float=0.5,
                     low_lvl_step: float=0.4,
                     mid_lvl_wnd: float=5.0,
                     mid_lvl_step: float=1.0):

            self.mw = mid_lvl_wnd
            self.ms = mid_lvl_step
            self.sw = low_lvl_wnd
            self.ss = low_lvl_step

        def __call__(self, path, sr=44100):

            def __load_audio():
                audio = Audio(path=path, sr=sr)
                return audio.load()

            def __pyaudio_analysis(sampling_rate: int = sr,
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
                            feature_extraction(signal=signal,
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
                            mid_feature_extraction(signal=signal,
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
