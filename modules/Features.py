import os
import numpy as np
import opensmile
import pandas as pd
import librosa as lb
import opensmile as smile
from torch import Tensor, reshape
from modules import Audio
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection

MID_LEVEL_NAMES = [
    'zcr_mean', 'energy_mean', 'energy_entropy_mean',
    'spectral_centroid_mean', 'spectral_spread_mean',
    'spectral_entropy_mean', 'spectral_flux_mean', 'spectral_rolloff_mean',
    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean',
    'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean',
    'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean',
    'mfcc_13_mean',
    'zcr_std', 'energy_std', 'energy_entropy_std',
    'spectral_centroid_std', 'spectral_spread_std',
    'spectral_entropy_std', 'spectral_flux_std', 'spectral_rolloff_std',
    'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std',
    'mfcc_5_std', 'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_std',
    'mfcc_9_std', 'mfcc_10_std', 'mfcc_11_std', 'mfcc_12_std',
    'mfcc_13_std',
    'zcr_kurtosis', 'energy_kurtosis', 'energy_entropy_kurtosis',
    'spectral_centroid_kurtosis', 'spectral_spread_kurtosis',
    'spectral_entropy_kurtosis', 'spectral_flux_kurtosis', 'spectral_rolloff_kurtosis',
    'mfcc_1_kurtosis', 'mfcc_2_kurtosis', 'mfcc_3_kurtosis', 'mfcc_4_kurtosis',
    'mfcc_5_kurtosis', 'mfcc_6_kurtosis', 'mfcc_7_kurtosis', 'mfcc_8_kurtosis',
    'mfcc_9_kurtosis', 'mfcc_10_kurtosis', 'mfcc_11_kurtosis', 'mfcc_12_kurtosis',
    'mfcc_13_kurtosis',
    'zcr_skewness', 'energy_skewness', 'energy_entropy_skewness',
    'spectral_centroid_skewness', 'spectral_spread_skewness',
    'spectral_entropy_skewness', 'spectral_flux_skewness', 'spectral_rolloff_skewness',
    'mfcc_1_skewness', 'mfcc_2_skewness', 'mfcc_3_skewness', 'mfcc_4_skewness',
    'mfcc_5_skewness', 'mfcc_6_skewness', 'mfcc_7_skewness', 'mfcc_8_skewness',
    'mfcc_9_skewness', 'mfcc_10_skewness', 'mfcc_11_skewness', 'mfcc_12_skewness',
    'mfcc_13_skewness'
]


class AudioFeature:

    def __init__(self, sr) -> None:
        self.sr = sr
        self.turn_taking = self.TurnTaking()
        self.prosody = self.Prosody()

    # Questa funzione deve essere implementata - controllare il join
    # perchè hanno campioni diversi
    def __call__(self,
                 turn_taking_df,
                 prosody_df):

        turn_taking_df.drop(
            columns=['Person_ovd', 'Person_vad', 'File_ovd'],
            inplace=True)
        turn_taking_df.rename(columns={'File_vad': 'File'},
                              inplace=True)
        prosody = prosody_df.loc[:, MID_LEVEL_NAMES]

        dataset = turn_taking_df.join(prosody)

        return dataset.reindex(sorted(dataset.columns), axis=1)

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

        @staticmethod
        def __make_df(path, iterable_track):
            file_name = os.path.splitext(os.path.basename(path))[0]

            data = {}

            for tracks, person, _ in iterable_track.itertracks(yield_label=True):
                # data['File'] = file_name
                data['Person'] = person
                data['Start'] = tracks.start
                data['End'] = tracks.end
                data['Duration'] = tracks.duration
                data['Middle'] = tracks.middle

            df = pd.DataFrame([data])

            if df.empty:
                data = {'Person': np.nan,
                        'Start': np.nan,
                        'End': np.nan,
                        'Duration': np.nan,
                        'Middle': np.nan
                        }
                df = pd.DataFrame([data])

            df['File'] = file_name
            # df.set_index('File', inplace=True)
            return df

        def __call__(self, path, sr=44100, frame_length=220500, hop_length=176400):

            def load_audio():
                audio = Audio(path, sr=sr)
                return audio.stream(frame_length=frame_length, hop_length=hop_length)

            def voice_activity(self, audio_in_memory):
                pipeline = VoiceActivityDetection(segmentation='pyannote/segmentation')
                hyper_parameters = {
                    "onset": self.vad_onset,
                    "offset": self.vad_offset,
                    "min_duration_on": self.vad_min_dur_on,
                    "min_duration_off": self.vad_min_dur_off
                }

                pipeline.instantiate(hyper_parameters)

                return pipeline(audio_in_memory)

            def overlapped_speech(self, audio_in_memory):
                pipeline = OverlappedSpeechDetection(segmentation='pyannote/segmentation')
                hyper_parameters = {
                    "onset": self.ovd_onset,
                    "offset": self.ovd_offset,
                    "min_duration_on": self.ovd_min_dur_on,
                    "min_duration_off": self.ovd_min_dur_off
                }

                pipeline.instantiate(hyper_parameters)

                return pipeline(audio_in_memory)

            def postprocess_turn_taking(self, df_vad, df_ovd):
                df_vad['Silence'] = (frame_length / sr) - df_vad['Duration']
                df_vad['Speech ratio'] = df_vad['Duration'] / df_vad['Silence']

                dataset = df_vad.join(df_ovd,
                                      lsuffix='_vad',
                                      rsuffix='_ovd')
                return dataset

            df_vad = pd.DataFrame()
            df_ovd = pd.DataFrame()

            stream = load_audio()
            for block in stream:
                waveform = reshape(Tensor(block), (1, -1))
                audio_in_memory = {"waveform": waveform, "sample_rate": sr}
                vad = voice_activity(self, audio_in_memory=audio_in_memory)
                ovd = overlapped_speech(self, audio_in_memory=audio_in_memory)

                df_vad = pd.concat([df_vad, AudioFeature.TurnTaking.__make_df(path, vad)])
                df_ovd = pd.concat([df_ovd, AudioFeature.TurnTaking.__make_df(path, ovd)])

            stop = df_vad.shape[0]
            df_vad.set_index(pd.RangeIndex(start=0, stop=stop), inplace=True)
            df_ovd.set_index(pd.RangeIndex(start=0, stop=stop), inplace=True)

            return postprocess_turn_taking(self, df_vad=df_vad, df_ovd=df_ovd)

    class Prosody:  # TO-DO: implementare possibilità di scegliere le feature da un file di configurazione
        def __init__(self):
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
                num_workers=4,
                multiprocessing=True
            )

        def __call__(self,
                     path,
                     sr, frame_length,
                     hop_factor):
            audio, sr = lb.load(path,
                                sr=sr)
            audio_feat = pd.DataFrame()
            for i in range(0, len(audio), hop_factor):
                signal = audio[i:i+frame_length]
                if len(signal) < len(frame_length):
                    signal = np.pad(signal, (0, frame_length - len(signal)))
                df = self.smile.process_signal(
                    signal=signal,
                    sampling_rate=sr
                )
                audio_feat = pd.concat([audio_feat, df])
            return audio_feat



