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
    'pcm_RMSenergy_sma_stddev', 'pcm_RMSenergy_sma_skewness', 'pcm_RMSenergy_sma_kurtosis', 'pcm_RMSenergy_sma_meanSegLen'
    'pcm_zcr_sma_stddev', 'pcm_zcr_sma_skewness', 'pcm_zcr_sma_kurtosis', 'pcm_zcr_sma_meanSegLen',
    'pcm_RMSenergy_sma_de_stddev', 'pcm_RMSenergy_sma_de_skewness', 'pcm_RMSenergy_sma_de_kurtosis', 'pcm_RMSenergy_sma_de_meanSegLen',
    'pcm_zcr_sma_de_stddev', 'pcm_zcr_sma_de_skewness', 'pcm_zcr_sma_de_kurtosis', 'pcm_zcr_sma_de_meanSegLen',
    'pcm_fftMag_spectralRollOff25.0_sma_stddev', 'pcm_fftMag_spectralRollOff25.0_sma_skewness', 'pcm_fftMag_spectralRollOff25.0_sma_kurtosis', 'pcm_fftMag_spectralRollOff25.0_sma_meanSegLen',
    'pcm_fftMag_spectralRollOff50.0_sma_stddev', 'pcm_fftMag_spectralRollOff50.0_sma_skewness', 'pcm_fftMag_spectralRollOff50.0_sma_kurtosis', 'pcm_fftMag_spectralRollOff50.0_sma_meanSegLen',
    'pcm_fftMag_spectralRollOff75.0_sma_stddev', 'pcm_fftMag_spectralRollOff75.0_sma_skewness', 'pcm_fftMag_spectralRollOff75.0_sma_kurtosis', 'pcm_fftMag_spectralRollOff75.0_sma_meanSegLen',
    'pcm_fftMag_spectralRollOff90.0_sma_stddev', 'pcm_fftMag_spectralRollOff90.0_sma_skewness', 'pcm_fftMag_spectralRollOff90.0_sma_kurtosis', 'pcm_fftMag_spectralRollOff90.0_sma_meanSegLen',
    'pcm_fftMag_spectralFlux_sma_stddev', 'pcm_fftMag_spectralFlux_sma_skewness', 'pcm_fftMag_spectralFlux_sma_kurtosis', 'pcm_fftMag_spectralFlux_sma_meanSegLen',
    'pcm_fftMag_spectralCentroid_sma_stddev', 'pcm_fftMag_spectralCentroid_sma_skewness', 'pcm_fftMag_spectralCentroid_sma_kurtosis', 'pcm_fftMag_spectralCentroid_sma_meanSegLen',
    'pcm_fftMag_spectralEntropy_sma_stddev',  'pcm_fftMag_spectralEntropy_sma_skewness',  'pcm_fftMag_spectralEntropy_sma_kurtosis',  'pcm_fftMag_spectralEntropy_sma_meanSegLen',
    'pcm_fftMag_spectralVariance_sma_stddev', 'pcm_fftMag_spectralVariance_sma_skewness', 'pcm_fftMag_spectralVariance_sma_kurtosis', 'pcm_fftMag_spectralVariance_sma_meanSegLen',
    'pcm_fftMag_spectralSkewness_sma_stddev', 'pcm_fftMag_spectralSkewness_sma_skewness', 'pcm_fftMag_spectralSkewness_sma_kurtosis', 'pcm_fftMag_spectralSkewness_sma_meanSegLen',
    'pcm_fftMag_spectralKurtosis_sma_stddev', 'pcm_fftMag_spectralKurtosis_sma_skewness', 'pcm_fftMag_spectralKurtosis_sma_kurtosis', 'pcm_fftMag_spectralKurtosis_sma_meanSegLen',
    'pcm_fftMag_spectralSlope_sma_stddev', 'pcm_fftMag_spectralSlope_sma_skewness', 'pcm_fftMag_spectralSlope_sma_kurtosis', 'pcm_fftMag_spectralSlope_sma_meanSegLen',
    'pcm_fftMag_psySharpness_sma_stddev',  'pcm_fftMag_psySharpness_sma_skewness',  'pcm_fftMag_psySharpness_sma_kurtosis', 'pcm_fftMag_psySharpness_sma_meanSegLen',
    'pcm_fftMag_spectralHarmonicity_sma_stddev', 'pcm_fftMag_spectralHarmonicity_sma_skewness', 'pcm_fftMag_spectralHarmonicity_sma_kurtosis', 'pcm_fftMag_spectralHarmonicity_sma_meanSegLen',
    'mfcc_sma[1]_stddev', 'mfcc_sma[1]_skewness', 'mfcc_sma[1]_kurtosis', 'mfcc_sma[1]_meanSegLen',
    'mfcc_sma[2]_stddev', 'mfcc_sma[2]_skewness', 'mfcc_sma[2]_kurtosis', 'mfcc_sma[2]_meanSegLen',
    'mfcc_sma[3]_stddev', 'mfcc_sma[3]_skewness', 'mfcc_sma[3]_kurtosis', 'mfcc_sma[3]_meanSegLen',
    'mfcc_sma[4]_stddev', 'mfcc_sma[4]_skewness', 'mfcc_sma[4]_kurtosis', 'mfcc_sma[4]_meanSegLen',
    'mfcc_sma[5]_stddev', 'mfcc_sma[5]_skewness', 'mfcc_sma[5]_kurtosis', 'mfcc_sma[5]_meanSegLen',
    'mfcc_sma[6]_stddev', 'mfcc_sma[6]_skewness', 'mfcc_sma[6]_kurtosis', 'mfcc_sma[6]_meanSegLen',
    'mfcc_sma[7]_stddev', 'mfcc_sma[7]_skewness', 'mfcc_sma[7]_kurtosis', 'mfcc_sma[7]_meanSegLen',
    'mfcc_sma[8]_stddev', 'mfcc_sma[8]_skewness', 'mfcc_sma[8]_kurtosis', 'mfcc_sma[8]_meanSegLen',
    'mfcc_sma[9]_stddev', 'mfcc_sma[9]_skewness', 'mfcc_sma[9]_kurtosis', 'mfcc_sma[9]_meanSegLen',
    'mfcc_sma[10]_stddev', 'mfcc_sma[10]_skewness', 'mfcc_sma[10]_kurtosis', 'mfcc_sma[10]_meanSegLen',
    'mfcc_sma[11]_stddev', 'mfcc_sma[11]_skewness', 'mfcc_sma[11]_kurtosis', 'mfcc_sma[11]_meanSegLen',
    'mfcc_sma[12]_stddev', 'mfcc_sma[12]_skewness', 'mfcc_sma[12]_kurtosis', 'mfcc_sma[12]_meanSegLen',
    'mfcc_sma[13]_stddev', 'mfcc_sma[13]_skewness', 'mfcc_sma[13]_kurtosis', 'mfcc_sma[13]_meanSegLen',
    'mfcc_sma[14]_stddev', 'mfcc_sma[14]_skewness', 'mfcc_sma[14]_kurtosis', 'mfcc_sma[14]_meanSegLen',
    'jitterLocal_sma_amean', 'jitterLocal_sma_stddev', 'jitterLocal_sma_skewness', 'jitterLocal_sma_kurtosis',
    'jitterDDP_sma_amean', 'jitterDDP_sma_stddev', 'jitterDDP_sma_skewness', 'jitterDDP_sma_kurtosis',
    'shimmerLocal_sma_amean', 'shimmerLocal_sma_stddev', 'shimmerLocal_sma_skewness', 'shimmerLocal_sma_kurtosis',
    'pcm_RMSenergy_sma_amean',
    'pcm_zcr_sma_amean',
    'pcm_fftMag_spectralRollOff25.0_sma_amean',
    'pcm_fftMag_spectralRollOff50.0_sma_amean',
    'pcm_fftMag_spectralRollOff75.0_sma_amean',
    'pcm_fftMag_spectralRollOff90.0_sma_amean',
    'pcm_fftMag_spectralFlux_sma_amean',
    'pcm_fftMag_spectralCentroid_sma_amean',
    'pcm_fftMag_spectralEntropy_sma_amean',
    'pcm_fftMag_spectralVariance_sma_amean',
    'pcm_fftMag_spectralSkewness_sma_amean',
    'pcm_fftMag_spectralKurtosis_sma_amean',
    'pcm_fftMag_spectralSlope_sma_amean',
    'pcm_fftMag_psySharpness_sma_amean',
    'pcm_fftMag_spectralHarmonicity_sma_amean',
    'mfcc_sma[1]_amean',
    'mfcc_sma[2]_amean',
    'mfcc_sma[3]_amean',
    'mfcc_sma[4]_amean',
    'mfcc_sma[5]_amean',
    'mfcc_sma[6]_amean',
    'mfcc_sma[7]_amean',
    'mfcc_sma[8]_amean',
    'mfcc_sma[9]_amean',
    'mfcc_sma[10]_amean',
    'mfcc_sma[11]_amean',
    'mfcc_sma[12]_amean',
    'mfcc_sma[13]_amean',
    'mfcc_sma[14]_amean'
]


class AudioFeature:

    def __init__(self) -> None:
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
                signal = audio[i:i + frame_length]
                if len(signal) < len(frame_length):
                    signal = np.pad(signal, (0, frame_length - len(signal)))
                df = self.smile.process_signal(
                    signal=signal,
                    sampling_rate=sr
                )
                audio_feat = pd.concat([audio_feat, df])
            return audio_feat
