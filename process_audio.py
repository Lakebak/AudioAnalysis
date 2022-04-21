import os
import glob
import pandas as pd
# import argparse
from tqdm.auto import tqdm
from modules import AudioFeature


def extract(path, sr, frame_length, hop_factor):
    feat_df = pd.DataFrame()
    extractor = AudioFeature()
    prosody = extractor.Prosody()
    turn_taking = extractor.TurnTaking()

    for file in tqdm(glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)):
        prosody_df = prosody(path=file,
                             sr=sr,
                             frame_length=frame_length,
                             hop_factor=hop_factor)
        turn_taking_df = turn_taking(path=file,
                                     sr=sr,
                                     frame_length=frame_length,
                                     hop_length=hop_factor)
        feature = extractor(turn_taking_df, prosody_df)
        feat_df = pd.concat([feat_df, feature])

    return feat_df


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser(
        description='extract prosody and turn taking features')
    parser.add_argument('-i', dest='path')
    parser.add_argument('-sr', dest='sr')
    parser.add_argument('-frame', dest='frame_length')
    parser.add_argument('-hop', dest='hop_factor')
    parser.add_argument('-o', dest='out_path')

    args = parser.parse_args()'''

    features = extract(path='/mnt/e/PhD/Data/audio/segmented',
                       sr=44100,
                       frame_length=220500,
                       hop_factor=176400)
    features.to_parquet('/mnt/e/PhD/Data/features/features_5s_44kHz.parquet', engine='fastparquet')

