import os
import glob
import pandas as pd
from modules import AudioFeature
from tqdm.auto import tqdm


path = '/mnt/f/PhD/_Segments/Segmented15s/.selected_audio/ami'
print("Instantiating...")
features = AudioFeature()
turn_taker = features.TurnTaking()
prosodier = features.Prosody()

dataset = pd.DataFrame()

print("Done.\nProcessing...")
for file in tqdm(glob.glob(os.path.join(path, '*.*'))):
    turn_feat = turn_taker(file)
    prosody_feat = prosodier(file)
    feats = features(turn_feat, prosody_feat)
    dataset = pd.concat([dataset, feats], axis=0)
print("Done.\n")

dataset.to_parquet("/mnt/f/PhD/_Features/ami/speech_5s.parquet", engine='fastparquet')
print("File saved")

