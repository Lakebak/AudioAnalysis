import os
import glob
import fnmatch
from p_tqdm import p_map
from modules import Audio

def remove_noise(file):
    audio = Audio(file, sr=44100)
    signal = audio.load()
    audio.denoise(signal=signal, save=True, out_path='/mnt/f/PhD/_Segments/Segmented15s/.selected_audio/multi', prop_decrease= 0.95, 
            time_constant_s= 0.75,
             freq_mask_smooth_hz=1000)

path = "/mnt/f/PhD/_Segments/44kHz/15s/multi"

print("Collecting...")
holistic_files = [file for file in glob.glob(os.path.join(path, "*.*")) if fnmatch.fnmatch(file, "*all-audio*")]

print("Done.\nStarting parallel...")
p_map(remove_noise, holistic_files)