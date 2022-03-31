import os
import glob
import fnmatch
from p_tqdm import p_map
from modules import Audio

def remove_noise(file):
    audio = Audio(file, sr=44100)
    signal = audio.load()
    audio.denoise(signal=signal, save=True, out_path='/mnt/f/PhD/_Segments/Segmented15s/.selected_audio/ami')

ami_path = "/mnt/f/PhD/_Segments/44kHz/15s/ami"

print("Collecting...")
holistic_files = [file for file in glob.glob(os.path.join(ami_path, "*.*")) if fnmatch.fnmatch(file, "*Mix-Headset*")]

print("Done.\nStarting parallel...")
p_map(remove_noise, holistic_files)