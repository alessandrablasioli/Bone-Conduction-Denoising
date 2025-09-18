from datasets import load_dataset
import numpy as np
from scipy.io import wavfile
import pandas as pd
import io
from IPython.display import Audio
import os
import torchaudio
import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pydub import AudioSegment
import random
import os
import torchaudio
from scipy.io import wavfile


def make_noise(speech, speechbc , list_noise, list_noisebc):
    """
        Creates noisy files starting from a clean waveform, randomply choises the noise from the 
        list_noise list. The process is done for both bone conduction and air conduction waveforms of the same speech.

        Returns the noisy waveforms of air and bone conduction audio. The noise will be the same for both.
    Args:
        - speech: the waveform of the clean air conduction speech
        - speechbc: the waveform of the clean bone conduction speech
        - list_noise: the list of noises that we can mix to the air conduction waveform
        - list_noisebc: the list of noises that we can mix to the bone conduction waveform
    """

    length = speech.shape[-1]

    #Choose a noise in a random way
    noise_array = random.choice(list_noise)
    if noise_array.shape[-1] < length:
        print('The selected noise is shorter then the speech, reselecting...')
        noise_array = random.choice(list_noise)

    #Get the index of the chosen noise 
    index = next((i for i, el in enumerate(list_noise) if np.array_equal(el, noise_array)), None)


    #Get same noise for the bc waveform 
    noisebc = list_noisebc[index]
    start = random.randint(0, (noise_array.shape[-1] - length))
    noise = noise_array[:, start:start + length]
    noise_bc = noisebc[:, start:start + length]

    #Mix noise with speech
    noisy_speech = speech + noise
    noisy_speech = noisy_speech.numpy().flatten()

    noisy_speechbc = speechbc + noise_bc
    noisy_speechbc = noisy_speechbc.numpy().flatten()
    
    return noisy_speech, noisy_speechbc


'''

Insert all the paths to your folders of:
    - speech_path: clean speech files for ac
    - noise_path: noise speechless files for ac
    - speech_path_bc: clean speech files for bc
    - noise_path_bc: noise speechless files for bc

Also add paths to your new folders in noisedir for ac and noisedirbc for bc noisy speech files.
Note: To execute for Validation and Test set run again, but change file paths
'''

speech_path = '/home/alessandrab/python_files/clean_air_train'
noise_path = '/home/alessandrab/python_files/noise_sl_air_train'
noisedir = '/home/alessandrab/python_files/noise_air_train_bcn'
speech_path_bc = '/home/alessandrab/python_files/clean_bone_train'
noise_path_bc = '/home/alessandrab/python_files/noise_sl_bone_train'
noisedirbc = '/home/alessandrab/python_files/noise_bone_train_bcn'



speech_list = []
noise_list = []
speech_listbc = []
noise_listbc = []


'''

Modify the extract_number function according to the naming of your files, 
then sort the bc and ac file names in file_names and file_namesbc

'''
def extract_number(file_name):
    return int(file_name.split('_')[1].split('.')[0])

file_names = sorted(
    (f for f in os.listdir(speech_path) if os.path.isfile(os.path.join(speech_path, f))),
    key=extract_number
)
file_namesbc = sorted(
    (f for f in os.listdir(speech_path) if os.path.isfile(os.path.join(speech_path, f))),
    key=extract_number
)

for file in file_names:
    
    file_path = os.path.join(speech_path, file)
    if file_path.endswith('.wav'):
        waveform, sample_rate = torchaudio.load(file_path)
        speech_list.append((waveform))

for file in os.listdir(noise_path):
    file_path = os.path.join(noise_path, file)
    if file_path.endswith('.wav'):
        waveform, sample_rate = torchaudio.load(file_path)
        noise_list.append((waveform))

for file in file_namesbc:
    file_path = os.path.join(speech_path_bc, file)
    if file_path.endswith('.wav'):
        waveform, sample_rate = torchaudio.load(file_path)
        speech_listbc.append((waveform))

for file in os.listdir(noise_path):
    file_path = os.path.join(noise_path_bc, file)
    if file_path.endswith('.wav'):
        waveform, sample_rate = torchaudio.load(file_path)
        noise_listbc.append((waveform))

os.makedirs(noisedir, exist_ok=True)
os.makedirs(noisedirbc, exist_ok=True)


''''

Iterate on the noises lists to mix clean bc and ac speech with noise and save them in the new folders

'''

for (i, speech_waveform), (j , speech_waveformbc) in zip(enumerate(speech_list),enumerate(speech_listbc)):
    #for j in range(10):
        noisy_speech, noisy_speechbc = make_noise(speech_waveform, speech_waveformbc, noise_list, noise_listbc)
        
        output_filename = os.path.join(noisedir, f"output{i}.wav") #"output{i}_version_{j}.wav"
        wavfile.write(output_filename, sample_rate, noisy_speech)

        output_filenamebc = os.path.join(noisedirbc, f"output{j}.wav") #"output{i}_version_{j}.wav"
        wavfile.write(output_filenamebc, sample_rate, noisy_speechbc)