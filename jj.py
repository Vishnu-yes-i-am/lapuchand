
# !pip install -q torchaudio omegaconf
# !pip install aksharamukha

import os
import torch
from pprint import pprint
from omegaconf import OmegaConf
# from IPython.display import Audio, display
from aksharamukha import transliterate
# from __future__ import division
import numpy as np
import threading
from scipy.io.wavfile import write

torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
models = OmegaConf.load('latest_silero_models.yml')

# Loading model
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='indic',
                                     speaker='v3_indic')
data = []
for i in range(118):
  filename = "file"+str(i)+".txt"
  with open("./files/"+filename , "r",encoding="utf8") as f :
    data.append(f.read())
print("DONE!!")
sample_rate=48000;
def change(data,filename):
    roman_text = transliterate.process('Devanagari', 'ISO', data)
    audio = model.apply_tts(roman_text,speaker='hindi_male') 
    write(filename, sample_rate, audio.numpy())
# for i in range(10):
#     el=data[i]
#     filename = './audios/file'+str(i)+".wav"
#     change(el,filename)
#     print(str(i) + " added")
threads = []
for i in range(10):
    el=data[i]
    filename = './audios/file'+str(i)+".wav"
    t = threading.Thread(target=change,args=(el,filename,))
    t.start()
    threads.append(t)
    print(str(i) + " added")

for t in threads:
    t.join()
print("DONE!!!")