'''
git clone https://github.com/sooftware/KoSpeech.git
cd KoSpeech
pip install -r requirements.txt
python setup.py install

pip install pyaudio
'''
import torch
import pyaudio
import numpy as np
from kospeech.data.audio.core import load_audio
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.models import SpeechToText
from kospeech.utils import load_checkpoint


model_path = 'path/to/your/model.pth'
vocab_path = 'path/to/your/vocab.txt'


vocab = KsponSpeechVocabulary(vocab_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_checkpoint(model_path).to(device)
model.eval()


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024


audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)

print("듣는 중 입니다...")

try:
    while True:
        frames = []

        
        for _ in range(0, int(RATE / CHUNK * 5)):  
            data = stream.read(CHUNK)
            frames.append(data)

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.recognize(audio_tensor)
            result = vocab.label_to_string(output)

        print("텍스트로 변환:", result)

except KeyboardInterrupt:
    print("종료...")


stream.stop_stream()
stream.close()
audio_interface.terminate()
