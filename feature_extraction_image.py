# librosa is a toolkit can extract feature of audio
import librosa
import librosa.display
from matplotlib import pyplot as plt
import os


def audio_data():
    audio_list = []
    sr_list = []
    for path, subdirs, files in os.walk('dataset/Data/genres_original/'):
        for name in subdirs:
            print(os.path.join(path, name))
            y, sr = librosa.load(os.path.join(path, name) + '/' + name + '.00000.wav', sr=22050)
            audio_data, _ = librosa.effects.trim(y)
            audio_list.append(audio_data)
            sr_list.append(sr)
    return audio_list, sr_list


def plot_wave(audio_data,sr_list,classi,index):

    fig = plt.figure(figsize=(6,2))
    librosa.display.waveshow(audio_data[index], sr=sr_list[index])
    plt.title('Blue wave_' + classi)
    plt.title("Wave_" + classi)
    plt.show()
    # plt.savefig will return empty image.
    fig.savefig('img/' + 'audio_wave_'+ classi)


def forier_transform(autdio_data,sr_list,classi,index):
    stft = librosa.stft(autdio_data[index])
    stft_db = librosa.amplitude_to_db(abs(stft))
    fig = plt.figure(figsize=(6,2))
    librosa.display.specshow(stft_db, sr=sr_list[index], x_axis='time', y_axis='hz')
    plt.title("Forier Transform_" + classi)
    plt.colorbar()
    plt.show()
    fig.savefig('img/' + 'forier_transform' + classi)

def mfcc(autdio_data, sr_list,classi,index):
    mfccs = librosa.feature.mfcc(autdio_data[index], sr=sr_list[index])
    fig = plt.figure(figsize=(6,2))
    print(mfccs.shape)
    # Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=sr_list[index], x_axis='time')
    plt.title("MFCC_" + classi)
    plt.show()
    fig.savefig('img/' + 'mfcc'+ classi)

def chroma(autdio_data, sr_list,classi,index):
    chromagram = librosa.feature.chroma_stft(autdio_data[index], sr=sr_list[index], hop_length=3000)
    fig = plt.figure(figsize=(6,2))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=3000, cmap='coolwarm')
    plt.title("Chroma_" + classi)
    plt.show()
    fig.savefig('img/' + 'chroma'+ classi)



audio_data,sr_list = audio_data()
plot_wave(audio_data,sr_list,'Blue',0)
forier_transform(audio_data,sr_list,'Blue',0)
mfcc(audio_data,sr_list,'Blue',0)
chroma(audio_data,sr_list,'Blue',0)

plot_wave(audio_data,sr_list,'Classical',1)
forier_transform(audio_data,sr_list,'Classical',1)
mfcc(audio_data,sr_list,'Classical',1)
chroma(audio_data,sr_list,'Classical',1)


plot_wave(audio_data,sr_list,'Metal',2)
forier_transform(audio_data,sr_list,'Metal',2)
mfcc(audio_data,sr_list,'Metal',2)
chroma(audio_data,sr_list,'Metal',2)