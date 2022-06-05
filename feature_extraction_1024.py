import os
import librosa
import numpy as np
import shutil
# reference https://github.com/CaifengLiu/music-genre-classification

def get_enforce_shape(dataset_rootpath, target_sr):
    GENRES = sorted(os.listdir(dataset_rootpath))
    for genre in GENRES:
        genre_path = os.path.join(dataset_rootpath, genre)
        for file in os.listdir(genre_path):
            audio, sr = librosa.load(os.path.join(genre_path, file), sr=None)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            enforce_shape = len(audio)
            return enforce_shape

def load_dataset(dataset_rootpath, target_sr=22050):
    GENRES = sorted(os.listdir(dataset_rootpath))
    X = []
    y = []
    count = 0
    enforce_shape = get_enforce_shape(dataset_rootpath, target_sr)
    for genre_index, genre in enumerate(GENRES):
        label = genre_index + 1
        genre_path = os.path.join(dataset_rootpath, genre)
        for file in os.listdir(genre_path):
            audio, sr = librosa.load(os.path.join(genre_path, file), sr=None)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if len(audio) < enforce_shape:
                audio = np.append(audio, np.zeros(shape=(enforce_shape - len(audio))))
            if len(audio) > enforce_shape:
                audio = audio[:enforce_shape]
            X.append(audio)
            y.append(label)
            count += 1
            if count % 100 == 0:
                print('Already process %d music' % count)
    return np.array(X, dtype=np.float32), np.array(y)


def get_stft_feature(X, frame_size, frame_shift_len):
    print('Starting extract stft feature......')
    stft_feature = []
    count = 0
    for audio in X:
        audio_stft = librosa.stft(audio, n_fft=frame_size, hop_length=frame_shift_len)
        audio_stft = librosa.amplitude_to_db(audio_stft)
        audio_stft = audio_stft.T
        stft_feature.append(audio_stft)
        count += 1
        if count % 100 == 0:
            print('Already process %d music' % count)
    return np.array(stft_feature, dtype=np.float32)


def get_melspec_feature(X, target_sr, frame_size, frame_shift_len, n_mels):
    print('Start extract melspectrogram feature......')
    melspec_feature = []
    count = 0
    for audio in X:
        audio_melspec = librosa.feature.melspectrogram(audio, sr=target_sr, n_fft=frame_size,
                                                       hop_length=frame_shift_len)
        audio_melspec = librosa.power_to_db(audio_melspec)
        audio_melspec = audio_melspec.T
        melspec_feature.append(audio_melspec)
        count += 1
        if count % 100 == 0:
            print('Already process %d music' % count)
    return np.array(melspec_feature, dtype=np.float32)


def get_mfcc_feature(X, target_sr, frame_size, frame_shift_len, n_mfcc):
    print('Start extract mfcc feature......')
    mfcc_feature = []
    count = 0
    for audio in X:
        audio_mfcc = librosa.feature.mfcc(audio, n_fft=frame_size, hop_length=frame_shift_len, n_mfcc=n_mfcc)
        audio_mfcc = audio_mfcc.T
        mfcc_feature.append(audio_mfcc)
        count += 1
        if count % 100 == 0:
            print('Already process %d music' % count)
    return np.array(mfcc_feature, dtype=np.float32)


def label_onehot_encode(y):
    y_onehot = []
    y_unique = sorted(set(y))
    num_classes = len(y_unique)
    for label in y:
        tmp = [0] * num_classes
        encode_index = y_unique.index(label)
        tmp[encode_index] = 1
        y_onehot.append(tmp)
    return np.array(y_onehot)


def refreshDir(path, file_name):
    shutil.rmtree(path)  # 直接删除该文件夹
    os.mkdir(path)  # 创建空文件夹
    full_path = path + file_name + '.npy'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.close()


target_sr = 22050
frame_size = 2048
frame_shift_len = 1024
dataset_rootpath = './dataset/Data/genres_original'

X, y = load_dataset(dataset_rootpath, target_sr=22050)
y_onehot = label_onehot_encode(y)
stft_feature = get_stft_feature(X, frame_size, frame_shift_len)
melspec_feature = get_melspec_feature(X, target_sr, frame_size, frame_shift_len, n_mels=128)
mfcc_feature = get_mfcc_feature(X, target_sr, frame_size, frame_shift_len, n_mfcc=25)
np.save('./dataset/GTZAN/raw_labes.npy', y)
np.save('./dataset/GTZAN/onehot_labels.npy', y_onehot)
np.save('./dataset/GTZAN/raw_audio.npy', X)
np.save('./dataset/GTZAN/without_split_feature/stft_feature_2048.npy', stft_feature)
np.save('./dataset/GTZAN/without_split_feature/melspec_feature_2048.npy', melspec_feature)


frame_size = 1024
frame_shift_len = 512
X = np.load('./dataset/GTZAN/raw_audio.npy')
stft_feature = get_stft_feature(X, frame_size, frame_shift_len)
melspec_feature = get_melspec_feature(X, target_sr, frame_size, frame_shift_len, n_mels=128)
mfcc_feature = get_mfcc_feature(X, target_sr, frame_size, frame_shift_len, n_mfcc=25)
np.save('./dataset/GTZAN/without_split_feature/stft_feature_1024.npy', stft_feature)
np.save('./dataset/GTZAN/without_split_feature/melspec_feature_1024.npy', melspec_feature)

frame_size = 4096
frame_shift_len = 2048
X = np.load('./dataset/GTZAN/raw_audio.npy')
stft_feature = get_stft_feature(X, frame_size, frame_shift_len)
melspec_feature = get_melspec_feature(X, target_sr, frame_size, frame_shift_len, n_mels=128)
mfcc_feature = get_mfcc_feature(X, target_sr, frame_size, frame_shift_len, n_mfcc=25)
np.save('./dataset/GTZAN/without_split_feature/stft_feature_4096.npy', stft_feature)
np.save('./dataset/GTZAN/without_split_feature/melspec_feature_4096.npy', melspec_feature)