import numpy as np
import wave
import librosa
import matplotlib.pyplot as plt



def plot_chart1():
    audio_file_path = "/bigdrive/dog/audios/[UCDiSE0bK_QjMLViaI2aTrrA]_[qZZOt2Jxt1U]_00008_00001.wav"
    raw_sentence = "39 119 10 119 10 10 119 4 113 83 83 83 25 9 9 9 15 15 15 15 15 15 15 105 105 105 105 105 4 4 4"
    train_sentence = "39 119 10 119 10 119 4 113 83 25 9 15 105 4"
    parse_sentence = "39 119 10 119 10 1194113 83 25915 105 4"
    
    segment = [1, 2, 3, 4, 6, 9, 12, 16, 23, 28]
    

    
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_file_path)
    
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    vertical_line = [duration * i / 31 * sample_rate for i in segment]
    print(duration)
    # Create time axis
    time = librosa.times_like(audio_data, sr=sample_rate)
    print(time)
    # Plot wave form
    plt.plot(audio_data)
    plt.vlines(vertical_line, -1, 1, colors='r', linestyles='dashed')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of Audio')
    plt.savefig('chart1.png')
    plt.close()
    
    
plot_chart1()