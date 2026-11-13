import os, math
import auditok, librosa, wave
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json
import csv
import pandas as pd
import shutil


def count_word_number(output_folder: str):
    word_count = defaultdict(int)
    
    # load data
    for file in os.listdir(output_folder):
        if file.endswith(".test"):
            with open(os.path.join(output_folder, file), "r") as f:
                prediction_lines = f.readlines()
        elif file.endswith(".truth"):
            with open(os.path.join(output_folder, file), "r") as f:
                truth_lines = f.readlines()
                
    for prediction, truth in zip(prediction_lines, truth_lines):
        
        prediction = prediction.strip().split()
        truth = truth.strip().split()
        
        idx_truth, idx_pred = 0, 0
        word = ""
        word_with_dash = ""
        while idx_truth < len(prediction):
            word += truth[idx_truth]
            word_with_dash += truth[idx_truth]
            
            if prediction[idx_pred] == word:
                word_count[word_with_dash] += 1
                idx_pred += 1
                idx_truth += 1
            else:
                word_with_dash += "-"
                idx_truth += 1
    
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    print(sorted_word_count[:100])
    

def find_file_name(sentence_file: str, search_word: str, map_file: str="data/tsv/train.tsv"):
    
    with open(sentence_file, "r") as f:
        sentence_lines = f.readlines()
    
    with open(map_file, "r") as f:
        lines = f.readlines()
    
    for idx, sline in enumerate(sentence_lines):
        if search_word in sline:
            print(lines[idx+1].split("\t")[0])
            print(sline)


def get_word_vol_from_energy(mapping_file: str, trans_file: str, energy_threshold=-1):
    """
    Get word volcabulary from audio energy
    """
    slope, intercept = 5.996613, -434.098864
    # load data
    audio_folder = "/bigdrive/dog/wav_audios"
    with open(mapping_file, "r") as f:
        lines = f.readlines()
        audio_files = [line.strip().split("\t")[0] for line in lines[1:]]
    with open(trans_file, "r") as f:
        lines = f.readlines()
        transcriptions = [line.strip().split(" ") for line in lines]
    
    # initialize word dictionary
    word_dictionary = defaultdict(int)
    dictionary_log = [["word", "filename", "word_path", "start_time", "end_time"]]
    
    # travrse audio files
    for audio_file_name, transcription in zip(audio_files, transcriptions):
        audio_file_path = os.path.join(audio_folder, audio_file_name)

        audio_region = auditok.load(audio_file_path)
        audio_duration_sec = audio_region.duration
        energy = auditok.signal.calculate_energy_single_channel(audio_region.samples, audio_region.sample_width)

        if energy_threshold < 0:
            energy_threshold = energy * slope + intercept
        else:
            energy_threshold = energy_threshold
        
        region = audio_region.split(
            min_dur=0.01,     # minimum duration of a valid audio event in seconds
            max_dur=1,       # maximum duration of an event
            max_silence=0.01, # maximum duration of tolerated continuous silence within an event
            energy_threshold=energy_threshold # threshold of detection
        )
        num_frames = len(transcription)
        for _, r in enumerate(region):
            start_frame_idx = math.floor(r.meta.start/audio_duration_sec*num_frames)
            end_frame_idx = math.ceil(r.meta.end/audio_duration_sec*num_frames)
            
            # record meta information
            combined_word = []
            for phoneme in transcription[start_frame_idx:end_frame_idx]:
                if len(combined_word) < 1:
                    combined_word.append(phoneme)
                else:
                    if phoneme == combined_word[-1]:
                        continue
                    else:
                        combined_word.append(phoneme)
            word_seq = "-".join(combined_word)
            word_dictionary[word_seq] += 1
            dictionary_log.append([word_seq, audio_file_name, audio_file_path, r.meta.start, r.meta.end])
        print(f"Finish processing {audio_file_name}")
    
    sorted_word_dictionary = sorted(word_dictionary.items(), key=lambda x: x[1], reverse=True)
    # Store word_dictionary as a JSON file
    with open(f"energy_vol/word_dictionary_et{energy_threshold}.json", "w") as json_file:
        json.dump(sorted_word_dictionary, json_file)

    # Store dictionary_log as a CSV file
    with open(f"energy_vol/dictionary_log_et{energy_threshold}.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(dictionary_log)


def cut_words_from_audio(mapping_file: str, output_folder: str):
    with open(mapping_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in tqdm(reader):
            audio_name = row[1]
            audio_file_path = row[2]
            word_name = row[0]
            start_time = float(row[3])
            end_time = float(row[4])
            
            output_subfolder = os.path.join(output_folder, word_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
                
            output_file_path = os.path.join(output_subfolder, f"{word_name}_{audio_name}.wav")
            with wave.open(audio_file_path, 'rb') as wave_file:
                wave_params = wave_file.getparams()
                wave_file.setpos(int(start_time * wave_params.framerate))
                frames = wave_file.readframes(int((end_time - start_time) * wave_params.framerate))
                with wave.open(output_file_path, 'wb') as output_wave_file:
                    output_wave_file.setparams(wave_params)
                    output_wave_file.writeframes(frames)
    
    print("Finish cutting words from audio files")
            

def plot_word_activity(activity_folder: str):
    for filename in os.listdir(activity_folder):
        file_basename = os.path.splitext(filename)[0]
        file_path = os.path.join(activity_folder, filename)
        
        data = pd.read_csv(file_path)

        for _, row in data.iterrows():
            word = row[0]
            # P(Activity | Word)
            plt.figure(figsize=(30, 6))
            plt.plot(row[1:-2])
            plt.title(f"Probability of Activity given {word}")
            output_folder = os.path.join("energy_vol/top100plot", file_basename)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(os.path.join(output_folder, f"{word}.png"))
            plt.close()
        
        for idx in range(1, data.shape[1]):
            # P(Word | Activity)
            plt.figure(figsize=(20, 6))
            plt.plot(data.iloc[:-1, idx])
            plt.title(f"Probability of {data.columns[idx]} given Activity")
            plt.savefig(os.path.join(output_folder, f"{data.columns[idx]}.png"))
            plt.close()


def select_plots():
    meta_file = "energy_vol/top100plot/selected_plot_list.txt"
    with open(meta_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            filepath = line.strip()
            filefolders = filepath.split("/")
            output_folder = os.path.join("energy_vol/top100plot/selected_plots", filefolders[-2])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            shutil.copy(filepath, output_folder)
            

def get_word_vol_from_adagram(result_folder: str, mapping_file: str):
    # read data
    with open(mapping_file, "r") as f:
        lines = f.readlines()
        audio_files = [line.strip().split("\t")[0] for line in lines[1:]]
    for filename in os.listdir(result_folder):
        if filename.endswith(".test"):
            with open(os.path.join(result_folder, filename), "r") as f:
                prediction_lines = f.readlines()
        elif filename.endswith(".truth"):
            with open(os.path.join(result_folder, filename), "r") as f:
                ori_lines = f.readlines()
    

def generate_words_clips_adagram(input_folder: str, ori_path: str, audio_path: str, output_folder: str, num_samples: int=4):
    word_count = defaultdict(int)
    audio_folder = "/bigdrive/dog/wav_audios"
    
    # load data
    for file in os.listdir(input_folder):
        if file.endswith(".test"):
            with open(os.path.join(input_folder, file), "r") as f:
                prediction_lines = f.readlines()
        elif file.endswith(".truth"):
            with open(os.path.join(input_folder, file), "r") as f:
                truth_lines = f.readlines()

        with open(ori_path, "r") as f:
            ori_lines = f.readlines()

        with open(audio_path, "r") as f:
            audio_lines = f.readlines()
                
    dictionary_log = [["word", "filename", "word_path", "start_time", "end_time"]]
     
    for idddx in range(len(ori_lines)):
        prediction = prediction_lines[idddx*num_samples]
        truth = truth_lines[idddx*num_samples]
        ori = ori_lines[idddx]
        audio_p = audio_lines[idddx]
        
        prediction = prediction.strip().split(" ")
        truth = truth.strip().split(" ")
        ori = ori.strip().split(" ")
        
        # get audio duration
        audio_file_path = audio_p.strip()
        region = auditok.load(os.path.join(audio_folder, audio_file_path))
        duration = region.duration
        
        timestamps_for_combined_phonemes = [0]
        combined_sen = []
        for idx, phoneme in enumerate(ori):
            if len(combined_sen) < 1:
                combined_sen.append(phoneme)
            else:
                if phoneme == combined_sen[-1]:
                    continue
                else:
                    combined_sen.append(phoneme)
                    timestamps_for_combined_phonemes.append(idx)
        timestamps_for_combined_phonemes.append(len(ori))
        assert len(timestamps_for_combined_phonemes) == len(truth) + 1, "The number of timestamps should be the same as the number of words"
        
        idx_truth, idx_pred = 0, 0
        word = ""
        word_with_dash = ""
        last_word_end = 0
        while idx_truth < len(truth):
            word += truth[idx_truth]
            word_with_dash += truth[idx_truth]
            
            if prediction[idx_pred] == word:
                word_count[word_with_dash] += 1
                start_time = timestamps_for_combined_phonemes[last_word_end] / len(ori) * duration
                end_time = timestamps_for_combined_phonemes[idx_truth+1] / len(ori) * duration
                dictionary_log.append([word_with_dash, audio_file_path, os.path.join(audio_folder, audio_file_path), start_time, end_time])
                
                word = ""
                word_with_dash = ""
                last_word_end = idx_truth + 1
                idx_pred += 1
                idx_truth += 1
            else:
                word_with_dash += "-"
                idx_truth += 1
        if idddx % 100 == 0:
            print(f"Finish processing {idddx}th audio")
    
    csv_output_file = os.path.join(output_folder, input_folder.split("/")[-1] + ".csv")
    with open(csv_output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(dictionary_log)
    
    # Store sorted_word_count as a JSON file
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    print(sorted_word_count[:100])
    json_output_file = os.path.join(output_folder, input_folder.split("/")[-1] + ".json")
    with open(json_output_file, "w") as json_file:
        json.dump(sorted_word_count, json_file)
    


if __name__ == "__main__":
    # count_word_number("dog_without_noise/24Jun10-235416-D28702-P10-S10-B40-O1000-t64-k0.75-Ggrammar.unigram")
    # find_file_name("data/dog-phone/train.dat", "4 96 70 82 39 112 70 97 105 82 70 4 112")
    # get_word_vol_from_energy("data/tsv/train.tsv", "data/dog-phone/train.km", 60)
    # cut_words_from_audio("energy_vol/dictionary_log.csv", "energy_vol/words")
    # plot_word_activity("energy_vol/top100")
    # select_plots()
    generate_words_clips_adagram(input_folder="short-dog/24Jun14-215338-D31294-P10-S10-B128-O244-t512-k0.8-Ggrammar.unigram", ori_path="short-dog/train_ori.dat", audio_path="short-dog/train_audio.dat", output_folder="short-dog/words/words_info")
                