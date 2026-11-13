import os
import numpy as np
import argparse


def read_transcription(transcription_file: str, mode: str='normal'):
    with open(transcription_file, "r") as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        combined_line = combine_phonemes(line.strip().split(" "), mode=mode)
        processed_lines.append(" ".join(combined_line))
    
    parent_path = os.path.dirname(transcription_file)
    new_file_path = os.path.join(parent_path, f"train_{mode}.dat")

    with open(new_file_path, "w") as f:
        f.write("\n".join(processed_lines))
        
        
def combine_phonemes(phonemes: list, mode: str='normal', single_length: int=1, strong_length: int=1) -> list:
    combined = []
    if mode == 'normal':
        #==============================================================
        # Combine continuous same phonemes together
        # e.g. ['a', 'a', 'b', 'b', 'b', 'c'] -> ['a', 'b', 'c']
        # e.g. ['a', 'a', 'b', 'a', 'a', 'c'] -> ['a', 'b', 'a', 'c']
        #==============================================================
        for i in range(len(phonemes)):
            if i == 0:
                combined.append(phonemes[i])
            else:
                if phonemes[i] == phonemes[i-1]:
                    continue
                else:
                    combined.append(phonemes[i])
    elif mode == 'transform':
        #==============================================================
        # also merge single phoneme into another phoneme
        # e.g. ['a', 'a', 'b', 'a', 'a', 'c'] -> ['a', 'c']
        #      |--------|  |  |--------|
        #        strong a   b   strong a
        # single_length: the length of transforming single phoneme
        # strong_length: the length of strong phoneme
        # in this case, single_length = 1, strong_length = 2
        #==============================================================
        phonemes = ['a' for _ in range(strong_length)] + phonemes + ['a' for _ in range(strong_length)]
        for i in range(len(phonemes)):
            if i < strong_length:
                continue
            elif i >= len(phonemes) - strong_length:
                break
            else:
                previous_strong_phoneme = _get_strong_phoneme(phonemes[i-strong_length:i])
                next_strong_phoneme = _get_strong_phoneme(phonemes[i+1:i+1+strong_length])
                phoneme = phonemes[i]
                if previous_strong_phoneme is not None and next_strong_phoneme is not None and previous_strong_phoneme == next_strong_phoneme:
                    phoneme = previous_strong_phoneme
                
                if i == strong_length:
                    combined.append(phoneme)
                else:
                    if phoneme == combined[-1]:
                        continue
                    else:
                        combined.append(phoneme)
    else:
        raise ValueError("Invalid mode")
    
    return combined


def statistics_phonemes(phonemes_file: str):
    with open(phonemes_file, "r") as f:
        lines = f.readlines()
    phonemes = []
    for line in lines:
        phonemes.extend(line.strip().split(" "))
    
    phoneme_dict = {}
    for phoneme in phonemes:
        if phoneme in phoneme_dict:
            phoneme_dict[phoneme] += 1
        else:
            phoneme_dict[phoneme] = 1
    sorted_phonemes = sorted(phoneme_dict.items(), key=lambda x: x[0])
    print(sorted_phonemes, len(sorted_phonemes))
    
    sentence_length = []
    for line in lines:
        line_phonemes = line.strip().split(" ")
        sentence_length.append(len(line_phonemes))
    
    mean_length = sum(sentence_length) / len(sentence_length)
    max_length = max(sentence_length)
    min_length = min(sentence_length)
    median_length = sorted(sentence_length)[len(sentence_length)//2]
    q1 = np.percentile(sentence_length, 25)
    q3 = np.percentile(sentence_length, 75)

    print("q1:", q1)
    print("q3:", q3)
    variance = sum([(l-mean_length)**2 for l in sentence_length]) / len(sentence_length)
    print("-----")
    print("mean_length:", mean_length/50)
    print("max_length:", max_length/50)
    print("min_length:", min_length/50)
    print("median_length:", median_length/50)
    print("variance:", variance)
    return phoneme_dict


def _get_strong_phoneme(phonemes: list):
    if all(phonemes[0] == p for p in phonemes):
        return phonemes[0]
    else:
        return None


def find_longest_sentence(phonemes_file: str, video_meta_file: str, top_n: int=10):
    with open(phonemes_file, "r") as f:
        lines = f.readlines()
        
    with open(video_meta_file, "r") as f:
        video_meta = f.readlines()
    print(video_meta[0])
    sentences_length = []
    for line in lines:
        sentences_length.append(len(line.strip().split(" ")))
    
    longest_index = sorted(range(len(sentences_length)), key=lambda x: sentences_length[x], reverse=True)[:top_n]
    audio_names = [video_meta[i+1].split("\t")[0] for i in longest_index]
    print(audio_names)
    return audio_names


def find_number_endwith_phoneme(phonemes_file: str, phoneme: str):
    with open(phonemes_file, "r") as f:
        lines = f.readlines()
    phoneme_count = 0
    for line in lines:
        if line.strip().split(" ")[-1] == phoneme:
            phoneme_count += 1
    print(phoneme_count)
    return phoneme_count


def retain_limit_length_sentence(phonemes_file: str, mapping_file: str, output_dir: str, limit_length: int):
    train_data_file = os.path.join(phonemes_file, "train.dat")
    with open(train_data_file, "r") as f:
        lines = f.readlines()
    with open(mapping_file, "r") as f:
        meta_lines = f.readlines()[1:]
    with open("data/dog-phone/train.km", "r") as f:
        ori_lines = f.readlines()
    
    
    audioname_list = [line.split("\t")[0] for line in meta_lines]
    processed_lines = []
    processed_audio_list = []
    processed_ori_list = []
    assert len(lines) == len(audioname_list), "The number of lines and audio names should be the same"
    for line, audio_name, ori in zip(lines, audioname_list, ori_lines):
        if len(line.strip().split(" ")) <= limit_length:
            processed_lines.append(" ".join(line.strip().split(" ")))
            processed_audio_list.append(audio_name)
            processed_ori_list.append(ori.strip())
    
    output_file = os.path.join(output_dir, "train.dat")
    output_audio_file = os.path.join(output_dir, "train_audio.dat")
    output_ori_file = os.path.join(output_dir, "train_ori.dat")
    # with open(output_file, "w") as f:
    #     f.write("\n".join(processed_lines))
    # with open(output_audio_file, "w") as f:
    #     f.write("\n".join(processed_audio_list))
    with open(output_ori_file, "w") as f:
        f.write("\n".join(processed_ori_list))
    


def add_noise(input_dir: str, output_dir: str, num_noise_phonemes: int, per_noise_frames: float):
    # train.dat -> train_noise.dat
    total_phonemes = []
    with open(os.path.join(input_dir, "train.dat"), "r") as f:
        lines = f.readlines()
    for line in lines:
        total_phonemes.extend(line.strip().split(" "))
        total_phonemes.append('bar')

    noise = [str(i) for i in range(10, num_noise_phonemes+11)]
    
    for _ in range(int(len(total_phonemes) * (per_noise_frames / (1 - per_noise_frames)))):
        random_index = np.random.randint(0, len(total_phonemes))
        random_phoneme = np.random.choice(noise)
        total_phonemes.insert(random_index, random_phoneme)
        
    sentences = []
    sentence = ""
    for phoneme in total_phonemes:
        if phoneme == 'bar':
            sentences.append(sentence)
            sentence = ""
        else:
            sentence += phoneme + " "
    with open(os.path.join(output_dir, "train.dat"), "w") as f:
        f.write("\n".join(sentences))
    
        
def delete_noise(input_dir: str, output_dir: str):
    noise_phonemes = ['4', '10', '20', '23', '24', '32', '37', '50', '52', '53', '54', '65', '71', '72', '78', '79', '82', '85', '87', '89', '108', '109', '119', '120', '122', '134', '137']
     # train.dat -> train_without_noise.dat
    with open(os.path.join(input_dir, "train.dat"), "r") as f:
        lines = f.readlines()
    processed_lines = []
    phoneme_list = []
    for line in lines:
        for phoneme in line.strip().split(" "):
            if phoneme in noise_phonemes:
                continue
            else:
                if phoneme_list and phoneme_list[-1] == phoneme:
                    continue
                else:
                    phoneme_list.append(phoneme)
        if phoneme_list:
            processed_lines.append(" ".join(phoneme_list))
        phoneme_list = []
    output_file = os.path.join(output_dir, "train.dat")
    with open(output_file, "w") as f:
        f.write("\n".join(processed_lines))


if __name__ == "__main__":
    # dt = read_transcription("data/dog-phone/train.km", mode='normal')
    # statistics_phonemes("data/dog-phone/train_normal.dat")
    # statistics_phonemes("data/dog-phone/train.km")
    # find_longest_sentence("data/dog-phone/train.km", "data/tsv/train.tsv", top_n=10)
    # find_number_endwith_phoneme("data/dog-phone/train.km", "4")
    # retain_limit_length_sentence("data/dog-phone", "data/tsv/train.tsv", "short-dog", limit_length=100)
    # add_noise("brent-phone", "noisy_brent", num_noise_phonemes=20, per_noise_frames=0.4)
    # delete_noise("data/dog-phone-l127-normal", "dog_without_noise")
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", type=str, help="The input directory")
    # parser.add_argument("--mode", type=str, help="The mode of processing")
    
    # args = parser.parse_args()
    # read_transcription(args.input_dir, mode=args.mode)
    delete_noise("/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/short-dog", "/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/short-dog-no-noise")
    