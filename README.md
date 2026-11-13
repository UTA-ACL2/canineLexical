# canineLexical

Code for the paper **Phonetic and Lexical Discovery of Canine Vocalization** (Findings of EMNLP 2024).
Theron S. Wang\*, Xingyuan Li\*, Chunhao Zhang, Mengyue Wu, Kenny Q. Zhu.
[Paper](https://aclanthology.org/2024.findings-emnlp.816/)

The pipeline turns YouTube dog videos into dog "sentences" and then into phone sequences. It discovers "words" with an adaptor grammar and links each word to the dog's activities before and after it is uttered.

```
videos ─► 1 AudioSep ─► 2 PANNs sentence cut ─► 3 dataset (37,919 sentences)
       ─► 4 Dog-HuBERT + k-means (140 phones) ─► 5 adaptor grammar (words)
       ─► 6 TSN activities + causal strength (semantics)
```

## Models and data

Large files are hosted on the Hugging Face Hub:

- [`ArlingtonCL2/canine-lexical-models`](https://huggingface.co/ArlingtonCL2/canine-lexical-models): Dog-HuBERT (`dog_hubert.pt`), the 140-phone k-means model, the fine-tuned PANNs dog detector and the TSN activity classifier.
- [`ArlingtonCL2/canine-lexical-data`](https://huggingface.co/datasets/ArlingtonCL2/canine-lexical-data) (dataset): HuBERT manifests, phone transcriptions (`train.km`), adaptor-grammar inputs and parses, every word occurrence, and the activity labels.

```bash
pip install huggingface_hub
python download.py            # writes ./checkpoints and ./data
```

Audio and video are not redistributed. Sentence file names give the source YouTube or AudioSet video.

## Code

The scripts are the ones used for the paper, kept as they were run. Many of them contain absolute paths from our servers, both in their defaults and in the example commands in their docstrings. Change these to your own paths before running.

## Repository layout

| Folder | Paper section | What it does |
|---|---|---|
| [`1_audio_cleanup/`](1_audio_cleanup) | §2.1 | `separate.py` runs [AudioSep](https://github.com/Audio-AGI/AudioSep) with the text query `"Dog"` on every clip. |
| [`2_sentence_extraction/first_cut/`](2_sentence_extraction/first_cut) | §2.2 | `first_cut.py`: pretrained [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) Cnn14_DecisionLevelMax, "Dog" class, threshold 0.05, ±1 s offset. Finds the regions with dog sounds before separation. |
| [`2_sentence_extraction/second_cut/`](2_sentence_extraction/second_cut) | §2.2, Table 1 | `train.py` (with `dataset.py`, `loss.py`, `evaluate.py`) fine-tunes PANNs on 1,483 hand-labelled clips. `second_cut.py` runs the fine-tuned model on the separated audio (threshold 0.28) and merges detections less than 0.5 s apart into "sentences". `find_offset*.py` and `about_f1score.py` were used to choose the threshold. `show_train_result.py` plots the training logs. |
| [`2_sentence_extraction/remove_noise/`](2_sentence_extraction/remove_noise) | §2.2 | `remove_noise.py` drops sentences whose Dog score is below 0.1, or that have a non-dog tag in the top 10 tags scoring close to or above Dog (`--topk 10 --threshold -0.7`). |
| [`3_dataset/`](3_dataset) | §2.3 | `get_tsv.py` writes the fairseq `train.tsv`/`valid.tsv` (34,056 + 3,863 = 37,919 sentences). `build_context_info.py` maps each sentence back to its video and time span for step 6. |
| [`4_phone_recognition/`](4_phone_recognition) | §3.1 | Dog-HuBERT training and k-means. See [below](#4-dog-hubert-and-phones). |
| [`5_word_discovery/`](5_word_discovery) | §3.2 | Adaptor grammar with hybrid variational-MCMC inference ([PyAdaGram](https://github.com/kzhai/PyAdaGram), Zhai et al. 2014), plus data preparation and word extraction. |
| [`6_semantics/`](6_semantics) | §3.3 | `activity.py` labels the dog's activity with a fine-tuned TSN ([mmaction2](https://github.com/open-mmlab/mmaction2)) in a window from 5 s before to 5 s after each sentence. `statistics.py` implements Algorithm 1 (`analysis2`), counts activity–word co-occurrences (`generate_p`) and selects the top words (`get_top100`). |
| [`evaluation/`](evaluation) | §4 | Scripts that reproduce Table 3, Table 4 and the word statistics. |
| [`results/`](results) | §4.2 | `vocabulary_word_counts.json`: every word found by the adaptor grammar, with its count (7,326 words). |

## Step by step

### 1. Audio clean-up (AudioSep)

Clone AudioSep, download `audiosep_base_4M_steps.ckpt`, and copy `1_audio_cleanup/*.py` into `AudioSep/work/`.

```bash
cd AudioSep
python work/separate.py --input_path first_cut.jsonlines --output_root out/separation \
       --log_root log --checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt --device 0
```

### 2. Sentence extraction (PANNs)

Each of the three sub-folders is a `work/` folder that goes inside a clone of [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn). The scripts import `pytorch.models` and `utils.config` from that repo. `second_cut/train.py` also needs `pip install git+https://github.com/fgnt/sed_scores_eval.git`. The order is:

1. `first_cut.py` on the raw audio, using the pretrained Cnn14_DecisionLevelMax.
2. AudioSep (step 1) on the first-cut regions.
3. `train.py` to fine-tune PANNs on the labelled clips, then `second_cut.py` (threshold 0.28) to cut sentences.
4. `remove_noise.py --topk 10 --threshold -0.7` using Cnn14_16k.

The exact commands we ran are in the docstring at the top of each script.

### 3. Dataset

`get_tsv.py` (`split_dataset`) builds the fairseq manifests for HuBERT. The paper's dataset has 34,056 training and 3,863 validation sentences.

### 4. Dog-HuBERT and phones

We used the standard [fairseq HuBERT recipe](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert). These settings are taken from the config saved in the final checkpoint:

| | Stage 1 | Stage 2 |
|---|---|---|
| Initialisation | `hubert_base_ls960.pt` | stage 1, 80k steps |
| Targets | k-means, 100 clusters | k-means, 150 clusters on **layer 6** of stage 1 |
| Updates | 80k | 60k (`checkpoint_145_60000.pt`) |
| Learning rate | 1e-4 | 1e-4 (warm-up 3,200, label rate 50 Hz, max_tokens 1.4M, 2 GPUs) |

Phones come from **layer 11** of the stage-2 model, clustered with `MiniBatchKMeans` (K = 140; k-means++, `max_iter` 1000, `n_init` 20, batch 50,000). We used fairseq's `simple_kmeans` scripts:

```bash
python dump_hubert_feature.py <tsv_dir> train checkpoint_145_60000.pt 11 1 0 <feat_dir>
python learn_kmeans.py <feat_dir> train 1 60k_11th_km140.pkl 140 --percent 1.0
python dump_km_label.py <feat_dir> train 60k_11th_km140.pkl 1 0 <label_dir>   # -> train.km
```

The released `dog_hubert.pt` is `checkpoint_145_60000.pt` with the optimizer state removed by `trim_checkpoint.py`. Its features are identical to the original's.

`find_cluster.py` sweeps K and records inertia, silhouette and Calinski–Harabasz scores for MiniBatchKMeans.

The result is `train.km`: one line per sentence and one cluster ID per 20 ms frame. The 29 cluster IDs we treat as noise are listed in `evaluation/phone_durations.py`.

### 5. Word discovery (adaptor grammar)

`5_word_discovery/PyAdaGram/` is Ke Zhai's PyAdaGram. Its original README is included as `README_PyAdaGram.md`.

1. **Prepare the data** with `data_process.py`:
   - `read_transcription(train.km, mode='normal')` merges repeated frame labels into phones and writes `train.dat`.
   - `retain_limit_length_sentence(..., limit_length=100)` keeps sentences of at most 100 phones (31,294 sentences). It writes `train.dat`, `train_audio.dat` (the audio file of each sentence) and `train_ori.dat` (the raw frame labels). Copy `train.dat` to `truth.dat`.
2. **Grammar** (`data/grammar.unigram`): `Words → Word | Word Words` and `Word → Chars`, with `Word` adapted and `Chars → Char+` over the 140 phones. This is the `Sentence → Word+`, `Word → Phone+` grammar in the paper.
3. **Train.** The paper's run went over the data once, in 244 batches of 128 sentences:
   ```bash
   cd 5_word_discovery/PyAdaGram
   python -m launch_train --input_directory=../short-dog --output_directory=../ \
          --grammar_file=../short-dog/grammar.unigram --number_of_documents=31294 \
          --batch_size=128 --training_iterations=244 --tau=512 --kappa=0.8 \
          --number_of_processes=32
   ```
   This writes `Word.model-244.avg.test` (the parse of each sentence) and `.truth`.
4. **Extract words.** `word_validation.py::generate_words_clips_adagram` aligns the parses back to the audio. It writes a CSV of every word occurrence (word, file, start, end) and a JSON of word counts. The JSON is `results/vocabulary_word_counts.json`.

`plot_chart.py` draws the parse example in Figure 5.

### 6. Semantics

- **Activity recognition.** The TSN config is in `6_semantics/tsn/configs/tsn_shiba_0525.py` (TSN-R50, 16 classes, of which the paper reports 14). It was fine-tuned on 2,534 labelled clips. `activity.py` splits each video into stable shots (`TransnetV2.py`), labels 2-second clips from 5 s before to 5 s after each sentence, and merges consecutive clips with the same label. Its input is the per-sentence video info from `3_dataset/build_context_info.py`.
- **Causal strength.** `statistics.py::statistic()` reads the word CSV from step 5 and the activity file.
  - `analysis2()` (Algorithm 1, thresholds 0.5 and 0.3) puts each activity clip *before* (`left`), *during* (`overlap`) or *after* (`right`) each word.
  - `generate_p()` writes `c_{left,overlap,right}_by_{word,action}.csv`. "Left" counts give the reaction CS and "right" counts give the request CS.
  - `get_top100()` keeps the top 200 words, after removing words that are at least half noise or that occur fewer than 15 times.
  - `show_top100()` plots, for each word, the activity distribution minus the overall activity distribution (Figures 7–8).

## Reproducing the evaluation

```bash
cd evaluation
python phone_test/table3.py                                     # Table 3
python phone_durations.py ../data/phones/train.km               # Table 4
python word_statistics.py ../data/words/word_occurrences.csv.gz # Sec. 4.2 durations, Figure 6 word lengths
```

| Result | Script | Output |
|---|---|---|
| Table 3, phone listening test (140 same-phone + 150 different-phone pairs) | `phone_test/table3.py` | Tester 1: 58.28 / 62.86 / 54.00; Tester 2: 60.00 / 66.43 / 54.00; agreement: 71.38 / 75.00 / 68.00 |
| Table 4, phone durations | `phone_durations.py` | Dog 60.5 / 40 / 65.5 ms; Noise 37.3 / 20 / 35.3 ms; All 52.5 / 40 / 58.0 ms |
| Word duration, top 100 / 200 | `word_statistics.py` | Mean 0.064 / 0.072 s; variance 0.0036 / 0.0047 (the paper reports 0.0037 / 0.0047) |

Tables 1 and 5–7, Figures 3, 4 and 7–8, and the 87.1% human judgement were produced with the code above, but the intermediate files and plotting scripts are not part of this release.

## Citation

```bibtex
@inproceedings{wang-etal-2024-phonetic,
    title = "Phonetic and Lexical Discovery of Canine Vocalization",
    author = "Wang, Theron S.  and Li, Xingyuan  and Zhang, Chunhao  and Wu, Mengyue  and Zhu, Kenny Q.",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.816",
    doi = "10.18653/v1/2024.findings-emnlp.816",
    pages = "13972--13983",
}
```

## Acknowledgements

This repository includes PyAdaGram by Ke Zhai (`5_word_discovery/PyAdaGram`) and builds on AudioSep, PANNs, fairseq HuBERT, mmaction2 and TransNetV2.
