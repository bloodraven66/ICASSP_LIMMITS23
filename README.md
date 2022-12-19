# ICASSP_LIMMITS23
This is the repository for baselines and evaluation scripts for the ICASSP Signal processing Grand Challenge - LIghtweight, Multi-speaker, Multi-lingual Indic TTS (LIMMITS) 2023.

For any queries, contact challenge.syspin@iisc.ac.in

---

Baseline:

We are sharing a baseline for track1. Check the `baselines` folder the codes. The pretrained model is shared through a huggingface repository - `https://huggingface.co/SYSPIN/LIMMITS_track1_baseline`.

The baseline is a multilingual fastspeech model with learnable speaker embeddings. A fork of `DeepForcedAligner` is used to align and extract durations for each speaker (`https://github.com/bloodraven66/DeepForcedAligner`). This repostory is used to extract speech features as well as tokenising the text. The TTS model is trained with 5 hrs of data from each speaker, trained with curriculum learning strategy. 3 hours of shorter wavs (by duration) per speaker are used to train for 100 epochs. After that, 2 hours of random data is added per speaker, and trained further for 400 more epochs.

The utterance level durations from the aligner is shared at `https://www.dropbox.com/sh/yfjwonzrdl5y13q/AACChRwzyqt-7Ae498lg35_9a?dl=0`, with the symbol at `https://www.dropbox.com/scl/fo/wos0ayhhtqs7g5qwxm1tn/h?dl=0&rlkey=mc6yl00h3vus0c3555qbpk13h`.

Speaker specific waveglow vocoders are used, trained with NVIDIA open sourced implementation. We use a fork (`https://github.com/bloodraven66/waveglow_for_LIMMITS23`) with custom inference file `infer.py`

---
Using the baseline:

Step 1: download data of all speakers

Step 2: clone all the repostories mentioned above.

Step 3: Use baselines/train.py to untar all downloaded files. Run it with the `python3 train.py hparams/track1.yaml` with `stop_prep = True` in the config file. Change the save path as required, in the config file.

Step 4: Use `DeepForcedAligner` to preprocess data. Set dataset_dir = data save path from previous step. Change metadata_path to {language}_{gender} combinations, eg: Hindi_M . Repeat it 6 times to preprocess all speakers.

Step 5: To start training the model, change paths for tokens and saved mel spectrograms in `track1.yaml`. If you are using the durations shared above, download it and specify the paths for it as well. The durations are expected to be in `DeepForcedAligner_LIMMITS32_data/{language}_{gender}_data_outputs/durations` and tokens to be in `DeepForcedAligner_LIMMITS32_data/{language}_{gender}_data/tokens`. 

For example: <br>
DeepForcedAligner_LIMMITS32_data/ <br>
├── Hindi_F_data <br>
│   ├── dataset.pkl <br>
│   ├── mels [16512 entries exceeds filelimit, not opening dir] <br>
│   ├── symbols.pkl <br>
│   └── tokens [16512 entries exceeds filelimit, not opening dir] <br>
├── Hindi_F_data_outputs <br>
│   ├── durations [16512 entries exceeds filelimit, not opening dir] <br>
│   └── predictions [16512 entries exceeds filelimit, not opening dir] <br>
├── Hindi_M_data <br>
│   ├── dataset.pkl <br>
│   ├── mels [17796 entries exceeds filelimit, not opening dir] <br>
│   ├── symbols.pkl <br>
│   └── tokens [17796 entries exceeds filelimit, not opening dir] <br>
├── Hindi_M_data_outputs <br>
│   ├── durations [17796 entries exceeds filelimit, not opening dir] <br>
│   └── predictions [17796 entries exceeds filelimit, not opening dir] <br>


Step 6: Start training with `python3 train.py hparams/track1.yaml`. Note that by default the repository uses [wandb](https://wandb.ai/) to log losses and generated samples. 

---

Vocoders:

The vocoder weights are shared through huggingface repositories - 
https://huggingface.co/SYSPIN/Telugu_m_vocoder_waveglow
https://huggingface.co/SYSPIN/Telugu_f_vocoder_waveglow
https://huggingface.co/SYSPIN/Hindi_m_vocoder_waveglow
https://huggingface.co/SYSPIN/Hindi_f_vocoder_waveglow
https://huggingface.co/SYSPIN/Marathi_m_vocoder_waveglow
https://huggingface.co/SYSPIN/Marathi_f_vocoder_waveglow



---

Objective evaluation:

Currently, 2 monolingual ASRs are being used for each of the 2 languages. These models are available on huggingface, evaluation/ASR_eval.py perfoms the evaluation task with all ASRs. Given a synthesised sample, it finds the CER with ASRs of that language, takes the result with the lower score. Finally, all the low score predicted text is used to obtained corpus level CER (instead of averaging the scores). CER is computed using jiwer - https://pypi.org/project/jiwer/.

Check the folders for the expected input/ output formats.

---


