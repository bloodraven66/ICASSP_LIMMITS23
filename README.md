# ICASSP_LIMMITS23
This is the repository for baselines and evaluation scripts for the ICASSP Signal processing Grand Challenge - LIghtweight, Multi-speaker, Multi-lingual Indic TTS (LIMMITS) 2023.

For any queries, contact challenge.syspin@iisc.ac.in

---

Baseline:

We are sharing a baseline for track1. Check the `baselines` folder the codes. The pretrained model is shared through a huggingface repository - `https://huggingface.co/SYSPIN/LIMMITS_track1_baseline`.

The baseline is a multilingual fastspeech model with learnable speaker embeddings. A fork of `DeepForcedAligner` is used to align and extract durations for each speaker (`https://github.com/bloodraven66/DeepForcedAligner`). The TTS model is trained with 5 hrs of data from each speaker, trained with curriculum learning strategy. 3 hours of shorter wavs (by duration) per speaker are used to train for 100 epochs. After that, 2 hours of random data is added per speaker, and trained further for 400 more epochs.

The utterance level durations from the aligner is shared at `https://www.dropbox.com/sh/yfjwonzrdl5y13q/AACChRwzyqt-7Ae498lg35_9a?dl=0`, with the symbol at `https://www.dropbox.com/scl/fo/wos0ayhhtqs7g5qwxm1tn/h?dl=0&rlkey=mc6yl00h3vus0c3555qbpk13h`.

Speaker specific waveglow vocoders are used, trained with NVIDIA open sourced implementation. We use a fork (`https://github.com/bloodraven66/waveglow_for_LIMMITS23`) with custom inference file `infer.py`

---

Vocoders:

The vocoder weights are shared through huggingface repositories - <br>
https://huggingface.co/SYSPIN/Telugu_m_vocoder_waveglow <br>
https://huggingface.co/SYSPIN/Telugu_f_vocoder_waveglow <br>
https://huggingface.co/SYSPIN/Hindi_m_vocoder_waveglow <br>
https://huggingface.co/SYSPIN/Hindi_f_vocoder_waveglow <br>
https://huggingface.co/SYSPIN/Marathi_m_vocoder_waveglow <br>
https://huggingface.co/SYSPIN/Marathi_f_vocoder_waveglow <br>



---

Objective evaluation:

Currently, 2 monolingual ASRs are being used for each of the 2 languages. These models are available on huggingface, evaluation/ASR_eval.py perfoms the evaluation task with all ASRs. Given a synthesised sample, it finds the CER with ASRs of that language, takes the result with the lower score. Finally, all the low score predicted text is used to obtained corpus level CER (instead of averaging the scores). CER is computed using jiwer - https://pypi.org/project/jiwer/.

Check the folders for the expected input/ output formats.

---


