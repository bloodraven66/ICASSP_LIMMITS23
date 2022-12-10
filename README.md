# ICASSP_LIMMITS23
This is the repository for baselines and evaluation scripts for the ICASSP Signal processing Grand Challenge - LIghtweight, Multi-speaker, Multi-lingual Indic TTS (LIMMITS) 2023.

For any queries, contact challenge.syspin@iisc.ac.in

Objective evaluation:

Currently, 2 monolingual ASRs are being used for each of the 2 languages. These models are available on huggingface, evaluation/ASR_eval.py perfoms the evaluation task with all ASRs. Given a synthesised sample, it finds the CER with ASRs of that language, takes the result with the lower score. Finally, all the low score predicted text is used to obtained corpus level CER (instead of averaging the scores). CER is computed using jiwer - https://pypi.org/project/jiwer/.

Check the folders for the expected input/ output formats.

