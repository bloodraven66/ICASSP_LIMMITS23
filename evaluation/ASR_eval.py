from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os
import numpy as np
from pathlib import Path
import argparse
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F
import jiwer
from pathlib import Path
import soundfile as sf
from espnet2.bin.asr_inference import Speech2Text
import torchaudio
import jiwer

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-f', '--folder', required=True)
parser.add_argument('-s', '--savepath', default='./demo_result')
parser.add_argument('-t', '--text', default='demo_files/text')

HINDI_VAKYANSH_ASR = 'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200'
HINDI_AI4B_ASR = 'ai4bharat/indicwav2vec-hindi'
TELUGU_CSTD_ASR = 'viks66/CSTD_Telugu_ASR'
TELUGU_VAKYANSH_ASR = 'Harveenchadha/vakyansh-wav2vec2-telugu-tem-100'
MARATHI_XLSR_SLR64_ASR = 'tanmaylaud/wav2vec2-large-xlsr-hindi-marathi'
MARATHI_XLSR_SLR64_ASR2 = 'sumedh/wav2vec2-large-xlsr-marathi' 

# MARATHI_SLR64_ASR = 'espnet/marathi_openslr64'
#link = https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt

def get_Wav2Vec2ForCTC(MODEL_ID):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return model, processor

def get_espnet_asr(key):
    model = Speech2Text.from_pretrained(key)
    return model, None

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech =  resampler(speech_array).squeeze().numpy()
    return speech

def get_transcript_espnet(wav_file, model):
    speech_array = speech_file_to_array_fn(wav_file)
    nbests = model(speech_array)
    preds, *_ = nbests[0]
    return preds

def get_transcript(wav_file, model, processor, argmax=True):
    if processor is None: transcription = get_transcript_espnet(wav_file, model); return transcription
    audio_input, sample_rate = sf.read(wav_file)
    resampled_audio = F.resample(torch.tensor(audio_input), sample_rate, 16000).numpy()
    input_values = processor(resampled_audio, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits.detach()
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].replace('<s>', '')
    return transcription

def hindi_proc(files):
    for idx, modelname in enumerate([HINDI_VAKYANSH_ASR, HINDI_AI4B_ASR]):

        model, processor = get_Wav2Vec2ForCTC(modelname)
        for filename in files:  
            transcript = get_transcript(filename, model, processor)
            save_transcript(filename, transcript, key=f'model{idx}')

def telugu_proc(files):

    model, processor = get_espnet_asr(TELUGU_CSTD_ASR)
    for filename in files: 
        transcript = get_transcript(filename, model, processor)
        save_transcript(filename, transcript, key='model1')

    model, processor = get_Wav2Vec2ForCTC(TELUGU_VAKYANSH_ASR)
    for filename in files:
        transcript = get_transcript(filename, model, processor)
        save_transcript(filename, transcript, key='model2') 

def marathi_proc(files):

    # model, processor = get_espnet_asr(MARATHI_SLR64_ASR)
    # for filename in files: 
    #     transcript = get_transcript(filename, model, processor)
    #     save_transcript(filename, transcript, key='model1')
#
    model, processor = get_Wav2Vec2ForCTC(MARATHI_XLSR_SLR64_ASR2)
    for filename in files:
        transcript = get_transcript(filename, model, processor)
        save_transcript(filename, transcript, key='model1') 
    
    model, processor = get_Wav2Vec2ForCTC(MARATHI_XLSR_SLR64_ASR)
    for filename in files:
        transcript = get_transcript(filename, model, processor)
        save_transcript(filename, transcript, key='model2') 


def save_transcript(filename, transcript, key):
    savepath = os.path.join(args.savepath, filename.split('/')[-2])
    if not os.path.exists(savepath): os.mkdir(savepath)
    savepath = os.path.join(savepath, Path(filename).stem+'_'+key+'.txt')
    with open(savepath, 'w') as f:
        f.write(transcript)

def read_files():
    langwise = {}
    for filename in os.listdir(args.folder):

        input_ln = filename.split('_')[1]
        if input_ln not in langwise: langwise[input_ln] = []
        langwise[input_ln].append(os.path.join(args.folder, filename))
    return langwise

def get_text():
    with open(args.text, 'r') as f:
        text = f.read().split('\n')[:-1]
    text = {t.split('\t')[0]:t.split('\t')[1] for t in text}
    return text

def evaluate():
    eval_folder = os.path.join(args.savepath, Path(args.folder).stem)
    eval_files = os.listdir(eval_folder)
    text = get_text()
    sent2res = {}
    for f in eval_files:
        sent = '_'.join(f.split('_')[1:3])
        if sent not in sent2res: sent2res[sent] = []
        sent2res[sent].append(f)

    gnds, hyps = [], []
    for sent in sent2res:
        scores, texts = [], []
        for fname in sent2res[sent]:
            with open(os.path.join(eval_folder, fname), 'r') as f:
                res = f.read()

            texts.append(res)
            cer = jiwer.cer(text[sent], res)
            scores.append(cer)
        
        best_score_idx = np.argmin(scores)
        gnds.append(text[sent])
        hyps.append(texts[best_score_idx])
    final_cer  = jiwer.cer(gnds, hyps)
    print(f'number of eval files:{len(gnds)}, corpus CER:{final_cer}')

def main():
    langwise = read_files()
    hindi_proc(langwise['hi'])
    telugu_proc(langwise['te'])
    marathi_proc(langwise['mr'])
    evaluate()

if __name__ == '__main__':
    args = parser.parse_args()
    main()

