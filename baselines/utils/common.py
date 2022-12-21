import yaml
import os
import librosa
import pickle
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from utils.logger import logger
from attrdict import AttrDict
from data_prep import tts_data_handler
from trainers import track1_trainer, track2_trainer
from models import fastspeech

def load_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def load(args):
    loaders = tts_data_handler.loaders(args)
    assert args.track == 'track1'  #only track1 supported as of now
    trainer = get_trainer(args)
    model, args = get_model(args)
    args = AttrDict(args)
    return loaders, trainer, model, args

def get_trainer(args):
    if args.track  == 'track1': 
        trainer =  track1_trainer
    elif args.track  == 'track2': 
        trainer =  track2_trainer
    else:
        raise NotImplementedError
    return trainer

def get_model(args):
    with open(os.path.join(args.dataset.metadata_path, '_'.join((args.track, args.dataset.token_map_name))), 'rb') as f:
        all_symbols = np.load(f)
    fs_args = load_config(args.model.model_config)
    if args.track == 'track1':
        
        model = fastspeech.FastSpeech(n_mel_channels=args.signal.n_mels,
                                    n_symbols=len(all_symbols)+1, **fs_args)
        args = {**args, **fs_args}
    elif args.track == 'track2':
        model = fastspeech.FastSpeech(n_mel_channels=args.signal.n_mels,
                                    n_symbols=len(all_symbols)+1, **fs_args)
        args = {**args, **fs_args}
    else:
        raise NotImplementedError
    return model, args

def get_files(path, extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

def handle_checkpoint(args, epoch):
    if epoch is not None:
        save_every = str(epoch // args.model.save_every)
        path = os.path.join(args.model.chk_path, 
                            '_'.join(
                                    [
                                    args.track,
                                    args.model.name,
                                    args.model.chk_tag,
                                    save_every + '.pth'
                                    ])
                            )
    else:
        path = sorted([f for f in os.listdir(args.model.chk_path) if f.startswith('_'.join(
                                    [
                                    args.track,
                                    args.model.name,
                                    args.model.chk_tag
                                    ]))])[-1]
        path = os.path.join(args.model.chk_path, path)
    return path

def save_checkpoint(save_path, epoch, model, optimizer, bestloss):
    logger.info('Saving checkpoint..')
    logger.info(f'{save_path}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': bestloss,
            }, save_path)

def load_checkpoint(model, load_path):
    try:    
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded checkpoint from {load_path}')
    except:
        logger.info('Checkpoint not found')
        return None
    return model

def get_audio(args, sample, lengths):
    logger.info('generating samples with griffin lim')
    MAX_WAV_VALUE = 32768.0
    if not isinstance(sample, list):
        sample = list(sample)
    audios = []
    for idx in tqdm(range(len(sample))):
        if lengths is not None:
            y_gen_tst = sample[idx][:int(lengths[idx])].T
        y_gen_tst = np.exp(y_gen_tst)
        S = librosa.feature.inverse.mel_to_stft(
                y_gen_tst,
                power=args.signal.power,
                sr=args.signal.sampling_rate,
                n_fft=args.signal.filter_length,
                fmin=args.signal.mel_fmin,
                fmax=args.signal.mel_fmax)
        audio = librosa.core.griffinlim(
                S,
                n_iter=32,
                hop_length=args.signal.hop_length,
                win_length=args.signal.win_length)
        audio = audio * MAX_WAV_VALUE
        audio = audio.astype('int16')
        audios.append(audio)
    return audios