from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
import torch
from utils.logger import logger


class TTS_DATASET(Dataset):
    def __init__(self, mode, args):
        filepath = os.path.join(args.dataset.metadata_path, args.track+'_'+mode+'.txt')
        with open(filepath, 'r') as f:
            ids = f.read().split('\n')[:-1]
        assert len(ids) == len(set(ids))
        
        mel_paths, dur_paths, token_paths = self.get_all_paths(ids, args)
        logger.info(f'found {len(ids)} samples in filelist for {mode} mode, using {len(dur_paths)} for loader')
        self.mel_paths = mel_paths
        self.dur_paths = dur_paths
        self.token_paths = token_paths
        self.spk_mapping = {'te_m':0, 'te_f':1, 'hi_f':2, 'hi_m':3, 'mr_f':4, 'mr_m':5}

    def get_all_paths(self, ids, args):
        spk_level_ids = {}
        mapping = {'te': 'Telugu', 'mr':'Marathi', 'hi':'Hindi'}
        for filename in ids:
            spk = '_'.join([mapping[filename.split('_')[0]] , filename.split('_')[1].upper()])
            if spk not in spk_level_ids: spk_level_ids[spk] = []
            spk_level_ids[spk].append(filename)
        mel_paths, dur_paths, token_paths = [], [], []
        skipped = 0
        for spk in spk_level_ids:
            dur_path = os.path.join(args.dataset.forced_algined_path, spk + args.dataset.duration_path_postfix)
            mel_path = os.path.join(args.dataset.forced_algined_path, spk + args.dataset.mel_path_postfix)
            token_path = args.dataset.token_save_path
            assert os.path.exists(dur_path), f'{dur_path}'
            
            for f in spk_level_ids[spk]:
                utt_mel_path = os.path.join(mel_path, f+'.npy')
                utt_dur_path = os.path.join(dur_path, f+'.npy')
                utt_token_path = os.path.join(token_path, f+'.npy')
                try:
                    assert os.path.exists(utt_mel_path), f'{utt_mel_path}'
                    assert os.path.exists(utt_dur_path), f'{utt_dur_path}'
                    assert os.path.exists(utt_token_path), f'{utt_token_path}'
                    mel_paths.append(utt_mel_path)
                    dur_paths.append(utt_dur_path)
                    token_paths.append(utt_token_path)
                except:
                    skipped += 1
        return mel_paths, dur_paths, token_paths
                
    def __len__(self):
        return len(self.mel_paths)

    def __getitem__(self, i):
        mel = torch.from_numpy(np.load(self.mel_paths[i])).T
        dur = torch.from_numpy(np.load(self.dur_paths[i]))
        token = torch.from_numpy(np.load(self.token_paths[i]))       
        id = Path(self.mel_paths[i]).stem
        spk = self.spk_mapping['_'.join(id.split('_')[:2])]
        return (token, mel, dur, id, spk)


class TTS_Collate():
    def __init__(self, ):
        pass
    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        dur_padded = torch.LongTensor(len(batch), max_input_len)
        dur_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]][2]
            dur_padded[i, :dur.size(0)] = dur

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.IntTensor(len(batch))
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        filenames = []
        speakers = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i] = mel.size(1)
            output_lengths[i] = mel.size(1)
            filenames.append(batch[ids_sorted_decreasing[i]][3])
            speakers.append(batch[ids_sorted_decreasing[i]][4])
        return text_padded, input_lengths, mel_padded, gate_padded, torch.from_numpy(np.array(speakers)), filenames,\
            dur_padded
