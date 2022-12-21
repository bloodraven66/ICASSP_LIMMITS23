import os
import random
import shutil
import numpy as np
import librosa
from . import tts_loader
from pathlib import Path
from torch.utils.data import DataLoader
from utils.logger import logger
from tqdm import tqdm
from multiprocessing import Pool
from utils import common

def loaders(args):
    handle_untar(args)
    get_dur(args)
    loader_keys = make_filelist(args)

    tokenize(args, 'train2' if args.track == 'track1' else 'train')
    if args.dataset.stop_prep: logger.info('Stopping..'); exit()
    loaders_ = {}
    for mode in loader_keys:
        dataset = tts_loader.TTS_DATASET(mode, args)
        collate_fn = tts_loader.TTS_Collate()
        shuffle=True if mode.startswith('train') else False
        loaders_[mode] = DataLoader(dataset, num_workers=1, shuffle=shuffle,
                          batch_size=args.model.batch_size, pin_memory=False,
                          drop_last=True, collate_fn=collate_fn)
    return loaders_

def tokenize(args, key):
    save_path = os.path.join(args.dataset.metadata_path, args.track+'_'+args.dataset.token_map_name)
    if os.path.exists(save_path) and not args.dataset.force_tokenize: return


    txt_files = common.get_files(args.dataset.untar_path, '.txt')
    txt_files = [str(t) for t in txt_files if 'metadata' not in str(t)]
    symbol_dict = {}
    if not os.path.exists(args.dataset.token_save_path):
        os.mkdir(args.dataset.token_save_path)

    logger.info('tokenising..')
    for file in tqdm(txt_files):
        token_save_path = os.path.join(args.dataset.token_save_path, Path(file).stem+'.npy')
        with open(file, 'r') as f:
            text = f.read()
        tokens = []
        for ch in text:
            if ch not in symbol_dict:
                symbol_dict[ch] = len(symbol_dict) + 1
            tokens.append(symbol_dict[ch])
        with open(token_save_path, 'wb') as f:
            np.save(f, tokens)

    with open(save_path, 'wb') as f:
        np.save(f, list(symbol_dict.keys()))
    

def make_filelist(args, num_test_per_spk=20, num_val_per_spk=70):
    assert args.track in ['track1', 'track2', 'track3']
    if args.track == 'track2':
        splits = {'train':[], 'val':[], 'test':[]}
        
        if not args.dataset.force_genfilelist:
            done = True
            for split_name in splits:
                save_path = os.path.join(args.dataset.metadata_path, args.track +'_'+ split_name+'.txt')
                
                if not os.path.exists(save_path):
                    done = False
            if done: return splits.keys()
        for filename in os.listdir(args.dataset.metadata_path):
            if not filename.startswith('dur'): continue
            with open(os.path.join(args.dataset.metadata_path, filename), 'r') as f:
                data = f.read().split('\n')[:-1]
            data = [Path(d.split('\t')[0]).stem for d in data]
            splits['test'].extend(data[:num_test_per_spk])
            splits['val'].extend(data[num_test_per_spk:num_test_per_spk+num_val_per_spk])
            splits['train'].extend(data[num_test_per_spk+num_val_per_spk:])
        
        for split_name in splits:
            save_path = os.path.join(args.dataset.metadata_path, args.track +'_'+ split_name+'.txt')
            with open(save_path, 'w') as f:
                for line in splits[split_name]:
                    f.write(line+'\n')
        return splits.keys()

    splits = {'train1':[], 'train2':[], 'val':[], 'test':[]}
    if not args.dataset.force_genfilelist:
        done = True
        for split_name in splits:
            save_path = os.path.join(args.dataset.metadata_path, args.track +'_'+ split_name+'.txt')
            if not os.path.exists(save_path):
                done = False
        if done: return splits.keys()

    for filename in os.listdir(args.dataset.metadata_path):
        if not filename.startswith('dur'): continue
        with open(os.path.join(args.dataset.metadata_path, filename), 'r') as f:
            data = f.read().split('\n')[:-1]
        data = {Path(d.split('\t')[0]).stem:float(d.split('\t')[-1]) for d in data}
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))
        included_keys, total_dur = set(), 0
        for (key, dur) in sorted_data.items():
            if total_dur > args.dataset.first_dataloader_dur * 60 * 60:
                break
            included_keys.add(key)
            total_dur += dur
        splits['train1'].extend(list(included_keys))
        splits['train2'].extend(list(included_keys))
        remaining_keys = list(sorted_data.keys())[len(included_keys):]
        train_keys = select_loop(remaining_keys, included_keys, sorted_data, (args.dataset.limit_total_dur-args.dataset.first_dataloader_dur) * 60 * 60)
        splits['train2'].extend(list(train_keys))
        included_keys.update(train_keys)
        dev_keys = select_loop(remaining_keys, included_keys, sorted_data, args.dataset.dev_dur * 60 * 60)
        splits['val'].extend(list(dev_keys))
        included_keys.update(dev_keys)
        test_keys = select_loop(remaining_keys, included_keys, sorted_data, args.dataset.test_dur * 60 * 60)
        splits['test'].extend(list(test_keys))

    for split_name in splits:
        save_path = os.path.join(args.dataset.metadata_path, args.track +'_'+ split_name+'.txt')
        assert len(splits[split_name]) == len(set(splits[split_name]))
        with open(save_path, 'w') as f:
            for line in splits[split_name]:
                f.write(line+'\n')
    return splits.keys()

def select_loop(remaining_keys, included_keys, sorted_data, limit_dur):
    total_dur = 0
    keys = set()
    while(True):
        random_key = random.choice(remaining_keys)
        if random_key in included_keys: continue
        if sorted_data[random_key] + total_dur > limit_dur:
            break
        total_dur += sorted_data[random_key]
        keys.add(random_key)
    return keys

def handle_untar(args):
    
    req_folders = ['_'.join([l, s, str(i)]) for l in args.dataset.langs for s in args.dataset.speakers for i in range(1, 6)]
    existing_folders = os.listdir(args.dataset.untar_path)
    if 'metadata' in existing_folders: existing_folders.remove('metadata')
    if 'tokens' in existing_folders: existing_folders.remove('tokens')
    if set(req_folders) == set(existing_folders):
        logger.info('All files found')
        if not args.dataset.force_untar: return
    logger.info('starting untar..')
    files = os.walk(args.dataset.tar_path)
    for (root, dirs, files) in os.walk(args.dataset.tar_path):
        for filename in tqdm(files):
            filename = os.path.join(root, filename)
            shutil.unpack_archive(filename, args.dataset.untar_path)
    return

def librosa_dur(filename):
    try:
        y, sr = librosa.load(filename)
        return len(y)/sr
    except:
        return None

def get_dur(args):
    save_folder = os.path.join(args.dataset.metadata_path)
    if not os.path.exists(save_folder): os.mkdir(save_folder)
    if len([f for f in os.listdir(save_folder) if f[:3] == 'dur']) == 6 and not args.dataset.force_getdur: return
    files = common.get_files(args.dataset.untar_path)
    files_per_spk = {}
    for filename in files:
        spk = '_'.join(filename.stem.split('_')[:2])
        if spk not in files_per_spk: files_per_spk[spk] = []
        files_per_spk[spk].append(filename)
    

    for key in files_per_spk:
        save_path = os.path.join(save_folder, 'dur_'+key)
        files = files_per_spk[key]
        with Pool(args.dataset.num_proc) as p:
            res = list(tqdm(p.imap(librosa_dur, files), total=len(files)))
        count = 0
        with open(save_path, 'w') as f:
            for i in range(len(res)):
                if res[i] is not None:
                    count += 1
                    f.write(str(files[i])+'\t'+str(res[i])+'\n')
        if count != len(res): logger.info(f'{len(res)-count} files skipped')
    return
        