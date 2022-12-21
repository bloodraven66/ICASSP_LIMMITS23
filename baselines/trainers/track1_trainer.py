import os 
import sys
import torch
import numpy as np
from tqdm import tqdm
from utils import common
import matplotlib.pyplot as plt
from utils.logger import logger
from utils.wandb_logger import WandbLogger
from models.fastspeech import FastSpeechLoss

os.environ["WANDB_SILENT"] = "true"

class Train_loop():
    def __init__(self, args, model):
        self.args = args
        if args.wandb_logging.disable:
            logger.info('wandb logging disabled')
            os.environ['WANDB_MODE'] = 'offline'
        self.logger = WandbLogger(args)
        if not args.model.infer:
            self.model = model.to(args.model.device)
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            logger.info(f'{round(pytorch_total_params/1000000, 2)}M params')
            self.criterion = FastSpeechLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=args.model.lr,
                                            betas=(0.9, 0.98),
                                            eps=1e-9,
                                            weight_decay=float(args.model.weight_decay))
        
            self.logger.log({'full_params':pytorch_total_params})
        
        self.bestloss = 100
        self.prev_chk = None

    
    def train(self, loader):
        self.model.train()
        epoch_log = {}
        pbar = tqdm(loader)
        for data in pbar:   
            data = self.to_gpu(data)
            out = self.model(data)
            loss, meta = self.criterion(out, (data[2], data[1], data[3], data[-1]))
            if torch.isnan(loss): logger.info('bad loss'); exit()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            for key in meta:
                if key not in epoch_log: epoch_log[key] = []
                epoch_log[key].append(meta[key].detach().cpu().item())
            pbar.set_description(f"ep:{self.epoch}")
            pbar.set_postfix({'loss':loss.item()})
        self.logger.log({'train_epoch_'+m:sum(epoch_log[m])/len(epoch_log[m]) for m in epoch_log})

    def validate(self, loader):
        self.model.eval()
        epoch_log = {}
        logged_speakers = {}
        log_data, log_out, log_spk, log_lens = [], [], [], []
        for data in loader:
            data = self.to_gpu(data)
            out = self.model(data)
            loss, meta = self.criterion(out, (data[2], data[1], data[3], data[-1]))
            if torch.isnan(loss): logger.info('bad loss'); exit()
            for key in meta:
                if key not in epoch_log: epoch_log[key] = []
                epoch_log[key].append(meta[key].detach().cpu().item())

            if self.epoch % self.args.wandb_logging.share_freq == 0:
                spks = data[-3].detach().cpu().numpy().tolist()

                
                for spk_idx in range(len(spks)):
                    if spks[spk_idx] not in logged_speakers:
                        logged_speakers[spks[spk_idx]] = 0
                    if logged_speakers[spks[spk_idx]] < self.args.wandb_logging.num_samples:
                        log_data.append(data[2][spk_idx].detach().cpu().numpy())
                        log_out.append(out[0][spk_idx].detach().cpu().numpy())
                        log_lens.append(data[3][spk_idx].detach().cpu().numpy())
                        log_spk.append(data[-2][spk_idx])
                        logged_speakers[spks[spk_idx]] += 1
        self.logger.log({'val_epoch_'+m:sum(epoch_log[m])/len(epoch_log[m]) for m in epoch_log})
        self.bestloss = sum(epoch_log['loss'])/len(epoch_log)
        path = common.handle_checkpoint(self.args, self.epoch)
        common.save_checkpoint(save_path=path, epoch=self.epoch, model=self.model, optimizer=self.optimizer, bestloss=self.bestloss)
        if self.epoch % self.args.wandb_logging.share_freq == 0:
            logger.info('adding plots and samples')
            self.logger.log_plots(log_data, log_out, log_spk)
            auds = common.get_audio(self.args, log_out,  log_lens)
            self.logger.log_audio(auds, log_spk)
        
    def infer(self, loader):
        self.model.eval()
        folder = os.path.join(self.args.model.gen_sample_loc, self.args.track)
        if not os.path.exists(folder): os.makedirs(folder)
        for data in loader:
            data = self.to_gpu(data)
            mel_out, mask = self.model(data, infer=True)[:2]
            mask = mask.sum(1).squeeze()
            names = data[-2]
            mel_out = mel_out.detach().cpu().numpy( )
            for idx in range(len(mel_out)):
                path = os.path.join(folder, names[idx]+'.npy')
                data = mel_out[idx][:mask[idx]].T
                with open(path, 'wb') as f:
                    np.save(f, data)
        
    def to_gpu(self, batch):
        return [batch[b].to(self.args.model.device)  if not isinstance(batch[b], list) else batch[b] for b in range(len(batch))]


def main(args, model, loaders):
    train_loop = Train_loop(args, model)
    if args.model.infer:
        logger.info('Eval mode!')
        chk = common.handle_checkpoint(args, epoch=None)
        model = common.load_checkpoint(model, load_path=chk)
        train_loop.model = model.to(args.model.device)
        train_loop.infer(loaders['test'])
        exit()
    if args.model.load_chk:
        train_loop.load_checkpoint()
    for epoch in range(args.model.num_epochs):
        train_loop.epoch = epoch
        if epoch < args.model.loader1_epochs:
            train_loop.train(loaders['train1'])
        else:
            train_loop.train(loaders['train2'])
        train_loop.validate(loaders['val'])
    train_loop.logger.end_run()
