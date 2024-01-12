import os
from abc import abstractmethod

import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from numpy import inf
from .metrics_clinical import CheXbertMetrics
import copy
from .optims import LinearWarmupCosineLRScheduler


class BaseTrainer(object):
    def __init__(self, model, criterion_cls, base_probs, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model = model
        self.device = device
        self.is_main_process = is_main_process

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        self.criterion_cls = criterion_cls
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: {}".format(num_parameters))
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = 0.999
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )
        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'val_' + args.monitor_metric

        self.mnt_best = 0 
        self.log_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                # for different shuffling
                self.train_dataloader.sampler.set_epoch(epoch)

            result = self._train_epoch_blip(epoch)
            dist.barrier()
            result = self.eval_blip(result)

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    self.mnt_best = log[self.mnt_metric]
                    self.log_best = copy.deepcopy(log)
                    best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                    torch.save(self.model.module.state_dict(), best_path)
                    print("Saving current best to {}".format(best_path))

            # print logged information 
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

        if self.is_main_process:
            print('Best results w.r.t {}:'.format(self.mnt_metric))
            for key, value in self.log_best.items():
                print('\t{:15s}: {}'.format(str(key), value))

class Trainer(BaseTrainer):
    def __init__(self, model, criterion_cls, base_probs, metric_ftns, args, train_dataloader, val_dataloader, test_dataloader, device, is_main_process):
        super(Trainer, self).__init__(model, criterion_cls, base_probs, metric_ftns, args, device, is_main_process)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer, 
            self.args.epochs, 
            self.args.min_lr, 
            self.args.init_lr, 
            decay_rate=None, 
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )

    def _train_epoch_blip(self, epoch):
        train_loss = 0
        self.model.train()
        for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_lm, loss_cls = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, self.base_probs)
            loss = loss_lm + self.args.cls_weight*loss_cls
            if batch_idx%10 == 0:
                print("{}/{} loss: {}, loss_lm: {}, loss_cls: {}".format(batch_idx, len(self.train_dataloader), loss.item(), loss_lm.item(), self.args.cls_weight*loss_cls.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        return log

    def eval_blip(self, log):
        self.model.module.eval()

        logits = []
        counts = []
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                reports, cls_preds, cls_preds_logits = self.model.module.generate(images, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
                ## logit adjustment
                cls_labels = (cls_labels==1).float()
                logit = cls_preds_logits*cls_labels
                logits.append(logit.cpu().numpy())
                counts.append(cls_labels.cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            #######
            logits = np.concatenate(logits, axis=0)
            counts = np.concatenate(counts, axis=0)
            logits = np.sum(logits, 0)
            counts = np.sum(counts, 0)
            logits = logits / counts
            logits /= np.max(logits)
            logits = np.append(logits, [1,1,1,1]) # 4 auxiliary diseases
            #######
            self.base_probs = logits # update class distribution
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            val_ce = self.chexbert_metrics.compute(val_gts, val_res)
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update(**{'val_' + k: v for k, v in val_ce.items()})

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.numpy().tolist()
                clip_memory = clip_memory.to(self.device) 
                ground_truths = captions
                reports, _, _ = self.model.module.generate(images, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)

                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
        return log

    
