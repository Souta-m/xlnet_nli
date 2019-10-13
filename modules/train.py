#!/usr/bin/env python3

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tqdm import trange, tqdm
import torch

from modules.log import get_logger


class TrainModel:

    def __init__(self, train_dataloader, val_dataloader, logger):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self._logger = logger


    def train(self, model, device, args):
        log = self._logger
        # Prepare optimizer and schedule (linear warmup and decay)
        log.info("Setup Optimizer and Loss Function")
        optimization_steps = len(self.train_dataloader) * args.epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=optimization_steps)

        # Train
        log.info("Training Started!")
        model.zero_grad()
        for epoch in trange(args.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch = tuple(t.to(device) for t in batch)  # Send data to target device
                model.train()
                model_input = {'input_ids': batch[0],  # word ids
                               'attention_mask': batch[1],  # input mask
                               'token_type_ids': batch[2],  # segment ids
                               'labels': batch[3]}  # labels

                outputs = model(**model_input)
                train_loss = outputs[0]
                if args.n_gpu > 1:
                    train_loss = train_loss.mean()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # grad clip
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                if (step + 1) % args.eval_steps == 0:
                    self.evaluation(epoch, step, model, optimization_steps, model, device, scheduler, args)


    def evaluation(self, train_epoch, train_step, optimization_steps, model, device, scheduler, args):
        epoch_val_loss = 0.0
        executed_steps = 0
        all_predictions = torch.tensor([])
        all_labels = torch.tensor([])
        all_loss = torch.tensor([])
        for batch in tqdm(self.val_dataloader, desc="Evaluation"):
            model.eval()
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                model_input = {'input_ids': batch[0],  # word ids
                         'attention_mask': batch[1],  # input mask
                         'token_type_ids': batch[2],  # segment ids
                         'labels': batch[3]}  # labels

                outputs = model(**model_input)
                val_loss, val_logits = outputs[0:2]
                predictions = torch.argmax(val_logits, dim=1)
                all_predictions = torch.cat([all_predictions, predictions])
                all_labels = torch.cat([all_labels, model_input['labels']])
                all_losses = torch.cat([all_losses, val_loss])
            executed_steps += 1

        epoch_val_loss = all_losses.mean() / executed_steps
        acc = torch.eq(all_predictions, all_labels).sum().item() / all_predictions.shape[0]
        self._log.info(f'Epoch:{train_epoch} Step:{train_step} - Val:[loss = {epoch_val_loss}, acc = {acc}]')
        self._log_optimizer_info(train_step, optimization_steps, scheduler, args)

    def _log_optimizer_info(self, step, t_total, scheduler, args):
        if step < args.warmup_steps:
            lr_scale = float(step) / float(max(1, args.warmup_steps))
        else:
            lr_scale = max(0.0,
                           float(t_total - step) / float(max(1.0, t_total - args.warmup_steps)))

        optimizer_summary = f'Step:{step} - LR [{scheduler.get_lr()}] - LR scaling[{lr_scale}] ' \
                            f'- t_total: {t_total} - Warmup: {args.warmup_steps}'
        self._log.info(optimizer_summary)

