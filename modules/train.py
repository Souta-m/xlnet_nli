#!/usr/bin/env python3

import os
import pathlib

import torch
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tqdm import trange, tqdm

from modules.evaluator import Evaluator


class TrainModel:

    def __init__(self, train_dataloader, val_dataloader, logger):
        self.train_dataloader = train_dataloader
        self.evaluator = Evaluator(val_dataloader, logger)
        self._logger = logger

    def __call__(self, model, device, args):
        log = self._logger
        # Prepare optimizer and schedule (linear warmup and decay)
        optimization_steps = (len(self.train_dataloader) * args.epochs) // args.gradient_accumulation_steps
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=optimization_steps)

        # Train
        log.info(f"Training Started with parameters {args}")
        model.zero_grad()
        global_step = 1
        for epoch in trange(args.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                model.train()
                batch = tuple(t.to(device) for t in batch)  # Send data to target device
                model_input = {'input_ids': batch[0],  # word ids
                               'attention_mask': batch[1],  # input mask
                               'token_type_ids': batch[2],  # segment ids
                               'labels': batch[3]}  # labels
                outputs = model(**model_input)
                train_loss = outputs[0]
                if args.gradient_accumulation_steps > 1:
                    train_loss = train_loss / args.gradient_accumulation_steps
                train_loss.backward()
                # Accumulates the gradient before optimize the model
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # grad clip
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                # Steps necessary to run the trained model into validation data set
                if (step + 1) % args.eval_steps == 0 and not args.eval_per_epoch:
                    self.evaluate_on_val_set(epoch, global_step, optimization_steps, model, device, scheduler, args)
                global_step += 1

            if args.eval_per_epoch:
                self.evaluate_on_val_set(epoch, global_step, optimization_steps, model, device, scheduler, args)

    def evaluate_on_val_set(self, train_epoch, train_step, optimization_steps, model, device, scheduler, args):
        all_predictions, all_labels, val_loss = self.evaluator(model, device, "Validation")
        val_acc = torch.eq(all_predictions, all_labels).sum().item() / all_predictions.shape[0]
        self._logger.info(
            f'Epoch:{train_epoch} Step:{train_step} - Val:[loss = {val_loss:0.4f}, acc = {val_acc:0.4f}]')
        self._log_optimizer_info(train_step, optimization_steps, scheduler, args)
        if args.min_acc_save < val_acc or val_loss < args.max_loss_save:
            self.checkpoint(model, val_acc, val_loss, train_step, train_epoch, args)

    def _log_optimizer_info(self, step, t_total, scheduler, args):
        if step < args.warmup_steps:
            lr_scale = float(step) / float(max(1, args.warmup_steps))
        else:
            lr_scale = max(0.0, float(t_total - step) / float(max(1.0, t_total - args.warmup_steps)))
        optimizer_summary = f'Step:{step} - LR [{scheduler.get_lr()}] - LR scaling[{lr_scale:0.3f}] ' \
            f'- t_total: {t_total} - Warmup: {args.warmup_steps}'
        self._logger.info(optimizer_summary)

    def checkpoint(self, model, accuracy, loss, step, epoch, args):
        file = f'{args.model_name}-acc{accuracy:0.3f}-loss{loss:0.3f}-step{step}-epoch{epoch}/'
        path = os.path.join(args.output_model_dir, file)
        if not os.path.exists(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        self._logger.info(f'Saving model with acc: {accuracy:0.3f} and loss: {loss:0.3f} into file {path}')
        model.save_pretrained(path)
