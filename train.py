#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import random
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from modules.preprocess import MNLIDatasetReader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange

from modules.log import get_logger


def train(args, device, log):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    LOG = get_logger(create_log_identifier(args))

    model_name = 'xlnet-base-cased'
    LOG.info(f'Setup {model_name} tokenizer...')
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=True)
    LOG.info(f'Setup model {model_name}...')
    xlnet_config = XLNetConfig.from_pretrained(model_name,
                                               output_hidden_states=True,
                                               output_attentions=True,
                                               num_labels=3,
                                               finetuning_task='mnli')

    model = XLNetForSequenceClassification.from_pretrained(model_name, config=xlnet_config)

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        LOG.info(f'Running on {args.n_gpu} GPUS')

    # Load features from datasets
    data_loader = MNLIDatasetReader(args, tokenizer, device, LOG)
    train_dataloader = data_loader.load_train_dataloader()
    val_dataloader = data_loader.load_val_dataloader()

    # Prepare optimizer and schedule (linear warmup and decay)
    LOG.info("Setup Optimizer and Loss Function")
    optimization_steps = len(train_dataloader) * args.epochs
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
        train_epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_epoch_iterator):
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
                evaluation(epoch, step, model, val_dataloader, scheduler, optimization_steps, args, device)

        train_epoch_iterator.close()


def evaluation(epoch, step, model, val_dataloader, scheduler, t_total, args, device):
    epoch_val_loss = 0.0
    executed_steps = 0
    preds = None
    labels = None
    for batch in tqdm(val_dataloader, desc="Evaluation Step for epoch {}".format(epoch)):
        model.eval()
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input = {'input_ids': batch[0],  # word ids
                     'attention_mask': batch[1],  # input mask
                     'token_type_ids': batch[2],  # segment ids
                     'labels': batch[3]}  # labels

            outputs = model(**input)
            val_loss, val_logits = outputs[0:2]
            epoch_val_loss += val_loss.mean().item()
        if preds is None:
            preds = val_logits.detach().cpu().numpy()
            labels = input['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, val_logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, input['labels'].detach().cpu().numpy(), axis=0)
        executed_steps += 1

    epoch_val_loss = epoch_val_loss / executed_steps
    preds = np.argmax(preds, axis=1)
    acc = (preds == labels).mean()
    LOG.info(f'Epoch:{epoch} Step:{step} - Val:[loss = {epoch_val_loss}, acc = {acc}]')
    LOG.info(f'Step:{step} - LR [{scheduler.get_lr()}] - t_total: {t_total} - warmup steps: {args.warmup_steps}')


def log_optimizer_info(step, optimization_steps, scheduler, args, log):
    if step < args.warmup_steps:
        lr_scale = float(step) / float(max(1, args.warmup_steps))
    else:
        lr_scale = max(0.0, float(optimization_steps - step) / float(max(1.0, optimization_steps - args.warmup_steps)))

    optimizer_summary = f'Step:{step} - LR [{scheduler.get_lr()}] - LR scaling[{lr_scale}] - t_total: {optimization_steps} - Warmup: {args.warmup_steps}'
    log.info(optimizer_summary)


def create_log_identifier(args):
    return f'batch{args.batch_size}-max_seq_len{args.max_seq_len}-warmup{args.warmup_steps}-epoch{args.epochs}.log'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--clip_norm', type=float, default=1.0, help="Gradient clipping parameter")
    argparser.add_argument('--epochs', type=int, default=2, help="Train epochs")
    argparser.add_argument('--max_seq_len', type=int, default=128, help="Max Sequence Length")
    argparser.add_argument('--eval_steps', type=int, default=500, help="Steps to execute validation phase")

    argparser.add_argument('--learning_rate', type=float, default=3e-5)
    argparser.add_argument('--adam_epsilon', type=float, default=1e-6)
    argparser.add_argument('--weight_decay', type=float, default=0.0)
    argparser.add_argument('--warmup_steps', type=int, default=4000)
    args = argparser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    tensor_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args, tensor_device)
