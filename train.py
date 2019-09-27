#!/usr/bin/env python3

import torch
import numpy as np
import argparse
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from modules.preprocess import MNLIDatasetReader
from torch.utils.data import DataLoader, RandomSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange

from modules.log import get_logger

LOG = get_logger('train_xlnet')


def train_configs(max_seq_len, tokenizer, device):
    base_path = '/home/ichida/dev_env/ml/data/multinli_1.0'
    return {
        'train_file': '{}/multinli_1.0_train_reduced.txt'.format(base_path),
        'val_file': '{}/multinli_1.0_dev_matched_reduced.txt'.format(base_path),
        'max_seq_len': max_seq_len,
        'tokenizer': tokenizer,
        'device': device
    }


def get_train_dataset_loader(tokenizer, device, batch_size):
    paths = train_configs(tokenizer, device)
    reader = MNLIDatasetReader(**paths)
    train_dataset = reader.load_val_dataset()
    return DataLoader(train_dataset, batch_size, RandomSampler(train_dataset))


def get_val_dataset_loader(tokenizer, device, batch_size):
    paths = train_configs(tokenizer, device)
    reader = MNLIDatasetReader(**paths)
    val_dataset = reader.load_val_dataset()
    return DataLoader(val_dataset, batch_size, RandomSampler(val_dataset))


def train(args, device):
    model_name = 'xlnet-base-cased'
    LOG.info(f'Setup {model_name} tokenizer...')
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    LOG.info(f'Setup model {model_name}...')
    model_name = 'xlnet-base-cased'
    xlnet_config = XLNetConfig.from_pretrained(model_name,
                                               output_hidden_states=True,
                                               output_attentions=True,
                                               num_labels=3,
                                               finetuning_task='NLI')

    model = XLNetForSequenceClassification.from_pretrained(model_name, config=xlnet_config)
    model.to(device)
    train_dataloader = get_train_dataset_loader(tokenizer, device, args.batch_size)

    LOG.info("Setup Optimizer and Loss Function")
    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = len(train_dataloader) // args.train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_name.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model_name.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    LOG.info("Training Started!")
    for epoch in trange(args.train_epochs, desc="Epoch"):
        train_epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_epoch_iterator):
            model_input = {'input_ids': batch[0],  # word ids
                           'attention_mask': batch[1],  # input mask
                           'token_type_ids': batch[2],  # segment ids
                           'labels': batch[3]}  # labels

            outputs = model(**model_input)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # grad clip
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        evaluation(epoch=epoch, model=model, tokenizer=tokenizer, batch_size=args.batch_size, device=device)


def evaluation(epoch, model, tokenizer, batch_size, device):
    val_dataloader = get_val_dataset_loader(tokenizer, device, batch_size)
    epoch_val_loss = 0.0
    executed_steps = 0
    for batch in tqdm(val_dataloader, desc="Evaluation Step for epoch {}".format(epoch)):
        tensor_batch = tuple(tensor.to(device) for tensor in batch)
        with torch.no_grad():
            model_input = {'input_ids': tensor_batch[0],  # word ids
                           'attention_mask': tensor_batch[1],  # input mask
                           'token_type_ids': tensor_batch[2],  # segment ids
                           'labels': tensor_batch[3]}  # labels

            outputs = model(**model_input)
            val_loss, val_logits = outputs[0:2]
            epoch_val_loss += val_loss.mean().item()
            if preds is None:
                preds = val_logits.detach().cpu().numpy()
                labels = model_input['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, val_logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, model_input['labels'].detach().cpu().numpy(), axis=0)
        executed_steps += 1

    epoch_val_loss = epoch_val_loss / executed_steps
    preds = np.argmax(preds, axis=1)
    acc = (preds == labels).mean()
    LOG.info(f'Epoch {epoch} - Val:[loss = {epoch_val_loss}, acc = {acc}]')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=16)

    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--clip_norm', type=float, default=1.0, help="Gradient clipping parameter")
    args.add_argument('--epochs', type=int, default=8, help="Train epochs")
    args.add_argument('--max_seq_len', type=int, default=128, help="Max Sequence Length")

    args.add_argument('--learning_rate', type=float, default=5e-5)
    args.add_argument('--adam_epsilon', type=float, default=1e-8)
    args.add_argument('--weight_decay', type=float, default=0.0)
    args.add_argument('--warmup_steps', type=int, default=0)

    tensor_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args, tensor_device)
