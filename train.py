#!/usr/bin/env python3

import argparse
import random

import numpy as np
import torch
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer

from modules.log import get_logger
from modules.preprocess import MNLIDatasetReader
from modules.train import TrainModel


def get_train_logger(args):
    logger_name = f'batch{args.batch_size}-seq_len{args.max_seq_len}-warmup{args.warmup_steps}-ep{args.epochs}'
    return get_logger(logger_name)


def train(args, device):
    log = get_train_logger(args)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log.info(f'Using device {device}')

    model_name = 'xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=True)
    xlnet_config = XLNetConfig.from_pretrained(model_name,
                                               output_hidden_states=True,
                                               output_attentions=True,
                                               num_labels=3,
                                               finetuning_task='mnli')

    model = XLNetForSequenceClassification.from_pretrained(model_name, config=xlnet_config)

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        log.info(f'Running on {args.n_gpu} GPUS')

    # Load features from datasets
    data_loader = MNLIDatasetReader(args, tokenizer, log)
    train_dataloader = data_loader.load_train_dataloader()
    val_dataloader = data_loader.load_val_dataloader()

    trainer = TrainModel(train_dataloader, val_dataloader, log)
    trainer.train(model, device, args)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--clip_norm', type=float, default=1.0, help="Gradient clipping parameter")
    argparser.add_argument('--epochs', type=int, default=2, help="Train epochs")
    argparser.add_argument('--max_seq_len', type=int, default=128, help="Max Sequence Length")
    argparser.add_argument('--eval_steps', type=int, default=500, help="Steps to execute validation phase")

    argparser.add_argument('--learning_rate', type=float, default=3e-5)
    argparser.add_argument('--adam_epsilon', type=float, default=1e-6)
    argparser.add_argument('--weight_decay', type=float, default=0.0)
    argparser.add_argument('--warmup_steps', type=int, default=4000)

    argparser.add_argument('--device', type=str, help='Device of execution. Values: cpu or cuda', default='cuda')
    argparser.add_argument('--base_path', type=str, default='../MNLI/', help='Base file directory')
    argparser.add_argument('--train_file', type=str, default='train.tsv', help='File that contains train data')
    argparser.add_argument('--val_file', type=str, default='dev_matched.tsv', help='File that contains validation data')

    args = argparser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    tensor_device = torch.device(args.device)
    train(args, tensor_device)
