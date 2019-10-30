#!/usr/bin/env python3

import argparse
import random
import os

import numpy as np
import torch
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer

from modules.log import get_logger
from modules.preprocess import MNLIDatasetReader
from modules.train import TrainModel


def get_train_logger(args):
    logger_name = f'{args.model_name}-batch{args.batch_size}-seq{args.max_seq_len}' \
        f'-warmup{args.warmup_steps}-ep{args.epochs}-{args.dataset_name}'
    return get_logger(logger_name)


def train(args, device):
    args.dataset_name = "MNLI"  # TODO: parametrize

    model_name = args.model_name
    log = get_train_logger(args)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log.info(f'Using device {device}')
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=True)
    xlnet_config = XLNetConfig.from_pretrained(model_name,
                                               output_hidden_states=True,
                                               output_attentions=True,
                                               num_labels=3,
                                               finetuning_task=args.dataset_name)

    model = XLNetForSequenceClassification.from_pretrained(model_name, config=xlnet_config)

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        log.info(f'Running on {args.n_gpu} GPUS')

    # Load features from datasets
    data_loader = MNLIDatasetReader(args, tokenizer, log)
    train_file = os.path.join(args.base_path, args.train_file)
    val_file = os.path.join(args.base_path, args.val_file)
    train_dataloader = data_loader.load_train_dataloader(train_file)
    val_dataloader = data_loader.load_val_dataloader(val_file)

    trainer = TrainModel(train_dataloader, val_dataloader, log)
    trainer.train(model, device, args)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', type=str, default='xlnet-large-cased')
    argparser.add_argument("--n_gpu", type=int)

    argparser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--clip_norm', type=float, default=1.0, help="Gradient clipping parameter")
    argparser.add_argument('--epochs', type=int, default=4, help="Train epochs")
    argparser.add_argument('--max_seq_len', type=int, default=128, help="Max Sequence Length")
    argparser.add_argument('--eval_steps', type=int, default=1000, help="Steps to execute validation phase")
    argparser.add_argument('--min_acc_save', type=float, default=0.86, help='Min acc to save the trained model')
    argparser.add_argument('--max_loss_save', type=float, default=0.40,
                           help='Maximum error value considered to save a model')

    argparser.add_argument('--learning_rate', type=float, default=3e-5)
    argparser.add_argument('--adam_epsilon', type=float, default=1e-6)
    argparser.add_argument('--weight_decay', type=float, default=0.0)
    argparser.add_argument('--warmup_steps', type=int, default=3500)

    argparser.add_argument('--device', type=str, help='Device of execution. Values: cpu or cuda', default='cuda')
    argparser.add_argument('--base_path', type=str, default='../MNLI/', help='Base file directory')
    argparser.add_argument('--train_file', type=str, default='train.tsv', help='File that contains train data')
    argparser.add_argument('--val_file', type=str, default='dev_matched.tsv', help='File that contains validation data')
    argparser.add_argument('--output_dir', type=str, default='saved_models/',
                           help='Directory of resulted pretrained model.')

    args = argparser.parse_args()

    args.n_gpu = torch.cuda.device_count() if not args.n_gpu else args.n_gpu

    tensor_device = torch.device(args.device)
    train(args, tensor_device)
