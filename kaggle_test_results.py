#!/usr/bin/env python3

import argparse
import random
import os

import numpy as np
import torch
import pandas as pd
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from modules.preprocess import KaggleMNLIDatasetReader

from modules.log import get_logger
from modules.test import Test


def validate_on_test_set(args, device):
    log = get_logger(f"test-results")
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    log.info(f'Using device {device}')

    model_name = 'xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=True)
    xlnet_config = XLNetConfig.from_pretrained(args.config_file)
    data_reader = KaggleMNLIDatasetReader(args, tokenizer, log)
    model = XLNetForSequenceClassification.from_pretrained(args.model_file, config=xlnet_config)

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    log.info(f'Running on {args.n_gpu} GPUS')

    test_executor = Test(tokenizer, log, data_reader)
    write_kaggle_results("matched", args.test_matched_file, test_executor, device, model)
    write_kaggle_results("mismatched", args.test_mismatched_file, test_executor, device, model)


def write_kaggle_results(dataset_type, file, test_executor, device, model):
    test_file = os.path.join(args.base_path, file)
    ids, preds = test_executor.validate_on_test_set(args, device, test_file, model)
    df = pd.DataFrame()
    df['pairID'] = ids
    df['gold_label'] = preds
    df.to_csv(f'{args.result_dir}/{dataset_type}.tsv', header=True, index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--n_gpu", type=int)
    argparser.add_argument("--max_seq_len", type=int, default=128)
    argparser.add_argument("--batch_size", type=int, default=8)
    argparser.add_argument('--device', type=str, help='Device of execution. Values: cpu or cuda', default='cuda')
    argparser.add_argument('--base_path', type=str, default='../MNLI/', help='Base file directory')
    argparser.add_argument('--test_matched_file', type=str, default='multinli_0.9_test_matched_unlabeled.txt',
                           help='File that contains test '
                                'matched data')
    argparser.add_argument('--test_mismatched_file', type=str, default='multinli_0.9_test_mismatched_unlabeled.txt',
                           help='File that contains test mismatched data')
    argparser.add_argument('--result_dir', type=str, default='nli_test_results/',
                           help='Directory that contains test results')
    argparser.add_argument('--model_file', type=str, help="Model file", required=True)
    argparser.add_argument('--config_file', type=str, help="Model config file", required=True)
    args = argparser.parse_args()

    args.n_gpu = torch.cuda.device_count() if not args.n_gpu else args.n_gpu

    tensor_device = torch.device(args.device)
    validate_on_test_set(args, tensor_device)
