#!/usr/bin/env python3
import argparse


def global_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--clip_norm', type=float, default=1.0, help="Gradient clipping parameter")
    parser.add_argument('--epochs', type=int, default=8, help="Train epochs")
    parser.add_argument('--max_seq_len', type=int, default=512, help="Max Sequence Length")
    parser.add_argument('--eval_per_epoch', action="store_true", default=True, help="Validates per epoch.")
    parser.add_argument('--min_acc_save', type=float, default=1.1, help='Min acc to save the trained model')
    parser.add_argument('--max_loss_save', type=float, default=-1, help='Max error value to save a model')
    parser.add_argument('--eval_steps', type=int, default=400, help="Steps to execute validation phase")
    parser.add_argument('--learning_rate', type=float, default=9e-6)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--output_model_dir', type=str, default='saved_models')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=30)
    parser.add_argument('--device', type=str, help='Device of execution. Values: cpu or cuda', default='cuda')
    parser.add_argument('--base_path', type=str, default='../NORMS/', help='Base file directory')
    parser.add_argument('--train_file', type=str, default='train_all.tsv', help='File that contains train data')
    parser.add_argument('--val_file', type=str, default='val_all.tsv', help='File that contains validation data')
    parser.add_argument('--test_file', type=str, default='test.tsv', help='File that contains test data')

    return parser
