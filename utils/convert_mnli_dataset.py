#!/usr/bin/env python3


import pandas as pd


def remove_unused_columns():
    base_path = '/home/ichida/dev_env/ml/data/multinli_1.0'
    files = {
        'train_file': '{}/multinli_1.0_train.jsonl'.format(base_path),
        'reduced_train_file': '{}/multinli_1.0_train_reduced.txt'.format(base_path),
        'val_file': '{}/multinli_1.0_dev_matched.jsonl'.format(base_path),
        'reduced_val_file': '{}/multinli_1.0_dev_matched_reduced2.txt'.format(base_path)
    }

    train_df = pd.read_json(files['train_file'], lines=True)
    train_df = train_df[['sentence1', 'sentence2', 'gold_label', 'genre']]
    train_df.to_csv(files['reduced_train_file'], index=False, sep='\t')

    val_df = pd.read_json(files['val_file'], lines=True)
    val_df = val_df[['sentence1', 'sentence2', 'gold_label', 'genre']]
    print(len(val_df))
    val_df.to_csv(files['reduced_val_file'], index=False, sep='\t')


def create_reduced_samples(n):
    base_path = '/home/ichida/dev_env/ml/data/multinli_1.0'
    files = {
        'train_file': '{}/multinli_1.0_train.jsonl'.format(base_path),
        'reduced_train_file': '{}/multinli_1.0_train_reduced.txt'.format(base_path),
        'val_file': '{}/multinli_1.0_dev_matched.jsonl'.format(base_path),
        'reduced_val_file': '{}/multinli_1.0_dev_matched_reduced.txt'.format(base_path)
    }

    train_df = pd.read_csv(files['reduced_train_file'], sep='\t')
    reduced_df = train_df.sample(n)
    reduced_df.to_csv(f'{base_path}/multinli_1.0_train_reduced_{n}samples.csv', index=False, sep='\t')


if __name__ == '__main__':
    remove_unused_columns()
