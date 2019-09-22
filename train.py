#!/usr/bin/env python3

from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from modules.preprocess import MNLIDatasetReader
from torch.utils.data import DataLoader, RandomSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule

def train_configs(tokenizer):
    base_path = '/home/ichida/dev_env/ml/data/multinli_1.0'
    return {
        'train_file': '{}/multinli_1.0_train_reduced.txt'.format(base_path),
        'val_file': '{}/multinli_1.0_dev_matched_reduced.txt'.format(base_path),
        'max_seq_len': 128,
        'tokenizer': tokenizer
    }


def train():

    batch_size = 8
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    clip_norm = 1.0
    warmup_steps = 0
    train_epochs = 8

    pretrained_weights = 'xlnet-base-cased'
    model = XLNetForSequenceClassification.from_pretrained(pretrained_weights,
                                                           output_hidden_states=True,
                                                           output_attentions=True)

    tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
    paths = train_configs(tokenizer)
    reader = MNLIDatasetReader(**paths)
    train_dataset = reader.load_train_features()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, train_sampler, batch_size)

    t_total = len(train_dataloader) // 1 * train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)




if __name__ == '__main__':
    train()
