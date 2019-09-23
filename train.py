#!/usr/bin/env python3

from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from modules.preprocess import MNLIDatasetReader
import torch
from torch.utils.data import DataLoader, RandomSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange


def train_configs(tokenizer):
    base_path = '/home/ichida/dev_env/ml/data/multinli_1.0'
    return {
        'train_file': '{}/multinli_1.0_train_reduced.txt'.format(base_path),
        'val_file': '{}/multinli_1.0_dev_matched_reduced.txt'.format(base_path),
        'max_seq_len': 128,
        'tokenizer': tokenizer
    }


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    clip_norm = 1.0
    warmup_steps = 0
    train_epochs = 8
    weight_decay = 0.0

    pretrained_weights = 'xlnet-base-cased'
    model = XLNetForSequenceClassification.from_pretrained(pretrained_weights,
                                                           output_hidden_states=True,
                                                           output_attentions=True)

    tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
    paths = train_configs(tokenizer)
    reader = MNLIDatasetReader(**paths)
    train_dataset = reader.load_train_dataset()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, train_sampler, batch_size)

    t_total = len(train_dataloader) // train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    for epoch in trange(train_epochs, desc="Epoch"):
        train_epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_epoch_iterator):
            # Send all tensors created in preprocess to target device (GPU agnostic)
            # TODO: test send tensor to gpu in preprocess!
            tensor_batch = tuple(tensor.to(device) for tensor in batch)

            model_input = {'input_ids': tensor_batch[0],  # word ids
                           'attention_mask': tensor_batch[1],  # input mask
                           'token_type_ids': tensor_batch[2],  # segment ids
                           'labels': tensor_batch[3]}  # labels

            outputs = model(**model_input)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # grad clip
            scheduler.step()
            optimizer.step()
            model.zero_grad()

        # TODO: run evaluation on val test and compare metrics during train epochs
        evaluation(epoch, model, tokenizer, reader, batch_size, device)


def evaluation(epoch, model, reader, val_batch_size, device):
    val_dataset = reader.load_val_dataset()
    val_sampler = RandomSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, val_sampler, val_batch_size)
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
            val_loss = outputs[0]
            epoch_val_loss += val_loss.mean().item()
        executed_steps += 1

    epoch_val_loss = epoch_val_loss / executed_steps
    print("Validation loss at epoch {}: {}".format(epoch, epoch_val_loss))


if __name__ == '__main__':
    train()
