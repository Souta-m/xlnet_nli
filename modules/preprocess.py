#!/usr/bin/env python3
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
from modules.log import get_logger


class MNLIData:

    def __init__(self, premise, hypothesis, label):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label

    @staticmethod
    def label_map():
        return {label: i for i, label in enumerate(["contradiction", "entailment", "neutral"])}


class XLNetInputFeatures:

    def __init__(self, word_ids, segment_ids, input_mask, label):
        self.word_ids = word_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label = label


class MNLIDatasetReader:

    def __init__(self, train_file, val_file, tokenizer, max_seq_len, device):
        self.train_df = self._load_df(train_file)
        self.val_df = self._load_df(val_file)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    def _truncate(self, prem_tokens, hyp_tokens, nr_special_tokens=3):
        while True:
            total_length = len(prem_tokens) + len(hyp_tokens) + nr_special_tokens
            if total_length <= self.max_seq_len:
                break
            # Balance sequence sizes
            if len(prem_tokens) > len(hyp_tokens):
                prem_tokens.pop()
            else:
                hyp_tokens.pop()

    def load_train_dataset(self):
        return self._load_features(self.train_df, "train")

    def load_val_dataset(self):
        return self._load_features(self.val_df, "val")

    def _load_df(self, path, sep='\t'):
        df = pd.read_csv(path, sep=sep)
        df['prem'] = df['sentence1'].astype(str)
        df['hyp'] = df['sentence2'].astype(str)
        df['label'] = df['gold_label'].astype(str)
        return df[['prem', 'hyp', 'label']]

    def _load_features(self, df, dataset_type, sep_token='<sep>', cls_token='<cls>'):

        log = get_logger('preprocess')

        cache_file = "cache/cache_max_len={}_xlnet_dataset={}.cache".format(self.max_seq_len, dataset_type)
        if os.path.exists(cache_file):
            log.info('File [] already exists. Using cached features.'.format(cache_file))
            features = torch.load(cache_file)
        else:
            log.info('Cache miss. Creating features.'.format(cache_file))
            # segment identifier for each sentence
            prem_segment = 0
            hyp_segment = 1
            cls_segment = 2
            pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
            pad_mask_id = 0
            pad_segment_id = 4

            df_length = len(df)

            features = []

            for i in tqdm(range(0, df_length)):
                try:
                    row = df.iloc[i]
                    data = MNLIData(row['prem'], row['hyp'], row['label'])
                    prem_tokens = self.tokenizer.tokenize(data.premise)
                    hyp_tokens = self.tokenizer.tokenize(data.hypothesis)
                    self._truncate(prem_tokens, hyp_tokens)

                    prem_segment_ids = [prem_segment] * (len(prem_tokens) + 1)  # considering SEP token
                    hyp_segment_ids = [hyp_segment] * (len(hyp_tokens) + 1)  # considering SEP token
                    pair_segment_ids = prem_segment_ids + hyp_segment_ids + [cls_segment]

                    # XLNET representation as well as defined in pytorch-transformers
                    pair_tokens = prem_tokens + [sep_token] + hyp_tokens + [sep_token, cls_token]
                    pair_word_ids = self.tokenizer.convert_tokens_to_ids(pair_tokens)

                    # Input mask, setting 1 in position that contains a word
                    input_mask = [1] * len(pair_word_ids)

                    # Pad sequences and its mask, pad_len should not be negative because we truncate the pair before
                    padding_len = self.max_seq_len - len(pair_word_ids)
                    # for XLNet, we need to do left pad on segment ids, mask and word identifiers
                    pair_word_ids = ([pad_token_id] * padding_len) + pair_word_ids
                    input_mask = ([pad_mask_id] * padding_len) + input_mask
                    pair_segment_ids = ([pad_segment_id] * padding_len) + pair_segment_ids
                    # Just validate whether this operation includes expected number of pads
                    assert len(pair_word_ids) == self.max_seq_len
                    assert len(input_mask) == self.max_seq_len
                    assert len(pair_segment_ids) == self.max_seq_len

                    if data.label not in MNLIData.label_map():
                        log.debug("Ignoring line {} of dataset {} due to have a INVALID LABEL".format(i, dataset_type))
                    else:
                        label = MNLIData.label_map()[data.label]
                        features.append(XLNetInputFeatures(pair_word_ids, pair_segment_ids, input_mask, label))
                except Exception as exception:
                    log.error("Error at iteration {}. {}".format(i, exception))
                    raise

            log.info('Features created. Saving in file [{}].'.format(cache_file))
            torch.save(features, cache_file)
            log.info('Features saved in file [{}].'.format(cache_file))

        tensor_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long, device=self.device)
        tensor_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)
        tensor_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
        tensor_labels = torch.tensor([f.label for f in features], dtype=torch.long, device=self.device)

        return TensorDataset(tensor_word_ids, tensor_segment_ids, tensor_input_mask, tensor_labels)
