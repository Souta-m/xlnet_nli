#!/usr/bin/env python3
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


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

    def __init__(self, train_file, val_file, tokenizer, max_seq_len):
        self.train_df = pd.read_csv(train_file, sep='\t')
        self.val_df = pd.read_csv(val_file, sep='\t')
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

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
        return self._load_features(self.train_df)

    def load_val_dataset(self):
        return self._load_features(self.val_df)

    def _load_features(self, df):

        features_df = pd.DataFrame()
        features_df['prem'] = df['sentence1'].astype(str)
        features_df['hyp'] = df['sentence2'].astype(str)
        features_df['label'] = df['gold_label'].astype(str)
        sep_token = '[SEP]'
        cls_token = '[CLS]'
        # segment identifier for each sentence
        prem_segment = 0
        hyp_segment = 1
        cls_segment = 1
        pad_token_id = 0

        df_length = len(df)

        features = []

        for i in tqdm(range(0, df_length)):
            try:
                row = features_df.iloc[i]
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
                input_mask = ([pad_token_id] * padding_len) + input_mask
                pair_segment_ids = ([pad_token_id] * padding_len) + pair_segment_ids
                # Just validate whether this operation includes expected number of pads
                assert len(pair_word_ids) == self.max_seq_len
                assert len(input_mask) == self.max_seq_len
                assert len(pair_segment_ids) == self.max_seq_len

                label = MNLIData.label_map()[data.label]

                features.append(XLNetInputFeatures(pair_word_ids, pair_segment_ids, input_mask, label))
            except Exception as exception:
                print("Error at iteration {}.".format(exception))
                raise

        tensor_word_ids = torch.tensor([feature.word_ids for feature in features], dtype=torch.long)
        tensor_segment_ids = torch.tensor([feature.segment_ids for feature in features], dtype=torch.long)
        tensor_input_mask = torch.tensor([feature.input_mask for feature in features], dtype=torch.long)
        tensor_labels = torch.tensor([feature.label for feature in features], dtype=torch.long)

        return TensorDataset(tensor_word_ids, tensor_segment_ids, tensor_input_mask, tensor_labels)
