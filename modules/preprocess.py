#!/usr/bin/env python3
import csv
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

    def __init__(self, word_ids, input_mask, segment_ids, label):
        self.word_ids = word_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label = label


class MNLIDatasetReader:

    def __init__(self, train_file, val_file, tokenizer, max_seq_len, device):
        self.train_file = train_file
        self.val_file = val_file
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
        return self._load_features(self.train_file, "train")

    def load_val_dataset(self):
        return self._load_features(self.val_file, "val")

    def _assert_seq_lens(self, word_ids, input_mask, segment_ids):
        assert len(word_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

    def _get_file_lines(self, filename, ignore_headers=True):
        lines = []
        with open(filename, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                lines.append(line)
        return lines[1:] if ignore_headers else lines

    def _load_features(self, filename, dataset_type):

        log = get_logger('preprocess')

        cache_file = f'cache/max_len={self.max_seq_len}_dataset={dataset_type}-xlnet.cache'
        if os.path.exists(cache_file):
            log.info(f'File {cache_file} already exists. Using cached features.')
            features = torch.load(cache_file)
        else:
            log.info(f'Cache miss. Retrieving features from file {filename}')
            lines = self._get_file_lines(filename)
            total_lines = len(lines)
            log.info(f'Loaded {total_lines} examples from dataset {dataset_type}')

            # special tokens and its ids for XLNET input
            prem_segment_id = 0
            hyp_segment_id = 1
            cls_segment_id = 2
            mask_pad_id = 0
            segment_pad_id = 4
            token_pad_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
            sep_token = '<sep>'
            cls_token = '<cls>'

            # Retrieve features from file lines
            features = []
            for i in tqdm(range(0, total_lines)):
                try:
                    line = lines[i]
                    data = MNLIData(line[0], line[1], line[2])
                    prem_tokens = self.tokenizer.tokenize(data.premise)
                    hyp_tokens = self.tokenizer.tokenize(data.hypothesis)
                    self._truncate(prem_tokens, hyp_tokens)
                    # XLNET representation = PREM + [SEP] + HYP + [SEP] + [CLS]
                    pair_tokens = prem_tokens + [sep_token] + hyp_tokens + [sep_token, cls_token]
                    pair_word_ids = self.tokenizer.convert_tokens_to_ids(pair_tokens)
                    # Input mask, setting 1 in position that contains a word and setting MASK_PAD otherwise
                    input_mask = [1] * len(pair_word_ids)
                    # Segment ID of each sentence
                    prem_segment_ids = [prem_segment_id] * (len(prem_tokens) + 1)  # considering SEP token
                    hyp_segment_ids = [hyp_segment_id] * (len(hyp_tokens) + 1)  # considering SEP token
                    pair_segment_ids = prem_segment_ids + hyp_segment_ids + [cls_segment_id]

                    # Pad sequences and its mask, pad_len should not be negative because we truncate the pair before
                    padding_len = self.max_seq_len - len(pair_word_ids)
                    # for XLNet, we need to do left pad on segment ids, mask and word identifiers
                    pair_word_ids = ([token_pad_id] * padding_len) + pair_word_ids
                    input_mask = ([mask_pad_id] * padding_len) + input_mask
                    pair_segment_ids = ([segment_pad_id] * padding_len) + pair_segment_ids

                    self._assert_seq_lens(pair_word_ids, input_mask, pair_segment_ids)
                    if data.label in MNLIData.label_map():
                        label_id = MNLIData.label_map()[data.label]
                        features.append(XLNetInputFeatures(pair_word_ids, input_mask, pair_segment_ids, label_id))
                except Exception as exception:
                    log.error("Error at iteration {}. {}".format(i, exception))
                    raise

            log.info('Features created. Saving in file [{}].'.format(cache_file))
            torch.save(features, cache_file)
        # Converting features into Tensors
        tensor_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)
        tensor_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        tensor_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        tensor_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return TensorDataset(tensor_word_ids, tensor_input_mask, tensor_segment_ids, tensor_labels)
