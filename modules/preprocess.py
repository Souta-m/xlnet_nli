#!/usr/bin/env python3
import csv
import os

from abc import abstractmethod

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm


class XLNetInputFeatures:

    def __init__(self, word_ids, input_mask, segment_ids, label):
        self.word_ids = word_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label = label


class DatasetReader:

    def __init__(self, args, tokenizer, log):
        """
        Reads datasets and transform texts into features. The files must be in CSV format.
        :param args: execution args
        :param tokenizer: model tokenizer
        """
        self.train_file = os.path.join(args.base_path, args.train_file)
        self.val_file = os.path.join(args.base_path, args.val_file)
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self._log = log

    @abstractmethod
    def parse_line(self, line_fields):
        """
        :param line_fields: an array that contains the field values given a dataset row
        :return: must return a tuple (sentence_a, sentence_b, label)
        """
        pass

    @abstractmethod
    def label_enumeration(self):
        """
        :return: List of expected labels contained in the dataset having the string representation and a index.
        """
        pass

    @abstractmethod
    def dataset_name(self):
        """
        :return: The name of dataset
        """
        pass

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

    def load_train_dataloader(self):
        train_dataset = self._load_features(self.train_file, "train")
        return DataLoader(train_dataset, self.batch_size, RandomSampler(train_dataset))

    def load_val_dataloader(self):
        val_dataset = self._load_features(self.val_file, "val")
        return DataLoader(val_dataset, self.batch_size, SequentialSampler(val_dataset))

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
        """
        Transform text into XLNet input features
        :param filename: Filename that contains the input text
        :param dataset_type: Type of dataset (train, val, test)
        :return: TensorDataset that contains each input features converted into torch.tensor
        """

        cache_file = f'cache/max_len={self.max_seq_len}_dataset={self.dataset_name()}_{dataset_type}-xlnet.cache'
        if os.path.exists(cache_file):
            self._log.info(f'File {cache_file} already exists. Using cached features.')
            features = torch.load(cache_file)
        else:
            self._log.info(f'Cache miss. Retrieving features from file {filename}')
            lines = self._get_file_lines(filename)
            total_lines = len(lines)
            self._log.info(f'Loaded {total_lines} examples from dataset {dataset_type}')

            # special tokens and its ids for XLNET input.
            segment_a_id = 0
            segment_b_id = 1
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
                    sentence_a, sentence_b, label = self.parse_line(lines[i])
                    tokens_a = self.tokenizer.tokenize(sentence_a)
                    tokens_b = self.tokenizer.tokenize(sentence_b)
                    self._truncate(tokens_a, tokens_b)
                    # XLNET representation = PREM + [SEP] + HYP + [SEP] + [CLS]
                    pair_tokens = tokens_a + [sep_token] + tokens_b + [sep_token, cls_token]
                    pair_word_ids = self.tokenizer.convert_tokens_to_ids(pair_tokens)
                    # Input mask, setting 1 in position that contains a word and setting MASK_PAD otherwise
                    input_mask = [1] * len(pair_word_ids)
                    # Segment ID of each sentence
                    segment_a_ids = [segment_a_id] * (len(tokens_a) + 1)  # considering SEP token
                    segment_b_ids = [segment_b_id] * (len(tokens_b) + 1)  # considering SEP token
                    pair_segment_ids = segment_a_ids + segment_b_ids + [cls_segment_id]

                    # Pad sequences and its mask, pad_len should not be negative because we truncate the pair before
                    padding_len = self.max_seq_len - len(pair_word_ids)
                    # for XLNet, we need to do left pad on segment ids, mask and word identifiers
                    pair_word_ids = ([token_pad_id] * padding_len) + pair_word_ids
                    input_mask = ([mask_pad_id] * padding_len) + input_mask
                    pair_segment_ids = ([segment_pad_id] * padding_len) + pair_segment_ids

                    self._assert_seq_lens(pair_word_ids, input_mask, pair_segment_ids)
                    if label in self.label_enumeration():
                        label_id = self.label_enumeration()[label]
                        features.append(XLNetInputFeatures(pair_word_ids, input_mask, pair_segment_ids, label_id))
                except Exception as exception:
                    self._log.error("Error at iteration {}. {}".format(i, exception))
                    raise

            self._log.info('Features created. Saving in file [{}].'.format(cache_file))
            torch.save(features, cache_file)
        # Converting features into Tensors
        tensor_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)
        tensor_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        tensor_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        tensor_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return TensorDataset(tensor_word_ids, tensor_input_mask, tensor_segment_ids, tensor_labels)


class MNLIDatasetReader(DatasetReader):

    def label_enumeration(self):
        return {label: i for i, label in enumerate(["contradiction", "entailment", "neutral"])}

    def parse_line(self, line_fields):
        return line_fields[8], line_fields[9], line_fields[-1]

    def dataset_name(self):
        return "MNLI"
