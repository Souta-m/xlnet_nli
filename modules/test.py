#!/usr/bin/env python3

import torch
from tqdm import tqdm

class Test:

    def __init__(self, tokenizer, log, dataset_reader):
        self._log = log
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader
        self.labels = dataset_reader.label_enumeration()

    def get_label_from_index(self, index):
        for label, label_index in self.labels.items():
            if index == label_index:
                return label
        return ""

    def validate_on_test_set(self, args, device, test_file, model):
        # Load features from datasets
        test_line = self.dataset_reader.get_file_lines(test_file)
        all_pair_ids = []
        all_predictions = []
        model.eval()
        with torch.no_grad():
            for line in tqdm(test_line):
                a, b, pair_id = self.dataset_reader.parse_line(line)
                pair_word_ids, input_mask, pair_segment_ids = self.dataset_reader.convert_text_to_features(a, b)
                tensor_word_ids = torch.tensor([pair_word_ids], dtype=torch.long, device=device)
                tensor_input_mask = torch.tensor([input_mask], dtype=torch.long, device=device)
                tensor_segment_ids = torch.tensor([pair_segment_ids], dtype=torch.long, device=device)
                model_input = {'input_ids': tensor_word_ids,  # word ids
                               'attention_mask': tensor_input_mask,  # input mask
                               'token_type_ids': tensor_segment_ids}
                outputs = model(**model_input)
                logits = outputs[0]
                prediction_id = torch.argmax(logits, dim=1)
                prediction = self.get_label_from_index(prediction_id.item())
                all_predictions.append(prediction)
                all_pair_ids.append(pair_id)

        return all_pair_ids, all_predictions
