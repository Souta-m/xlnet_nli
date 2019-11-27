#!/usr/bin/env python3

import torch
import os
from torch.nn import Softmax

from modules.log import get_logger
from modules.train import TrainModel
from modules.preprocess import init_dataset_reader
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer


class Model:

    def __init__(self, args, task_name, weight_file=None, config_file=None):
        self.args = args
        self.device = args.device
        self.log = self.get_train_logger(args, task_name)
        self.softmax = Softmax(dim=1)
        self.tokenizer = XLNetTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.dataset_reader = init_dataset_reader(task_name, args, self.tokenizer, self.log)

        config = args.model_name if config_file is None else config_file
        model_weights = args.model_name if weight_file is None else weight_file
        xlnet_config = XLNetConfig.from_pretrained(config,
                                                   output_hidden_states=True,
                                                   output_attentions=True,
                                                   num_labels=3,
                                                   finetuning_task=task_name)

        model = XLNetForSequenceClassification.from_pretrained(model_weights, config=xlnet_config)
        self.model = model.to(args.device)

    def get_train_logger(self, args, task_name):
        logger_name = f'{args.model_name}-batch{args.batch_size}-seq{args.max_seq_len}' \
            f'-warmup{args.warmup_steps}-ep{args.epochs}-{task_name}'
        return get_logger(logger_name)

    def predict(self, sentence_a, sentence_b, softmax_scores=True):
        self.model.eval()
        with torch.no_grad():
            features = self._get_features(sentence_a, sentence_b)
            outputs = self.model(**features)
            logits = outputs[0]
            return self.softmax(logits) if softmax_scores else logits

    def _get_features(self, sentence_a, sentence_b):
        word_ids, mask, segment_ids = self.dataset_reader.convert_text_to_features(sentence_a, sentence_b)
        tensor_word_ids = torch.tensor([word_ids], dtype=torch.long, device=self.device)
        tensor_input_mask = torch.tensor([mask], dtype=torch.long, device=self.device)
        tensor_segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        return {'input_ids': tensor_word_ids,  # word ids
                'attention_mask': tensor_input_mask,  # input mask
                'token_type_ids': tensor_segment_ids}

    def finetune_model(self):
        train_file = os.path.join(self.args.base_path, self.args.train_file)
        val_file = os.path.join(self.args.base_path, self.args.val_file)
        train_dataloader = self.data_loader.load_train_dataloader(train_file)
        val_dataloader = self.data_loader.load_val_dataloader(val_file)

        trainer = TrainModel(train_dataloader, val_dataloader, self.log)
        trainer.train(self.model, self.device, self.args)
