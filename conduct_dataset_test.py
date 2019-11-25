import argparse
import random
import os
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from pytorch_transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from modules.preprocess import ConductDatasetReader
from modules.utils import global_args
from modules.log import get_logger


def run(args):
    nli_model_path = 'saved_models/xlnet-base-cased/'
    model_file = os.path.join(nli_model_path, 'pytorch_model.bin')
    config_file = os.path.join(nli_model_path, 'config.json')
    log = get_logger('conduct_test')
    model_name = 'xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=True)
    xlnet_config = XLNetConfig.from_pretrained(config_file)
    model = XLNetForSequenceClassification.from_pretrained(model_file, config=xlnet_config)
    dataset_reader = ConductDatasetReader(args, tokenizer, log)
    file_lines = dataset_reader.get_file_lines('data/dados.tsv')

    results = []
    softmax_fn = torch.nn.Softmax(dim=1)

    model.eval()
    with torch.no_grad():
        for line in tqdm(file_lines):
            premise, hypothesys, conflict = dataset_reader.parse_line(line)
            pair_word_ids, input_mask, pair_segment_ids = dataset_reader.convert_text_to_features(premise, hypothesys)
            tensor_word_ids = torch.tensor([pair_word_ids], dtype=torch.long, device=args.device)
            tensor_input_mask = torch.tensor([input_mask], dtype=torch.long, device=args.device)
            tensor_segment_ids = torch.tensor([pair_segment_ids], dtype=torch.long, device=args.device)
            model_input = {'input_ids': tensor_word_ids,  # word ids
                           'attention_mask': tensor_input_mask,  # input mask
                           'token_type_ids': tensor_segment_ids}
            outputs = model(**model_input)
            logits = outputs[0]
            nli_scores, nli_class = get_scores_and_class(logits, softmax_fn)
            nli_scores = nli_scores.detach().cpu().numpy()
            results.append({
                "conduct": premise,
                "complaint": hypothesys,
                "nli_class": nli_class,
                "nli_contradiction_score": nli_scores[0],
                "nli_entailment_score": nli_scores[1],
                "nli_neutral_score": nli_scores[2],
                "conflict": conflict
            })

    df = pd.DataFrame(results)
    df.to_csv('results/final_results.tsv', sep='\t', index=False)

def get_label_from_index(index):
    nli_labels = ["contradiction", "entailment", "neutral"]
    return nli_labels[index]

def get_scores_and_class(logits, softmax_fn):
    softmax_score = softmax_fn(logits)
    class_id = softmax_score.argmax(dim=1)
    class_label = get_label_from_index(class_id)
    return softmax_score.squeeze(), class_label

if __name__ == '__main__':
    args = global_args()
    args = args.parse_args()
    args.device = 'cpu'
    run(args)
