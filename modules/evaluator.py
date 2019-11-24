#!/usr/bin/env python3
import torch
from tqdm import tqdm


class Evaluator:

    def __init__(self, dataloader, logger):
        self.dataloader = dataloader
        self.logger = logger

    def __call__(self, model, device, loader_type):
        all_predictions = torch.tensor([], device=device, dtype=torch.long)
        all_labels = torch.tensor([], device=device, dtype=torch.long)
        all_losses = torch.tensor([], device=device, dtype=torch.float)
        for batch in tqdm(self.dataloader, desc=loader_type):
            model.eval()
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                model_input = {'input_ids': batch[0],  # word ids
                               'attention_mask': batch[1],  # input mask
                               'token_type_ids': batch[2],  # segment ids
                               'labels': batch[3]}  # labels
                outputs = model(**model_input)
                loss, logits = outputs[0:2]
                predictions = torch.argmax(logits, dim=1)
                all_predictions = torch.cat([all_predictions, predictions])
                all_labels = torch.cat([all_labels, model_input['labels']])
                all_losses = torch.cat([all_losses, loss.reshape(1)])
        return all_predictions, all_labels, all_losses.mean()
