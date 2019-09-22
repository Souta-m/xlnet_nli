import torch
from pytorch_transformers import *

# PyTorch-Transformers has a unified API
# for 7 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
'''
MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,      'gpt2'),
          (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024'),
          (RobertaModel,    RobertaTokenizer,   'roberta-base')]
'''
#MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased')]
# Let's encode some text in a sequence of hidden-states using each model:
#for model_class, tokenizer_class, pretrained_weights in MODELS:
#    # Load pretrained model/tokenizer
#    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#    model = model_class.from_pretrained(pretrained_weights)
#    model.to(torch.device('cuda'))
    # Encode text
    #input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    #with torch.no_grad():
    #    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
#print('modeo loaded')
# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
#BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
#                      BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
#                      BertForQuestionAnswering]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#for model_class in BERT_MODEL_CLASSES:
    # Load pretrained model/tokenizer
#    model = model_class.from_pretrained('bert-base-uncased')
print('Loading Model')
# Models can return full list of hidden-states & attentions weights at each layer
model_class = XLNetForSequenceClassification
pretrained_weights = 'xlnet-base-cased'
model = model_class.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)
tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
all_hidden_states, all_attentions = model(input_ids)[-2:]

# Models are compatible with Torchscript
#model = model_class.from_pretrained(pretrained_weights, torchscript=True)
#traced_model = torch.jit.trace(model, (input_ids,))

# Simple serialization for models and tokenizers
#model.save_pretrained('./directory/to/save/')  # save
#odel = model_class.from_pretrained('./directory/to/save/')  # re-load
#tokenizer.save_pretrained('./directory/to/save/')  # save
#tokenizer = tokenizer_class.from_pretrained('./directory/to/save/')  # re-load

#import time

#time.sleep(20)