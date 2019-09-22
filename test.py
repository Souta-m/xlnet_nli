
import time
from tqdm import tqdm
import modules.preprocess
from modules.preprocess import MNLIDatasetReader
from pytorch_transformers.tokenization_xlnet import XLNetTokenizer
from pytorch_transformers import XLNetConfig

base_path = '/home/ichida/dev_env/ml/data/multinli_1.0'
train_file = '{}/multinli_1.0_train_reduced.txt'.format(base_path)
val_file = ''
test_file = ''
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_config = XLNetConfig()
max_seq_len = 128

reader = MNLIDatasetReader(train_path=train_file,
                           val_path=val_file,
                           test_path=test_file,
                           max_seq_len=max_seq_len,
                           tokenizer=tokenizer)

features = reader.load_train_features()

pass

