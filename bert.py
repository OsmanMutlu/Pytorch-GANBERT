import csv
import os
import logging
import argparse
import random
import datetime
from tqdm import tqdm, trange
from pathlib import Path
import math
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix

import subprocess
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(filename = '{}_log.txt'.format(datetime.datetime.now()),
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

logger.info(PYTORCH_PRETRAINED_BERT_CACHE)

max_seq_length = 512
batch_size = 32
learning_rate = 2e-5
unlabel_rate = 0.05

label_list = ["0", "1"]
tokenizer = BertTokenizer.from_pretrained("/home/omutlu/.pytorch_pretrained_bert/bert-base-uncased-vocab.txt")
bert_model = "/home/omutlu/.pytorch_pretrained_bert/bert-base-uncased.tar.gz"
repo_path = "/home/omutlu/domain_adaptation/"
# train_filename = "/home/omutlu/domain_adaptation/data/sorted_data/books/train.json"
# val_filename = "/home/omutlu/domain_adaptation/data/sorted_data/books/val.json"
# test_filename = "/home/omutlu/domain_adaptation/data/sorted_data/electronics/test.json"
train_filename = repo_path + "data/domain_corpus_data/clef/balanced_india_train.json"
val_filename = repo_path + "data/domain_corpus_data/clef/india_dev.json"
test_filename = repo_path + "data/domain_corpus_data/clef/india_test.json"
output_file = repo_path + "models/doc_gan/base-bert_512_32_0,05.pt"
total_epoch_num = 15
warmup_proportion = 0.1

train = True
multi_gpu = True
device = torch.device("cuda:4")
# device = "cpu"
device_ids = [4,5,6,7]
dev_metric = "f1"
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Hyper-params
print("Maximum sequence length : %d" %max_seq_length)
print("Batch size : %d" %batch_size)
print("Learning rate : %.8f" %learning_rate)

class DomainData(Dataset):
    def __init__(self, examples, label_list, max_seq_length, tokenizer):
        self.examples = examples
        self.label_list = label_list
        # label_map = {}
        label_map = {"-1":-1}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        self.label_map = label_map
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def convert_examples_to_features(self, text, label=None):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        tokens_a = self.tokenizer.tokenize(text)
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)

        tokens.append("[SEP]")

    #    tokens = [token for token in tokens if token in self.tokenizer.vocab.keys() else "[UNK]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        if label:
            label_id = self.label_map[label]
            return input_ids, input_mask, label_id
        else:
            return input_ids, input_mask

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids, input_mask, label_id = self.convert_examples_to_features(ex[0], label=ex[1]) # input is -> text, label

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        label_ids = torch.tensor(label_id, dtype=torch.long)

        return input_ids, input_mask, label_ids

def get_examples(filename, train=False):
    lines = pd.read_json(filename, orient="records", lines=True)
    examples = []
    for (i, line) in lines.iterrows():
        guid = i
        text = str(line.text)
        label = str(int(line.label))
        if train and np.random.rand() > unlabel_rate:
            label = "-1" # make the sample unlabeled

        examples.append((text, label))

    return examples

if train:
    train_examples = get_examples(train_filename, train=True)
    val_examples = get_examples(val_filename)
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    train_dataloader = DataLoader(dataset=DomainData(train_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = DomainData(val_examples, label_list, max_seq_length, tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    num_train_steps = int(len(train_examples) / batch_size * total_epoch_num)

    model = BertForSequenceClassification.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=len(label_list))

    model.to(device)
    if multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    global_step = 0
    best_score = 0.0
    model.train()
    for epoch_num in trange(int(total_epoch_num), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            src_input_ids, src_input_mask, label_ids = batch

            loss, _ = model(src_input_ids, attention_mask=src_input_mask, labels=label_ids)

            if multi_gpu:
                loss = loss.mean()

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += src_input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            model.zero_grad()
            global_step += 1

        # VALIDATION
        model.eval()
        all_preds = np.array([])
        all_label_ids = np.array([])
        eval_loss = 0
        nb_eval_steps = 0
        for src_input_ids, src_input_mask, label_ids in val_dataloader:
            src_input_ids = src_input_ids.to(device)
            src_input_mask = src_input_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(src_input_ids, attention_mask=src_input_mask, labels=label_ids)

            eval_loss += tmp_eval_loss.mean().item()

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            all_preds = np.append(all_preds, np.argmax(logits, axis=1))
            all_label_ids = np.append(all_label_ids, label_ids)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="micro", labels=list(range(0,len(label_list))))
        mcc = matthews_corrcoef(all_preds, all_label_ids)
        result = {"eval_loss": eval_loss,
                  "precision_micro": precision,
                  "recall_micro": recall,
                  "f1_micro": f1,
                  "mcc": mcc}

        if dev_metric == "f1":
            score = f1
        elif dev_metric == "recall":
            score = recall
        elif dev_metric == "precision":
            score = precision

        if best_score < score:
            best_score = score
            logger.info("Saving model...")
            model_to_save = model.module if hasattr(model, 'module') else model  # To handle multi gpu
            torch.save(model_to_save.state_dict(), output_file)

        logger.info("***** Epoch " + str(epoch_num + 1) + " *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %.4f", key, result[key])

        model.train() # back to training


test_examples = get_examples(test_filename)
test_dataloader = DataLoader(dataset=DomainData(test_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=len(label_list))

model.load_state_dict(torch.load(output_file))
model.to(device)
if multi_gpu:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

all_preds = np.array([])
all_label_ids = np.array([])
test_loss = 0.0
nb_test_steps = 0
model.eval()
for src_input_ids, src_input_mask, label_ids in test_dataloader:
    src_input_ids = src_input_ids.to(device)
    src_input_mask = src_input_mask.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        tmp_test_loss, logits = model(src_input_ids, attention_mask=src_input_mask, labels=label_ids)

    test_loss += tmp_test_loss.mean().item()

    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    all_preds = np.append(all_preds, np.argmax(logits, axis=1))
    all_label_ids = np.append(all_label_ids, label_ids)

    nb_test_steps += 1

test_loss = test_loss / nb_test_steps
precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="micro", labels=list(range(0,len(label_list))))
mcc = matthews_corrcoef(all_preds, all_label_ids)
result = {"test_loss": test_loss,
          "precision_micro": precision,
          "recall_micro": recall,
          "f1_micro": f1,
          "mcc": mcc}

logger.info("***** TEST RESULTS *****")
for key in sorted(result.keys()):
    logger.info("  %s = %.4f", key, result[key])
