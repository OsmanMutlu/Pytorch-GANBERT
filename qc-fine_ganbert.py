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

from model import Generator1, Discriminator

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(filename = '{}_log.txt'.format(datetime.datetime.now()),
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

logger.info(PYTORCH_PRETRAINED_BERT_CACHE)

max_seq_length = 64
batch_size = 64
bert_learning_rate = 2e-5
learning_rate = 2e-5
noise_size = 100
epsilon = 1e-8
label_rate = 0.02

label_list = ["ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc", "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body", "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed", "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang", "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product", "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol", "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc", "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country", "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count", "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other", "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize", "NUM_weight"]

tokenizer = BertTokenizer.from_pretrained("/home/omutlu/.pytorch_pretrained_bert/bert-base-uncased-vocab.txt")
bert_model = "/home/omutlu/.pytorch_pretrained_bert/bert-base-uncased.tar.gz"
repo_path = "/home/omutlu/domain_adaptation/"
train_filename = repo_path + "doc_gan/ganbert_tf/data/labeled_and_unlabeled.tsv"
test_filename = repo_path + "doc_gan/ganbert_tf/data/test.tsv"
bert_output_file = repo_path + "models/doc_gan/qc-fine_gan-bert2_64_64_2e-5_50_bert.pt"
dis_output_file = repo_path + "models/doc_gan/qc-fine_gan-bert2_64_64_2e-5_50_dis.pt"
total_epoch_num = 50
warmup_proportion = 0.1

train = True
multi_gpu = True
device = torch.device("cuda:4")
# device = "cpu"
device_ids = [4,5,6,7]
dev_metric = "f1"
seed = 42
n_disc = 3

nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
adversarial_loss = torch.nn.BCELoss()

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

def get_examples(filename):
    examples = []

    with open(filename, 'r') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list[1:]:
            split = line.split(" ")
            question = ' '.join(split[1:])

            text_a = question
            inn_split = split[0].split(":")
            label = inn_split[0] + "_" + inn_split[1]

            if label == "UNK_UNK":
                label = "-1"
            else:
                # Balance out the labeled data
                repeat_num = int(1/label_rate)
                repeat_num = int(math.log(repeat_num, 2)) - 1
                if repeat_num < 0:
                    repeat_num = 0

                for _ in range(0, repeat_num):
                    examples.append((text_a, label))

            examples.append((text_a, label))
        f.close()

    return examples

if train:
    train_examples = get_examples(train_filename)
    # random.shuffle(train_examples)
    train_dataloader = DataLoader(dataset=DomainData(train_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size, shuffle=False, drop_last=False)

    num_train_steps = int(len(train_examples) / batch_size * total_epoch_num)

    bert = BertModel.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
    generator = Generator1(noise_size=noise_size, output_size=768, hidden_sizes=[768], dropout_rate=0.1)
    discriminator = Discriminator(input_size=768, hidden_sizes=[768], num_labels=len(label_list), dropout_rate=0.1)

    bert.to(device)
    if multi_gpu:
        bert = torch.nn.DataParallel(bert, device_ids=device_ids)

    generator.to(device)
    discriminator.to(device)

    param_optimizer = list(bert.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    bert_optimizer = BertAdam(optimizer_grouped_parameters,
                              lr=bert_learning_rate,
                              warmup=warmup_proportion,
                              t_total=num_train_steps)

    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    dis_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    global_step = 0
    best_loss = 1000.0
    bert.train()
    generator.train()
    discriminator.train()
    for epoch_num in trange(int(total_epoch_num), desc="Epoch"):
        # only_train_dis = False
        # if epoch_num == 0 or tr_d_loss/2 > tr_g_loss:
        #     only_train_dis = True

        tr_g_loss = 0
        tr_d_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            src_input_ids, src_input_mask, label_ids = batch

            # Adversarial ground truths
            valid = torch.zeros(src_input_ids.shape[0], 1, device=device)
            fake = torch.ones(src_input_ids.shape[0], 1, device=device)

            bert.zero_grad()
            discriminator.zero_grad()
            generator.zero_grad()

            # Random noise
            noise = torch.zeros(src_input_ids.shape[0],noise_size, device=device).uniform_(0, 1).requires_grad_(True)

            # Real representations
            _, doc_rep = bert(src_input_ids, attention_mask=src_input_mask, output_all_encoded_layers=False)
            D_real_features, D_real_logits, D_real_probs = discriminator(doc_rep)

            # Generated representations
            gen_rep = generator(noise)

            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep.detach()) # for discriminator

            # TRAIN SEPERATELY:
            # Discriminator loss and step
            bert_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            label_probs = torch.nn.functional.softmax(D_real_probs[:,:-1], dim=-1)
            d_label_loss = nll_loss(label_probs, label_ids.view(-1))
            d_gan_loss = adversarial_loss(D_real_probs[:,-1], valid) + adversarial_loss(D_fake_probs[:,-1], fake)
            d_loss = d_label_loss + d_gan_loss

            d_loss.backward()
            bert_optimizer.step()
            dis_optimizer.step()

            # Generator loss and step
            gen_optimizer.zero_grad()
            discriminator.zero_grad()
            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep) # for generator
            # NOTE: 0 and 1 is reserved as negative and positive, so we use the last index for "isFake?" probability.
            g_loss = adversarial_loss(D_fake_probs[:,-1], valid)
            g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features.detach(), dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss += g_feat_reg

            # if not only_train_dis:
            # if True:
            if global_step > 200 or global_step % n_disc == 0:
                g_loss.backward()
                gen_optimizer.step()

            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()
            nb_tr_examples += src_input_ids.size(0)
            nb_tr_steps += 1
            global_step += 1

        tr_g_loss /= nb_tr_steps
        tr_d_loss /= nb_tr_steps

        curr_loss = (tr_g_loss + tr_d_loss * 2) / 3
        if curr_loss < best_loss:
            logger.info("***** Saving Model *****")
            best_loss = curr_loss
            model_to_save = bert.module if hasattr(bert, 'module') else bert  # To handle multi gpu
            torch.save(model_to_save.state_dict(), bert_output_file)
            torch.save(discriminator.state_dict(), dis_output_file)

        logger.info("***** Epoch " + str(epoch_num + 1) + " *****")
        logger.info("  gen_loss = %.4f", tr_g_loss)
        logger.info("  dis_loss = %.4f", tr_d_loss)


test_examples = get_examples(test_filename)
test_dataloader = DataLoader(dataset=DomainData(test_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size)

bert = BertModel.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
discriminator = Discriminator(input_size=768, hidden_sizes=[768], num_labels=len(label_list), dropout_rate=0.1)

bert.load_state_dict(torch.load(bert_output_file))
discriminator.load_state_dict(torch.load(dis_output_file))

discriminator.to(device)
bert.to(device)
if multi_gpu:
    bert = torch.nn.DataParallel(bert, device_ids=device_ids)


all_preds = np.array([])
all_label_ids = np.array([])
test_loss = 0.0
nb_test_steps = 0
bert.eval()
discriminator.eval()
for src_input_ids, src_input_mask, label_ids in test_dataloader:
    src_input_ids = src_input_ids.to(device)
    src_input_mask = src_input_mask.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        _, doc_rep = bert(src_input_ids, attention_mask=src_input_mask)
        _, logits, probs = discriminator(doc_rep)
        probs = torch.nn.functional.softmax(probs[:,:-1], dim=-1)
        tmp_test_loss = nll_loss(probs, label_ids.view(-1))

    test_loss += tmp_test_loss.mean().item()

    logits = logits[:,:-1]
    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    all_preds = np.append(all_preds, np.argmax(logits, axis=1))
    all_label_ids = np.append(all_label_ids, label_ids)

    nb_test_steps += 1

print(all_preds)
test_loss = test_loss / nb_test_steps
precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="micro", labels=list(range(0,len(label_list))))
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="macro", labels=list(range(0,len(label_list))))
mcc = matthews_corrcoef(all_preds, all_label_ids)
result = {"test_loss": test_loss,
          "precision_micro": precision,
          "recall_micro": recall,
          "f1_micro": f1,
          "precision_macro": prec_macro,
          "recall_macro": rec_macro,
          "f1_macro": f1_macro,
          "mcc": mcc}

logger.info("***** TEST RESULTS *****")
for key in sorted(result.keys()):
    logger.info("  %s = %.4f", key, result[key])
