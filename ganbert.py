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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import Generator1, Discriminator

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

max_seq_length = 512
batch_size = 32
bert_learning_rate = 2e-5
learning_rate = 1e-4
noise_size = 100
epsilon = 1e-8
unlabel_rate = 0.02

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
bert_output_file = repo_path + "models/doc_gan/gan-bert_512_32_1e-4_0,02_bert.pt"
dis_output_file = repo_path + "models/doc_gan/gan-bert_512_32_1e-4_0,02_dis.pt"
total_epoch_num = 30
warmup_proportion = 0.1

train = True
multi_gpu = True
device = torch.device("cuda:4")
# device = "cpu"
device_ids = [4,5,6,7]
dev_metric = "f1"
seed = 42

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

def get_examples(filename, train=False):
    lines = pd.read_json(filename, orient="records", lines=True)
    examples = []
    for (i, line) in lines.iterrows():
        guid = i
        text = str(line.text)
        label = str(int(line.label))
        if train and np.random.rand() > unlabel_rate:
            label = "-1" # make the sample unlabeled
        else:
            # Balance out the labeled data
            repeat_num = int(1/unlabel_rate)
            repeat_num = int(math.log(repeat_num, 2)) - 1
            if repeat_num < 0:
                repeat_num = 0

            for _ in range(0, repeat_num):
                examples.append((text, label))

        examples.append((text, label))

    return examples

def get_examples_qc_rest(filename, train=False):
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
                repeat_num = int(1/unlabel_rate)
                repeat_num = int(math.log(repeat_num, 2)) - 1
                if repeat_num < 0:
                    repeat_num = 0

                for _ in range(0, repeat_num):
                    examples.append((text_a, label))

            examples.append((text_a, label))
        f.close()

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
    best_score = 0.0
    bert.train()
    generator.train()
    discriminator.train()
    for epoch_num in trange(int(total_epoch_num), desc="Epoch"):
        # only_train_dis = False
        # if epoch_num == 0 or tr_d_loss/3 > tr_g_loss:
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


            # TRAIN SEPERATELY:

            # Generator loss and step
            gen_optimizer.zero_grad()
            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep) # for generator
            g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features.detach(), dim=0) - torch.mean(D_fake_features, dim=0), 2))
            # NOTE: 0 and 1 is reserved as negative and positive, so we use the -1 index for "isFake?" probability.
            g_loss = adversarial_loss(D_fake_probs[:,-1], valid)
            g_loss += g_feat_reg

            # if not only_train_dis:
            if True:
            # if global_step > -1 or (tr_d_loss / (nb_tr_steps+1)) < 2.0:
                g_loss.backward()
                gen_optimizer.step()


            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep.detach()) # for discriminator
            # Discriminator loss and step
            bert_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            label_probs = torch.nn.functional.softmax(D_real_probs[:,:-1], dim=-1)
            d_label_loss = nll_loss(label_probs, label_ids.view(-1))
            # d_reg_loss = 0.5 - torch.mean(torch.std(label_probs, dim=0))
            # d_label_loss += d_reg_loss
            d_gan_loss = (adversarial_loss(D_real_probs[:,-1], valid) + adversarial_loss(D_fake_probs[:,-1], fake)) / 2
            d_loss = d_label_loss + d_gan_loss
            d_loss.backward()
            bert_optimizer.step()
            dis_optimizer.step()


            # TRAIN TOGETHER:
            # bert_optimizer.zero_grad()
            # gen_optimizer.zero_grad()
            # dis_optimizer.zero_grad()

            # # Generator loss
            # # NOTE: 0 and 1 is reserved as negative and positive, so we use the 2 index for "isReal?" probability.
            # g_loss = -1 * torch.mean(torch.log(1 - D_fake_probs[:,2] + epsilon))
            # g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            # g_loss += g_feat_reg

            # # Discriminator loss
            # label_probs = torch.nn.functional.softmax(D_real_probs[:,:-1], dim=-1)
            # d_label_loss = nll_loss(label_probs, label_ids.view(-1))
            # d_gan_loss = -1 * torch.mean(torch.log(1 - D_real_probs[:,2] + epsilon)) + -1 * torch.mean(torch.log(D_fake_probs[:,2] + epsilon))
            # d_loss = d_label_loss + d_gan_loss

            # loss = d_loss + g_loss
            # loss.backward()
            # bert_optimizer.step()
            # gen_optimizer.step()
            # dis_optimizer.step()


            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()
            nb_tr_examples += src_input_ids.size(0)
            nb_tr_steps += 1
            global_step += 1

        tr_g_loss /= nb_tr_steps
        tr_d_loss /= nb_tr_steps

        # VALIDATION
        bert.eval()
        discriminator.eval()

        all_preds = np.array([])
        all_label_ids = np.array([])
        eval_loss = 0
        nb_eval_steps = 0
        for src_input_ids, src_input_mask, label_ids in val_dataloader:
            src_input_ids = src_input_ids.to(device)
            src_input_mask = src_input_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                _, doc_rep = bert(src_input_ids, attention_mask=src_input_mask)
                _, logits, probs = discriminator(doc_rep)
                print(probs)
                probs = torch.nn.functional.softmax(probs[:,:-1], dim=-1)
                tmp_eval_loss = nll_loss(probs, label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()

            logits = logits[:,:-1]
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            all_preds = np.append(all_preds, np.argmax(logits, axis=1))
            all_label_ids = np.append(all_label_ids, label_ids)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="micro", labels=list(range(0,len(label_list))))
        mcc = matthews_corrcoef(all_preds, all_label_ids)
        result = {"gen_loss": tr_g_loss,
                  "dis_loss": tr_d_loss,
                  "eval_loss": eval_loss,
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
            model_to_save = bert.module if hasattr(bert, 'module') else bert  # To handle multi gpu
            torch.save(model_to_save.state_dict(), bert_output_file)
            torch.save(discriminator.state_dict(), dis_output_file)

        logger.info("***** Epoch " + str(epoch_num + 1) + " *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %.4f", key, result[key])

        bert.train() # back to training
        discriminator.train()


test_examples = get_examples(test_filename)
test_dataloader = DataLoader(dataset=DomainData(test_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size)

bert = BertModel.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
discriminator = Discriminator(input_size=768, hidden_sizes=[768], num_labels=2, dropout_rate=0.1)

bert.load_state_dict(torch.load(bert_output_file))
bert.to(device)
if multi_gpu:
    bert = torch.nn.DataParallel(bert, device_ids=device_ids)

discriminator.load_state_dict(torch.load(dis_output_file))
discriminator.to(device)

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
    all_preds = np.append(all_preds, np.argmax(logits, axis=-1))
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
