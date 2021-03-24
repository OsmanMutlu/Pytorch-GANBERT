import sys
import os
import logging
import random
import datetime
from tqdm import tqdm, trange
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

from model import Generator1, Discriminator, get_weights_from_tf

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import DomainData, get_examples_qc_fine, get_examples

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

repo_path = sys.argv[1]
task_name = sys.argv[2] # "qc_fine", "amazon", "clef"
tokenizer = BertTokenizer.from_pretrained(repo_path + "/../.pytorch_pretrained_bert/bert-base-uncased-vocab.txt")
bert_model = repo_path + "/../.pytorch_pretrained_bert/bert-base-uncased.tar.gz"
nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
adversarial_loss = torch.nn.BCELoss()

learning_rate = 2e-5
noise_size = 100
epsilon = 1e-8
label_rate = 0.02
total_epoch_num = 30
warmup_proportion = 0.1


if task_name == "qc_fine":
    max_seq_length = 64
    batch_size = 64
    validation = False
    train_filename = repo_path + "/../domain_adaptation/doc_gan/ganbert_tf/data/labeled_and_unlabeled.tsv"
    test_filename = repo_path + "/../domain_adaptation/doc_gan/ganbert_tf/data/test.tsv"
    bert_output_file = repo_path + "/../domain_adaptation/models/doc_gan/qc-fine_gan-bert_64_64_2e-5_30_bert.pt"
    dis_output_file = repo_path + "/../domain_adaptation/models/doc_gan/qc-fine_gan-bert_64_64_2e-5_30_dis.pt"
    label_list = ["ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc", "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body", "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed", "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang", "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product", "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol", "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc", "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country", "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count", "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other", "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize", "NUM_weight"]
    get_examples_fct = get_examples_qc_fine

elif task_name == "amazon":
    max_seq_length = 512
    batch_size = 32
    validation = True
    train_filename = "/../domain_adaptation/data/sorted_data/books/train.json"
    val_filename = "/../domain_adaptation/data/sorted_data/books/val.json"
    test_filename = "/../domain_adaptation/data/sorted_data/electronics/test.json"
    bert_output_file = repo_path + "/../domain_adaptation/models/doc_gan/books_gan-bert_512_32_2e-5_0,02_bert.pt"
    dis_output_file = repo_path + "/../domain_adaptation/models/doc_gan/books_gan-bert_512_32_2e-5_0,02_dis.pt"
    label_list = ["0", "1"]
    get_examples_fct = get_examples

elif task_name == "clef":
    max_seq_length = 512
    batch_size = 32
    validation = True
    train_filename = repo_path + "/../domain_adaptation/data/domain_corpus_data/clef/balanced_india_train.json"
    val_filename = repo_path + "/../domain_adaptation/data/domain_corpus_data/clef/india_dev.json"
    test_filename = repo_path + "/../domain_adaptation/data/domain_corpus_data/clef/india_test.json"
    bert_output_file = repo_path + "/../domain_adaptation/models/doc_gan/asd_india_gan-bert_512_32_2e-5_0,02_bert.pt"
    dis_output_file = repo_path + "/../domain_adaptation/models/doc_gan/asd_india_gan-bert_512_32_2e-5_0,02_dis.pt"
    # bert_output_file = repo_path + "/../domain_adaptation/models/doc_gan/india_gan-bert_512_32_2e-5_0,02_bert.pt"
    # dis_output_file = repo_path + "/../domain_adaptation/models/doc_gan/india_gan-bert_512_32_2e-5_0,02_dis.pt"
    label_list = ["0", "1"]
    get_examples_fct = get_examples

else:
    raise "Task name not found!"

tf_weights_loadable_version = True
load_tf_weights = False
train = True
multi_gpu = True
device = torch.device("cuda:3")
# device = "cpu"
device_ids = [3,4,5,6]
dev_metric = "f1"
seed = 129

if load_tf_weights:
    if not tf_weights_loadable_version:
        raise "Please use the tf_weights_loadable_version of the model"
    if train:
        raise "Why are you training if you are loading already trained weights?"

# ******** IMPORTANT NOTE: ********
# Original implementation of GAN-BERT has an extra class that is never used.
# For examples, if you originally have positive and negative classes for your task:
# You have some unlabeled examples too, let's call them "UNK" or "-1".
# Original implementation adds another class to your discriminator that is never used.
# So, in order to correctly transfer the weights from the original version,
# we just add an imaginary extra label to label_list and model. There is no sample
# labeled with this, so there is no update to its corresponding weights, so it
# is never used. Note that this may hurt the model if you are training from scratch.
if tf_weights_loadable_version:
    label_list.insert(0, "AnythingCanGoHere!")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Hyper-params
print("Maximum sequence length : %d" %max_seq_length)
print("Batch size : %d" %batch_size)
print("Learning rate : %.8f" %learning_rate)

if train:
    # Get examples
    train_examples = get_examples_fct(train_filename, train=True, label_rate=label_rate)
    random.shuffle(train_examples)
    train_dataloader = DataLoader(dataset=DomainData(train_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size, shuffle=True, drop_last=False)

    if validation:
        val_examples = get_examples_fct(val_filename, label_rate=label_rate)
        val_dataloader = DataLoader(dataset=DomainData(val_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size, shuffle=False, drop_last=False)

    num_train_steps = int(len(train_examples) / batch_size * total_epoch_num)

    # Create model
    bert = BertModel.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
    generator = Generator1(noise_size=noise_size, output_size=768, hidden_sizes=[768], dropout_rate=0.1)
    discriminator = Discriminator(input_size=768, hidden_sizes=[768], num_labels=len(label_list), dropout_rate=0.1)

    bert.to(device)
    if multi_gpu:
        bert = torch.nn.DataParallel(bert, device_ids=device_ids)

    generator.to(device)
    discriminator.to(device)

    # param_optimizer = list(bert.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    #     ]
    # bert_optimizer = BertAdam(optimizer_grouped_parameters,
    #                           lr=bert_learning_rate,
    #                           warmup=warmup_proportion,
    #                           t_total=num_train_steps)

    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    dis_optimizer = torch.optim.AdamW(list(bert.parameters()) + list(discriminator.parameters()), lr=learning_rate)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    # Start training
    global_step = 0
    best_loss = 1000.0
    best_score = -1.0
    all_scores = []
    bert.train()
    generator.train()
    discriminator.train()
    for epoch_num in trange(int(total_epoch_num), desc="Epoch"):

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

            # Real representations
            _, doc_rep = bert(src_input_ids, attention_mask=src_input_mask, output_all_encoded_layers=False)
            D_real_features, D_real_logits, D_real_probs = discriminator(doc_rep)

            # Random noise
            noise = torch.zeros(src_input_ids.shape[0],noise_size, device=device).uniform_(0, 1)#.requires_grad_(True)
            # Generated representations
            gen_rep = generator(noise)

            ############
            # Discriminator loss and step
            ############
            if tf_weights_loadable_version:
                label_probs = torch.nn.functional.softmax(D_real_probs[:,1:], dim=-1)
                real_prob = D_real_probs[:,0]
            else:
                label_probs = torch.nn.functional.softmax(D_real_probs[:,:-1], dim=-1)
                real_prob = D_real_probs[:,-1]

            d_label_loss = nll_loss(label_probs, label_ids.view(-1))
            d_gan_real_loss = adversarial_loss(real_prob, valid)
            d_real_loss = d_label_loss + d_gan_real_loss

            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep.detach()) # for discriminator
            if tf_weights_loadable_version:
                fake_prob = D_fake_probs[:,0]
            else:
                fake_prob = D_fake_probs[:,-1]

            d_gan_fake_loss = adversarial_loss(fake_prob, fake)
            # d_gan_fake_loss = -torch.mean(torch.log(D_fake_probs[:,0] + epsilon))

            d_loss = d_real_loss + d_gan_fake_loss
            d_loss.backward()
            dis_optimizer.step()

            ############
            # Generator loss and step
            ############
            generator.zero_grad()
            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep) # for generator

            if tf_weights_loadable_version:
                fake_prob = D_fake_probs[:,0]
            else:
                fake_prob = D_fake_probs[:,-1]

            g_loss = adversarial_loss(fake_prob, valid)
            g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features.detach(), dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss += g_feat_reg

            g_loss.backward()
            gen_optimizer.step()

            tr_g_loss += g_loss.mean().item()
            tr_d_loss += d_loss.mean().item()
            nb_tr_examples += src_input_ids.size(0)
            nb_tr_steps += 1
            global_step += 1

        tr_g_loss /= nb_tr_steps
        tr_d_loss /= nb_tr_steps

        if validation:
            # VALIDATION
            bert.eval()
            discriminator.eval()

            all_preds = np.array([])
            all_label_ids = np.array([])
            eval_loss = 0
            nb_eval_steps = 0
            for val_step, (src_input_ids, src_input_mask, label_ids) in enumerate(val_dataloader):
                src_input_ids = src_input_ids.to(device)
                src_input_mask = src_input_mask.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    _, doc_rep = bert(src_input_ids, attention_mask=src_input_mask)
                    _, _, probs = discriminator(doc_rep)

                    if tf_weights_loadable_version:
                        probs = torch.nn.functional.softmax(probs[:,1:], dim=-1)
                    else:
                        probs = torch.nn.functional.softmax(probs[:,:-1], dim=-1)

                    tmp_eval_loss = nll_loss(probs, label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()

                probs = probs.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                all_preds = np.append(all_preds, np.argmax(probs, axis=1))
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
            elif dev_metric == "mcc":
                score = mcc

            all_scores.append(score)
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

        else:
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

    if validation:
        print("All scores are: ")
        print(all_scores)


test_examples = get_examples_fct(test_filename, label_rate=label_rate)
test_dataloader = DataLoader(dataset=DomainData(test_examples, label_list, max_seq_length, tokenizer), batch_size=batch_size)

bert = BertModel.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
discriminator = Discriminator(input_size=768, hidden_sizes=[768], num_labels=len(label_list), dropout_rate=0.1)

if load_tf_weights:
    bert = get_weights_from_tf(bert, repo_path + "/../domain_adaptation/doc_gan/ganbert_tf/ganbert_output_model/", model_name="bert")
    discriminator = get_weights_from_tf(discriminator, repo_path + "/../domain_adaptation/doc_gan/ganbert_tf/ganbert_output_model/", model_name="dis")
    # generator = get_weights_from_tf(generator, repo_path + "/../domain_adaptation/doc_gan/ganbert_tf/ganbert_output_model/", model_name="gen")

else:
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
        _, _, probs = discriminator(doc_rep)
        if tf_weights_loadable_version:
            probs = torch.nn.functional.softmax(probs[:,1:], dim=-1)
        else:
            probs = torch.nn.functional.softmax(probs[:,:-1], dim=-1)

        tmp_test_loss = nll_loss(probs, label_ids.view(-1))

    test_loss += tmp_test_loss.mean().item()

    probs = probs.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    all_preds = np.append(all_preds, np.argmax(probs, axis=1))
    all_label_ids = np.append(all_label_ids, label_ids)

    nb_test_steps += 1

# print(all_preds)
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
