import torch
from torch.utils.data import Dataset
import json
import numpy as np

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

def get_examples(filename, train=False, label_rate=0.02):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    examples = []
    for (i, line) in enumerate(lines):
        guid = i
        line = json.loads(line)
        text = str(line["text"])
        label = str(int(line["label"]))
        if train:
            if np.random.rand() > label_rate:
                label = "-1" # make the sample unlabeled
            else:
                # Balance out the labeled data
                repeat_num = int(1/label_rate)
                repeat_num = int(np.log2(repeat_num)) + 1
                if repeat_num < 0:
                    repeat_num = 0

                for _ in range(0, repeat_num):
                    examples.append((text, label))

        examples.append((text, label))

    return examples

def get_examples_qc_fine(filename, train=False, label_rate=0.02):
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
                repeat_num = int(np.log2(repeat_num)) - 1
                if repeat_num < 0:
                    repeat_num = 0

                for _ in range(0, repeat_num):
                    examples.append((text_a, label))

            examples.append((text_a, label))
        f.close()

    return examples
