from torch import nn
from torch import tensor as torch_tensor
from tensorflow.train import load_checkpoint
import re

# NOTE: Produces a document representation directly from noise. Since we do not use a source sample here in any way, this method can't really be used for DA. This is purely to see if we can train a GAN.
# Same as GAN-BERT paper
class Generator1(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator1, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise) # generated document representation
        return output_rep

class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs

def convert_to_tf_param_name(torch_param_name, model_name="bert"):
    tf_param_name = ""
    if model_name == "bert":
        tf_param_name = tf_param_name + "bert/"
        if "embeddings" in torch_param_name:
            if ".weight" in torch_param_name:
                torch_param_name = torch_param_name[:-7]

            tf_param_name = tf_param_name + torch_param_name.replace(".", "/")

        elif "encoder" in torch_param_name:
            torch_param_name = re.sub("layer\.([0-9]+)\.", "layer_\g<1>.", torch_param_name)
            torch_param_name = torch_param_name.replace("weight", "kernel").replace(".", "/")
            tf_param_name = tf_param_name + torch_param_name

        else: # pooler
            torch_param_name = torch_param_name.replace("weight", "kernel").replace(".", "/")
            tf_param_name = tf_param_name + torch_param_name

    elif model_name == "dis": # discriminator
        tf_param_name = tf_param_name + "Discriminator/"
        torch_param_name = torch_param_name.replace("logit", "dense_1").replace("layers.0", "dense")
        torch_param_name = torch_param_name.replace("weight", "kernel").replace(".", "/")
        tf_param_name = tf_param_name + torch_param_name

    elif model_name == "gen": # generator
        tf_param_name = tf_param_name + "Generator/"
        torch_param_name = torch_param_name.replace("layers.3", "dense_1").replace("layers.0", "dense")
        torch_param_name = torch_param_name.replace("weight", "kernel").replace(".", "/")
        tf_param_name = tf_param_name + torch_param_name

    else:
        raise "Wrong model_name!"

    return tf_param_name

def get_weights_from_tf(model, tf_model_path, model_name="bert"):
    reader = load_checkpoint(tf_model_path)
    weight_names = sorted(reader.get_variable_to_shape_map().keys())
    for name, param in model.named_parameters():
        curr_tf_param = convert_to_tf_param_name(name, model_name=model_name)
        curr_tf_weight = reader.get_tensor(curr_tf_param)
        if "kernel" in curr_tf_param:
            curr_tf_weight = curr_tf_weight.transpose()

        param.data = torch_tensor(curr_tf_weight)

    return model
