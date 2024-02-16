import os
import sys
import csv
import math
import json
import time
import random
import requests
import regex as re
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import Dataset
from torch.utils.data.dataloader import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### -------Utils-------#####


def set_seed(seed):
    """
    Set the random seeds for reproducibility across different runs.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for PyTorch CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch CUDA (if available)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """
    Set up logging for the experiment.

    This function creates necessary directories and logs configurations.

    Args:
        config (object): Configuration object containing experiment settings.

    """
    # Get the working directory from the configuration
    work_dir = config.system.work_dir
    # Create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)

    # Log the command-line arguments (if any) used for running the script
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    # Log the experiment configuration itself in JSON format
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """ 
    A lightweight configuration class inspired by YACS.

    This class provides a flexible way to manage configurations for experiments or applications.
    """

    def __init__(self, **kwargs):
        """
        Initialize the configuration node with given key-value pairs.

        Args:
            **kwargs: Key-value pairs representing configuration options.
        """
        self.__dict__.update(kwargs)

    def __str__(self):
        """
        Return a string representation of the configuration node.
        """
        return self._str_helper(0)

    def _str_helper(self, indent):
        """
        A helper method to support nested indentation for pretty printing.

        Args:
            indent (int): The level of indentation.

        Returns:
            str: A string representation of the configuration node with proper indentation.
        """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """
        Return a dictionary representation of the configuration node.

        Returns:
            dict: A dictionary containing the configuration options.
        """
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        """
        Update the configuration node with values from a dictionary.

        Args:
            d (dict): A dictionary containing configuration options to merge.
        """
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        Update the configuration node from a list of strings, typically from the command line.

        Args:
            args (list): A list of strings representing configuration overrides in the format `--arg=value`.
                         Example: ['--model.n_layer=10', '--trainer.batch_size=32']
        """
        for arg in args:
            keyval = arg.split('=')
            assert len(
                keyval) == 2, "Expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval

            try:
                val = literal_eval(val)
            except ValueError:
                pass

            assert key[:2] == '--'
            key = key[2:]
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            assert hasattr(
                obj, leaf_key), f"{key} is not an attribute that exists in the config"

            print("Command line overwriting config attribute %s with %s" %
                  (key, val))
            setattr(obj, leaf_key, val)


def load_text(path):
    """
    Load text data from a file.

    Args:
        path (str): The path to the text file.

    Returns:
        str or None: The content of the text file as a string if successful, otherwise None.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file '{path}': {e}")
        return None


def load_json_to_text(path):
    """
    Loads JSON data from the specified file path and converts it into a single text string.

    Parameters:
        path (str): The file path to the JSON file.

    Returns:
        str: A single text string containing the JSON data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an error occurs while reading the JSON file or converting it to text.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            # Convert JSON data to a single string
            text = json.dumps(json_data)
        return text
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the JSON file '{path}': {e}")
        return None


def load_csv_to_text(path):
    """
    Loads data from a CSV file and converts it into a single text string.

    Parameters:
        path (str): The file path to the CSV file.

    Returns:
        str: A single text string containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an error occurs while reading the CSV file or converting it to text.
    """
    try:
        with open(path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            # Read all rows from the CSV file
            rows = [row for row in csv_reader]
            # Flatten the list of rows into a single list of strings
            text = '\n'.join(','.join(row) for row in rows)
        return text
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file '{path}': {e}")
        return None


def save(path, model):
    """
    Save the model state dictionary to a file.

    Args:
        path (str): Path to save the model state dictionary.
        model (torch.nn.Module): Model to save.
    """
    torch.save(model.state_dict(), path)


def load(path, model, device=None):
    """
    Load the model state dictionary from a file.

    Args:
        path (str): Path to the saved model state dictionary.
        model (torch.nn.Module): Model to load the state dictionary into.
        device (str, optional): Device to load the model onto (e.g., 'cuda' or 'cpu').
                                If None, it will automatically detect CUDA availability.

    Returns:
        None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))


def Generate(model, prompt, encoder_path, max_len):
    """
    Generate text based on a given prompt using the provided model.

    Args:
        model (torch.nn.Module): The language model used for text generation.
        prompt (str): The prompt to start text generation.
        encoder_path (str): Path to the encoder file for encoding/decoding characters.
        max_len (int): Maximum length of the generated text.

    Returns:
        str: Generated text based on the prompt.
    """
    # Initialize the Encoder with the provided encoder path
    enc = Encoder(encoder_path)

    # Initialize the context tensor for generation
    context = torch.zeros((1, 1), dtype=torch.long, device='cuda')

    # Move the model to the GPU
    model.to('cuda')

    # Encode the prompt using the Encoder and convert it to a tensor
    prompt_encoded = torch.tensor(enc.encode(
        prompt), dtype=torch.long, device='cuda')
    prompt_encoded = prompt_encoded.unsqueeze(0)

    # Concatenate the context tensor with the encoded prompt
    context_with_prompt = torch.cat(
        (context, prompt_encoded), dim=1).to('cuda')

    # Generate text based on the context with prompt and the maximum length
    res = model.generate(context_with_prompt, max_len)

    # Decode the generated text and return it as a string
    return enc.decode(res[0].tolist())


def get_gpt2_vocab_size():
    """Returns the vocab_size of openai's gpt-2 model."""
    return 50257


def print_config():
    """
    Function to print the configurations of various GPT models.
    Each model is represented by a dictionary containing its configuration parameters.

    Models and their configurations:
    - openai-gpt:   12 layers, 12 attention heads, 768-dimensional embeddings
    - gpt2:         12 layers, 12 attention heads, 768-dimensional embeddings
    - gpt2-medium:  24 layers, 16 attention heads, 1024-dimensional embeddings
    - gpt2-large:   36 layers, 20 attention heads, 1280-dimensional embeddings
    - gpt2-xl:      48 layers, 25 attention heads, 1600-dimensional embeddings
    - gopher-44m:   8 layers, 16 attention heads, 512-dimensional embeddings
    - gpt-mini:     6 layers, 6 attention heads, 192-dimensional embeddings
    - gpt-micro:    4 layers, 4 attention heads, 128-dimensional embeddings
    - gpt-nano:     3 layers, 3 attention heads, 48-dimensional embeddings
    """
    models_config = {
        'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
        'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
        'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
        'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
    }

    # Iterate through each model and print its configuration
    for model, config in models_config.items():
        print(f"Model: {model}")
        print(f"\tNumber of layers: {config['n_layer']}")
        print(f"\tNumber of attention heads: {config['n_head']}")
        print(f"\tEmbedding size: {config['n_embd']}")
        print()
##### -------Utils-------#####

##### -------Model-------#####


CN = CfgNode


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function.

    This implementation is identical to the GELU function used in Google BERT and OpenAI GPT models.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        """
        Forward pass of the GELU activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the GELU activation function.
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.

    This class implements a causal self-attention mechanism, where attention is restricted to the left in the input sequence.
    """

    def __init__(self, config):
        """
        Initialize the CausalSelfAttention layer.

        Args:
            config (CfgNode): Configuration for the attention layer.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure attention is applied only to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        Forward pass of the CausalSelfAttention layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the causal self-attention mechanism.
        """
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """ 
    An unassuming Transformer block. 

    This class defines a basic Transformer block, consisting of a causal self-attention layer 
    followed by a feed-forward neural network with layer normalization and residual connections.
    """

    def __init__(self, config):
        """
        Initialize the Block.

        Args:
            config (CfgNode): Configuration for the Transformer block.
        """
        super().__init__()

        # Layer normalization for the input to the self-attention layer
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Causal self-attention layer
        self.attn = CausalSelfAttention(config)

        # Layer normalization for the output from the self-attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Feed-forward neural network (MLP) with layer normalization and dropout
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))

        # Define the forward pass of the MLP
        m = self.mlp
        self.mlpf = lambda x: m.dropout(
            m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        """
        Forward pass of the Block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the Transformer block.
        """
        # Apply self-attention, layer normalization, and residual connection
        x = x + self.attn(self.ln_1(x))

        # Apply feed-forward neural network, layer normalization, and residual connection
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """ 
    GPT Language Model.

    This class defines the GPT language model, including its architecture and functionality for training and generation.
    """

    @staticmethod
    def get_default_config():
        """
        Get the default configuration for the GPT model.

        Returns:
            CfgNode: Default configuration for the GPT model.
        """
        C = CN()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        """
        Initialize the GPT model.

        Args:
            config (CfgNode): Configuration for the GPT model.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None,
                           config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given  # exactly one of these (XOR)

        # Translate from model_type to detailed configuration
        if type_given:
            config.merge_from_dict({
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        # Define the transformer layers
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        """
        Initialize weights for the module.

        Args:
            module (nn.Module): The module for weight initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        Configure the optimizer for training the model.

        Args:
            train_config: Training configuration.

        Returns:
            torch.optim.Optimizer: PyTorch optimizer object.
        """
        # Separate parameters into those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate that all parameters are considered
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "Parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "Parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        # Create the PyTorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        Args:
            idx (torch.Tensor): The input tensor.
            targets (torch.Tensor): The target tensor for training.

        Returns:
            torch.Tensor: The logits from the model.
            torch.Tensor: The loss if targets are provided, otherwise None.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long,
                           device=device).unsqueeze(0)  # shape (1, t)

        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        # Position embeddings
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Generate new tokens given a conditioning sequence.

        Args:
            idx (torch.Tensor): Conditioning sequence of indices.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Temperature parameter for sampling.
            do_sample (bool): Whether to sample from the distribution or take the most likely token.
            top_k (int): Number of top-k tokens to consider.

        Returns:
            torch.Tensor: Generated sequence of indices.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


##### -------Model-------#####

##### --------Dataset--------#####

class CharDataset(Dataset):
    """
    Dataset class for character-level text data.
    """

    def __init__(self, data, block_size, output_dir=None):
        """
        Initialize the CharDataset.

        Args:
            data (str): Text data.
            block_size (int): Size of the text blocks.
            output_dir (str, optional): Output directory to save encoder.json.
        """
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.data = data

        if output_dir:
            encoder_file = os.path.join(output_dir, 'encoder.json')
            with open(encoder_file, 'w') as f:
                json.dump(self.stoi, f)

    def get_vocab_size(self):
        """
        Get the vocabulary size.

        Returns:
            int: Vocabulary size.
        """
        return self.vocab_size

    def get_block_size(self):
        """
        Get the block size.

        Returns:
            int: Block size.
        """
        return self.block_size

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing input and target tensors.
        """
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def _encode(self, s):
        """
        Encode a string into a list of integers.

        Args:
            s (str): Input string.

        Returns:
            list: List of encoded integers.
        """
        return [self.stoi[c] for c in s if c in self.stoi]

    def _decode(self, l):
        """
        Decode a list of integers into a string.

        Args:
            l (list): List of integers.

        Returns:
            str: Decoded string.
        """
        return ''.join([self.itos.get(i, '') for i in l])


class Encoder:
    """
    Class for encoding and decoding text using an encoder dictionary.
    """

    def __init__(self, encoder_path):
        """
        Initialize the Encoder.

        Args:
            encoder_path (str): Path to the encoder JSON file.
        """
        with open(encoder_path, 'r') as f:
            self.stoi = json.load(f)
        self.itos = {v: k for k, v in self.stoi.items()}

    def encode(self, s):
        """
        Encode a string into a list of integers.

        Args:
            s (str): Input string.

        Returns:
            list: List of encoded integers.
        """
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        """
        Decode a list of integers into a string.

        Args:
            l (list): List of integers.

        Returns:
            str: Decoded string.
        """
        return ''.join([self.itos.get(i, '') for i in l])


class FineEncoderDataset(Dataset):
    def __init__(self, data, encoder_path, block_size):
        """
        Initialize the dataset with the provided data, encoder path, and block size.

        Args:
            data (str): Input data for the dataset.
            encoder_path (str): Path to the encoder file.
            block_size (int): Size of each block in the dataset.
        """
        self.data = data
        with open(encoder_path, 'r') as f:
            self.stoi = json.load(f)
        self.itos = {v: k for k, v in self.stoi.items()}
        self.block_size = block_size

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data) - self.block_size

    def encode(self, s):
        """
        Encode the input string using the encoder.

        Args:
            s (str): Input string to encode.

        Returns:
            list: Encoded representation of the input string.
        """
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        """
        Decode the input list using the encoder.

        Args:
            l (list): Input list to decode.

        Returns:
            str: Decoded representation of the input list.
        """
        return ''.join([self.itos.get(i, '') for i in l])

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the input and target tensors.
        """
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(self.encode(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(self.encode(chunk[1:]), dtype=torch.long)
        return x, y

##### --------Dataset--------#####

##### -------train-------#####


class Trainer:
    """
    Trainer class for training neural networks.
    """

    @staticmethod
    def get_default_config():
        """
        Get the default configuration for the Trainer.

        Returns:
            CfgNode: Default configuration for the Trainer.
        """
        C = CN()
        C.device = 'auto'  # Device to train on
        C.num_workers = 4  # Number of workers for data loading
        C.max_iters = None  # Maximum number of iterations
        C.batch_size = 64  # Batch size
        C.learning_rate = 3e-4  # Learning rate
        C.betas = (0.9, 0.95)  # Betas for the optimizer
        C.weight_decay = 0.1  # Weight decay applied only on matmul weights
        C.grad_norm_clip = 1.0  # Gradient norm clipping
        return C

    def __init__(self, config, model, train_dataset):
        """
        Initialize the Trainer.

        Args:
            config (CfgNode): Configuration for the Trainer.
            model (nn.Module): The neural network model.
            train_dataset: The training dataset.
        """
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        """
        Add a callback function to be triggered on an event.

        Args:
            onevent (str): Event to trigger the callback.
            callback: Callback function to be added.
        """
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        """
        Set a callback function to be triggered on an event.

        Args:
            onevent (str): Event to trigger the callback.
            callback: Callback function to be set.
        """
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        """
        Trigger the callbacks for a specific event.

        Args:
            onevent (str): Event to trigger the callbacks.
        """
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self, num_epochs=None):
        """
        Run the training loop.

        Args:
            num_epochs (int, optional): Number of epochs to train for. If None, train indefinitely.
        """
        model, config = self.model, self.config

        self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        total_iterations = config.max_iters * (num_epochs if num_epochs else 1)
        iter_per_epoch = config.max_iters

        for epoch in range(num_epochs if num_epochs else 1):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            data_iter = iter(train_loader)
            with tqdm(total=iter_per_epoch, desc=f'Epoch {epoch + 1}/{num_epochs}' if num_epochs else f'Epoch {epoch + 1}', unit='batch') as pbar:
                for _ in range(iter_per_epoch):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        batch = next(data_iter)
                    batch = [t.to(self.device) for t in batch]
                    x, y = batch

                    logits, self.loss = model(x, y)

                    model.zero_grad(set_to_none=True)
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip)
                    self.optimizer.step()

                    self.trigger_callbacks('on_batch_end')
                    self.iter_num += 1
                    tnow = time.time()
                    self.iter_dt = tnow - self.iter_time
                    self.iter_time = tnow

                    epoch_loss += self.loss.item()
                    pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})
                    pbar.update(1)

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs if num_epochs else 1} completed in {epoch_time} seconds. Loss: {epoch_loss / iter_per_epoch}")


class FineTuner:
    def __init__(self, model, data, encoder_path, block_size, train_config):
        """
        Initialize the FineTuner with the provided model, data, encoder path, block size, and training configuration.

        Args:
            model (torch.nn.Module): Model to fine-tune.
            data (str): Input data for fine-tuning.
            encoder_path (str): Path to the encoder file.
            block_size (int): Size of each block in the dataset.
            train_config (CfgNode): Training configuration for the fine-tuning process.
        """
        self.model = model
        self.data = data
        self.encoder_path = encoder_path
        self.block_size = block_size
        self.train_config = train_config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def fine_tune(self, epoch=1):
        """
        Perform fine-tuning of the model.

        Args:
            epoch (int, optional): Number of epochs for fine-tuning. Default is 1.
        """
        # Create dataset for fine-tuning
        fine_dataset = FineEncoderDataset(
            self.data, self.encoder_path, self.block_size)

        # Initialize Trainer with the provided training configuration, model, and dataset
        trainer = Trainer(self.train_config, self.model, fine_dataset)

        # Run the training process for the specified number of epochs
        trainer.run(epoch)

##### -------train-------#####
