import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm


class LanguageDataset:
    """
    A class representing a language dataset for the Bigram Language Model.

    Attributes:
        path (str): The file path to the text data.
        train_val_split (int): The percentage split between training and validation data.

    Methods:
        _load_dataset: Load the text data from the given file path.
        _encode: Convert the text data into a list of numerical representations using character encoding.
        _decode: Convert a list of numerical representations back into the original text.
        _calculate_split: Calculate the split index for separating training and validation data.
    """

    def __init__(self, path, train_val_split=90):
        """
        Initialize the LanguageDataset with the provided text data.

        Args:
            path (str): The file path to the text data.
            train_val_split (int, optional): The percentage split between training and validation data.
                                            Defaults to 90 (90% training, 10% validation).
        """
        self.text = self._load_dataset(path)
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor(self._encode(self.text), dtype=torch.long)
        partition = self._calculate_split(train_val_split)
        n = int(partition * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def _load_dataset(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def _encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def _decode(self, l):
        return ''.join([self.itos.get(i, '') for i in l])

    def _calculate_split(self, train_val_split):
        if not (0 < train_val_split <= 100):
            raise ValueError(
                "train_val_split should be between 1 and 100 (inclusive).")
        else:
            return train_val_split / 100


def count_parameters(model):
    """
    Count the total number of trainable parameters in the given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for which the parameters need to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def get_batch(split, train_data, val_data, block_size=None, batch_size=None, device=None):
    """
    Get a batch of data for training or validation from the specified split.

    Args:
        split (str): The split to get the data from ('train' or 'val').
        train_data (torch.Tensor): The training data tensor.
        val_data (torch.Tensor): The validation data tensor.
        block_size (int, optional): The size of each input sequence block. Defaults to None.
        batch_size (int, optional): The number of sequences in a batch. Defaults to None.
        device (str, optional): The device to store the batch data on ('cpu' or 'cuda'). Defaults to None.

    Returns:
        torch.Tensor: The input tensor representing the batch of sequences (x).
        torch.Tensor: The target tensor representing the batch of sequences shifted by one (y).
    """
    set_hyperparams(block_size=block_size,
                    batch_size=batch_size, device=device)
    block_size = hyperparams['block_size']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=None, block_size=None, batch_size=None, device=None):
    """
    Estimate the average loss of the model on the training and validation data.

    Args:
        model (torch.nn.Module): The PyTorch model for which the loss is to be estimated.
        train_data (torch.Tensor): The training data tensor.
        val_data (torch.Tensor): The validation data tensor.
        eval_iters (int, optional): The number of iterations for estimating the loss. Defaults to None.
        block_size (int, optional): The size of each input sequence block. Defaults to None.
        batch_size (int, optional): The number of sequences in a batch. Defaults to None.
        device (str, optional): The device to perform computations on ('cpu' or 'cuda'). Defaults to None.

    Returns:
        dict: A dictionary containing the average loss on the training and validation data.
            Keys: 'train' and 'val'
            Values: Average loss values (float)
    """
    set_hyperparams(eval_iters=eval_iters, block_size=block_size,
                    batch_size=batch_size, device=device)
    eval_iters = hyperparams['eval_iters']
    block_size = hyperparams['block_size']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data,
                             block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module used in the Transformer block.

    Args:
        num_heads (int): The number of attention heads.
        head_size (int): The size of each attention head.
        n_embd (int): The embedding dimension of the input.
        block_size (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.

    Methods:
        forward: Perform the forward pass of the multi-head self-attention module.

    Attributes:
        heads (nn.ModuleList): List of individual attention heads (Head modules).
        proj (nn.Linear): Linear layer for projecting concatenated attention heads to the embedding dimension.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        """
        Initialize the MultiHeadAttention with attention heads and projection layer.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            n_embd (int): The embedding dimension of the input.
            block_size (int): The maximum length of the input sequence.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform the forward pass of the multi-head self-attention module.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C), where B is the batch size,
                            T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: The output tensor after applying the multi-head self-attention.
                        It has the same shape as the input tensor (B, T, C).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Head(nn.Module):
    """
    A single head of self-attention used in the Transformer block.

    Args:
        head_size (int): The size of the attention head.
        n_embd (int): The embedding dimension of the input.
        block_size (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.

    Methods:
        forward: Perform the forward pass of the attention head.

    Attributes:
        key (nn.Linear): Linear layer for computing the keys.
        query (nn.Linear): Linear layer for computing the queries.
        value (nn.Linear): Linear layer for computing the values.
        tril (torch.Tensor): Lower triangular mask to prevent attending to future tokens.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        """
        Initialize the Head with linear layers and a lower triangular mask.

        Args:
            head_size (int): The size of the attention head.
            n_embd (int): The embedding dimension of the input.
            block_size (int): The maximum length of the input sequence.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform the forward pass of the attention head.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C), where B is the batch size,
                            T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: The output tensor after the self-attention operation.
                        It has the same shape as the input tensor (B, T, C).
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1)*C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class FeedForward(nn.Module):
    """
    Feed-forward neural network module used in the Transformer block.

    Args:
        n_embd (int): The embedding dimension of the input.
        dropout (float): The dropout rate.

    Methods:
        forward: Perform the forward pass of the feed-forward neural network.

    Attributes:
        net (nn.Sequential): The sequential neural network module consisting of linear layers and activation functions.
    """

    def __init__(self, n_embd, dropout):
        """
        Initialize the FeedForward with linear layers and activation functions.

        Args:
            n_embd (int): The embedding dimension of the input.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Perform the forward pass of the feed-forward neural network.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C), where B is the batch size,
                            T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: The output tensor after the feed-forward operation.
                        It has the same shape as the input tensor (B, T, C).
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block module used in the BigramLanguageModel.

    Args:
        n_embd (int): The embedding dimension of the input.
        n_head (int): The number of attention heads in the multi-head self-attention.
        block_size (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.

    Methods:
        forward: Perform the forward pass of the transformer block.

    Attributes:
        sa (MultiHeadAttention): Multi-head self-attention module.
        ffwd (FeedForward): Feed-forward neural network module.
        ln1 (nn.LayerNorm): Layer normalization for the first sub-layer.
        ln2 (nn.LayerNorm): Layer normalization for the second sub-layer.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initialize the Transformer block with self-attention and feed-forward modules.

        Args:
            n_embd (int): The embedding dimension of the input.
            n_head (int): The number of attention heads in the multi-head self-attention.
            block_size (int): The maximum length of the input sequence.
            dropout (float): The dropout rate.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Perform the forward pass of the transformer block.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C), where B is the batch size,
                            T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: The output tensor after applying the transformer block.
                        It has the same shape as the input tensor (B, T, C).
        """
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """
    Bigram Language Model based on the Transformer architecture.

    Args:
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens in the language.
        n_embd (int): The embedding dimension of the input. (Default: None)
        n_head (int): The number of attention heads in the multi-head self-attention. (Default: None)
        n_layer (int): The number of Transformer blocks in the model. (Default: None)
        block_size (int): The maximum length of the input sequence. (Default: None)
        dropout (float): The dropout rate for regularization. (Default: None)
        device (str or torch.device): The device on which the model will be initialized and run.
                                    If not provided, 'cuda' will be used if available, else 'cpu'.

    Methods:
        forward: Perform the forward pass of the BigramLanguageModel.
        generate: Generate new tokens using the trained language model.

    Attributes:
        token_embedding_table (nn.Embedding): Embedding layer for token embeddings.
        positional_embedding_table (nn.Embedding): Embedding layer for positional embeddings.
        blocks (nn.Sequential): Sequential container for multiple Transformer blocks (Block instances).
        ln_f (nn.LayerNorm): Layer normalization for the final output of the model.
        lm_head (nn.Linear): Linear layer to project the final output to the vocabulary size.
    """

    def __init__(self, vocab_size, n_embd=None, n_head=None, n_layer=None, block_size=None, dropout=None, device=None):
        """
        Initialize the BigramLanguageModel with the specified hyperparameters.

        Args:
            vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens in the language.
            n_embd (int): The embedding dimension of the input. (Default: None)
            n_head (int): The number of attention heads in the multi-head self-attention. (Default: None)
            n_layer (int): The number of Transformer blocks in the model. (Default: None)
            block_size (int): The maximum length of the input sequence. (Default: None)
            dropout (float): The dropout rate for regularization. (Default: None)
            device (str or torch.device): The device on which the model will be initialized and run.
                                        If not provided, 'cuda' will be used if available, else 'cpu'.
        """
        set_hyperparams(n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                        block_size=block_size, dropout=dropout, device=device)
        n_embd = hyperparams['n_embd']
        n_head = hyperparams['n_head']
        n_layer = hyperparams['n_layer']
        block_size = hyperparams['block_size']
        dropout = hyperparams['dropout']
        super(BigramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.positional_embedding_table = nn.Embedding(
            self.block_size, self.n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def _build_block(self):
        return Block(self.n_embd, self.n_head)

    def forward(self, idx, target=None):
        """
        Perform the forward pass of the BigramLanguageModel.

        Args:
            idx (torch.Tensor): The input tensor representing the token indices of shape (B, T), where B is the batch size,
                                and T is the sequence length.
            target (torch.Tensor): The target tensor representing the token indices for computing the loss.
                                It should have the same shape as idx. If None, no loss will be computed.
                                (Default: None)

        Returns:
            torch.Tensor: The logits tensor after the forward pass of the model. It has the shape (B, T, vocab_size).
            torch.Tensor: The loss tensor, computed using F.cross_entropy, if the target is provided. Otherwise, None.
                          It has the shape (B * T,) if target is not None.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(
            torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens using the trained language model.

        Args:
            idx (torch.Tensor): The input tensor representing the token indices of shape (B, T), where B is the batch size,
                                and T is the sequence length.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            list: A list of generated token indices as integers.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx[0].tolist()


# Create a dictionary to store the hyperparameters
hyperparams = {
    'batch_size': 64,                           # Batch size for training
    'block_size': 256,                         # Maximum length of the input sequence
    'max_iters': 5000,                         # Maximum number of training iterations
    'learning_rate': 5.859375e-05,             # Learning rate for the optimizer
    # Device to run the model (GPU if available, else CPU)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Number of iterations for evaluation during training
    'eval_iters': 200,
    # Interval for printing evaluation results during training
    'eval_interval': 500,
    'n_embd': 384,                             # Embedding dimension of the input
    # Number of attention heads in the multi-head self-attention
    'n_head': 6,
    'n_layer': 6,                              # Number of Transformer blocks in the model
    'dropout': 0.2,                            # Dropout rate for regularization
}

# Function to set the hyperparameters


def set_hyperparams(**kwargs):
    """
    Update the hyperparameters dictionary with the provided keyword arguments.

    Args:
        **kwargs: Keyword arguments with hyperparameter names as keys and their corresponding values.

    Note:
        This function updates the global hyperparams dictionary with the provided non-None values in the kwargs.
    """
    global hyperparams
    # Filter out any None values from kwargs and update the hyperparameters dictionary
    valid_kwargs = {key: value for key,
                    value in kwargs.items() if value is not None}
    hyperparams.update(valid_kwargs)


def load_dataset(path, train_val_split=90):
    """
    Load a dataset from the given file path and create a LanguageDataset object.

    Args:
        path (str): The file path of the dataset to load.
        train_val_split (int, optional): Percentage split between training and validation data.
                                        Default is 90, which means 90% training and 10% validation.

    Returns:
        LanguageDataset: An instance of the LanguageDataset class containing the loaded dataset.

    Note:
        The LanguageDataset class is used to preprocess and split the dataset into training and validation data.
        It also provides methods for encoding and decoding text data into numerical representations.
    """
    return LanguageDataset(path, train_val_split)

# Create and initialize the model


def initialize_model(vocab_size, n_embd=None, n_head=None, n_layer=None, block_size=None, dropout=None, device=None):
    """
    Initialize a BigramLanguageModel with the given hyperparameters.

    Args:
        vocab_size (int): The size of the vocabulary, which determines the number of tokens in the language model.
        n_embd (int, optional): The embedding dimension of the input. If not provided, it will be set to a default value.
        n_head (int, optional): The number of attention heads in the multi-head self-attention. If not provided,
                                it will be set to a default value.
        n_layer (int, optional): The number of Transformer blocks in the model. If not provided, it will be set to a default value.
        block_size (int, optional): The maximum length of the input sequence. If not provided, it will be set to a default value.
        dropout (float, optional): The dropout rate for regularization. If not provided, it will be set to a default value.
        device (str, optional): The device on which to place the model (e.g., 'cpu' or 'cuda'). If not provided, it will be set
                                based on the availability of a GPU.

    Returns:
        BigramLanguageModel: An instance of the BigramLanguageModel class initialized with the given hyperparameters.

    Note:
        The BigramLanguageModel is a variant of the GPT-2 architecture that uses a bigram language modeling objective.
        The function sets the hyperparameters using the provided values or default values from hyperparams dictionary,
        initializes the model with those hyperparameters, and returns the model ready for training or inference.
    """
    set_hyperparams(n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                    block_size=block_size, dropout=dropout, device=device)
    n_embd = hyperparams['n_embd']
    n_head = hyperparams['n_head']
    n_layer = hyperparams['n_layer']
    block_size = hyperparams['block_size']
    dropout = hyperparams['dropout']
    device = hyperparams['device']
    model = BigramLanguageModel(
        vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    return model.to(device)


def train(model, train_data, val_data, learning_rate=None, max_iters=None, eval_interval=None, device=None, eval_iters=None):
    """
    Train the provided model on the given training data and validate it on the validation data.

    Args:
        model (BigramLanguageModel): The language model to train.
        train_data (torch.Tensor): The training data tensor.
        val_data (torch.Tensor): The validation data tensor.
        learning_rate (float, optional): The learning rate for the optimizer. If not provided, it will be set to a default value.
        max_iters (int, optional): The maximum number of training iterations. If not provided, it will be set to a default value.
        eval_interval (int, optional): Interval for printing evaluation results during training.
                                    If not provided, it will be set to a default value.
        device (str, optional): The device on which to train the model (e.g., 'cpu' or 'cuda').
                                If not provided, it will be set based on the availability of a GPU.
        eval_iters (int, optional): Number of iterations for evaluation during training.
                                    If not provided, it will be set to a default value.

    Returns:
        BigramLanguageModel: The trained model.

    Note:
        The function performs the training loop for the given model. It sets hyperparameters using the provided values
        or default values from the hyperparams dictionary. The model is moved to the specified device, and the optimizer is
        initialized with the given learning rate. The training loop consists of forward and backward passes, optimization,
        and evaluation at specified intervals. The evaluation results, such as training loss and validation loss, are printed
        at regular intervals. The function returns when the maximum number of iterations is reached.
    """
    set_hyperparams(learning_rate=learning_rate, max_iters=max_iters,
                    eval_interval=eval_interval, device=device)
    device = hyperparams['device']
    m = model.to(device)
    learning_rate = hyperparams['learning_rate']
    max_iters = hyperparams['max_iters']
    eval_interval = hyperparams['eval_interval']
    eval_iters = hyperparams['eval_iters']
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    for iters in tqdm(range(max_iters+1)):
        if iters % eval_interval == 0:
            losses = estimate_loss(m, train_data, val_data, eval_iters)
            print(
                f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data,
                           block_size=None, batch_size=None)
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('Training Completed!')
    return model


def save(path, model):
    """
    Save the state_dict of the provided model to the specified file path.

    Args:
        path (str): The file path where the model state_dict will be saved.
        model (BigramLanguageModel): The model whose state_dict will be saved.

    Note:
        This function uses PyTorch's `torch.save()` method to save the state_dict of the model to the specified file path.
        The state_dict contains the learned parameters of the model, allowing it to be later loaded and used for inference.
    """
    torch.save(model.state_dict(), path)


def load(path, model, device=None):
    """
    Load the model state_dict from the specified file path and assign it to the provided model.

    Args:
        path (str): The file path from which the model state_dict will be loaded.
        model (BigramLanguageModel): The model to which the loaded state_dict will be assigned.
        device (str, optional): The device on which to load the model (e.g., 'cpu' or 'cuda').
                                If not provided, it will be set based on the availability of a GPU.

    Note:
        This function uses PyTorch's `torch.load()` method to load the state_dict of the model from the specified file path.
        The loaded state_dict is then assigned to the provided model, effectively restoring the model's learned parameters.
        The device on which to load the model can also be specified; otherwise, it will default to 'cuda' if a GPU is available,
        or 'cpu' otherwise.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))


def fine_tune_model(model, learning_rate, max_iters, eval_interval, train_data, val_data):
    """
    Fine-tune the provided model using the given training data.

    Args:
        model (BigramLanguageModel): The model to be fine-tuned.
        learning_rate (float): The learning rate for the optimizer during fine-tuning.
        max_iters (int): The maximum number of iterations for fine-tuning.
        eval_interval (int): The interval at which to evaluate the model's performance during fine-tuning.
        train_data (torch.Tensor): The training data for fine-tuning.
        val_data (torch.Tensor): The validation data for evaluating the model's performance during fine-tuning.

    Returns:
        BigramLanguageModel: The Fine-tuned model.

    Note:
        This function uses the AdamW optimizer with the provided learning rate to optimize the model's parameters during
        fine-tuning. It also employs the ReduceLROnPlateau scheduler to adjust the learning rate based on validation loss.
        The model is fine-tuned for the specified number of iterations. At each evaluation interval, the model's performance
        is evaluated on the validation data using the `estimate_loss` function. The learning rate is adjusted based on the
        validation loss using the ReduceLROnPlateau scheduler to prevent overshooting the optimal parameter values.
        Once the fine-tuning process is completed, the function prints a message indicating that the fine-tuning is finished.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.1)

    for iteration in tqdm(range(max_iters+1)):
        if iteration % eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data, eval_iters=None)
            print(
                f"Step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data,
                           block_size=model.block_size, batch_size=None, device=None)
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Adjust learning rate based on validation loss
        val_loss = losses['val']
        scheduler.step(val_loss)
    print('Fine-tuning completed!')
    return model


def gpt2_124M(training_data_path, lr, max_iters, dropout=0.1, eval_iters=200, train_val_split=85):
    """
    Train a GPT-2 model with 124 million parameters on the provided training data.

    Args:
        training_data_path (str): The path to the training data file.
        lr (float): The learning rate for the optimizer during training.
        max_iters (int): The maximum number of iterations for training.
        dropout (float, optional): The dropout rate to be used in the model. Default is 0.1.
        eval_iters (int, optional): The interval at which to evaluate the model's performance during training.
                                    Default is 200.
        train_val_split (int, optional): The percentage of data to be used for training, and the rest for validation.
                                        Default is 85.

    Returns:
        BigramLanguageModel: The trained GPT-2 model.

    Note:
        This function uses the LanguageDataset class to load the training data and split it into training and validation sets.
        The GPT-2 model architecture with 124 million parameters is defined with the specified hyperparameters (n_embd, n_head,
        n_layer, and block_size) and dropout rate. The model is then initialized using the `initialize_model` function and moved
        to the GPU if available.

        The training loop is performed for the specified number of iterations. At each evaluation interval, the model's performance
        is evaluated on the validation data using the `estimate_loss` function. The training loss and validation loss are printed
        at each evaluation interval to monitor the model's progress during training.

        Once the training is completed, the trained GPT-2 model is returned.
        Parameters count depends on the vocab_size (Using a 1.05 MB text file its get Parameters count of 86.199034 million)
    """
    # Load the dataset
    dataset = LanguageDataset(training_data_path, train_val_split)
    train_data = dataset.train_data
    val_data = dataset.val_data
    vocab_size = dataset.vocab_size

    # Define hyperparameters based on GPT-2 architecture(GPT-2 124 Million Parameters)
    n_embd = 768
    n_head = 12
    n_layer = 12
    block_size = 1024

    # Initialize the model
    model = initialize_model(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # count trainable params
    params = count_parameters(model=model)
    print(f"Total trainable params: {params} ")

    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for iteration in tqdm(range(max_iters+1)):
        if iteration % eval_iters == 0:  # Print loss every 200 iterations
            losses = estimate_loss(
                model, train_data, val_data, eval_iters=None)
            print(
                f"Step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data=train_data, val_data=val_data)
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Training completed!')
    return model


def gpt2_finetune(pretrained_model, training_data_path, lr, max_iters, eval_iters=200, train_val_split=85):
    """
    Fine-tune a pretrained GPT-2 model on the provided training data.

    Args:
        pretrained_model (BigramLanguageModel): The pretrained GPT-2 model to be fine-tuned.
        training_data_path (str): The path to the training data file.
        lr (float): The learning rate for the optimizer during fine-tuning.
        max_iters (int): The maximum number of iterations for fine-tuning.
        eval_iters (int, optional): The interval at which to evaluate the model's performance during fine-tuning.
                                    Default is 200.
        train_val_split (int, optional): The percentage of data to be used for training, and the rest for validation.
                                        Default is 85.

    Note:
        This function fine-tunes a pretrained GPT-2 model on the provided training data using the specified learning rate
        and number of iterations. The LanguageDataset class is used to load the training data and split it into training
        and validation sets.

        The optimizer and scheduler are created based on the parameters provided. The fine-tuning loop is performed for
        the specified number of iterations. At each evaluation interval, the model's performance is evaluated on the
        validation data using the `estimate_loss` function. The training loss and validation loss are printed at each
        evaluation interval to monitor the model's progress during fine-tuning.

        The learning rate is adjusted based on the validation loss using the ReduceLROnPlateau scheduler.

        Once the fine-tuning is completed, the pretrained GPT-2 model is updated with the fine-tuned weights and returned.
    """
    dataset = LanguageDataset(training_data_path, train_val_split)
    train_data = dataset.train_data
    val_data = dataset.val_data

    # Create the optimizer
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.1)

    # Fine-tuning loop
    for iteration in tqdm(range(max_iters+1)):
        if iteration % eval_iters == 0:  # Print loss every 200 iterations
            losses = estimate_loss(
                pretrained_model, train_data, val_data, eval_iters=None)
            print(
                f"Step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data=train_data,
                           val_data=val_data, block_size=pretrained_model.block_size)
        logits, loss = pretrained_model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Adjust learning rate based on validation loss
        val_loss = losses['val']
        scheduler.step(val_loss)

    print('Fine-tuning completed!')
    return pretrained_model


default_hyperparams = {
    'batch_size': 64,
    'block_size': 256,
    'max_iters': 5000,
    'learning_rate': 5.859375e-05,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_iters': 200,
    'n_embd': 384,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.2,
}


def set_hyperparamsdefault():
    """
    Set the default hyperparameters for the GPT-2 model (Bigram implementation) using the set_hyperparams function.
    """
    set_hyperparams(**default_hyperparams)
