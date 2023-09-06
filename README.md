# GPT2LM.bigram

Description: [Implementation of GPT-2 language model for text generation, training custom models and fine-tuning.]

---

## LanguageDataset

---

A class representing a language dataset for the Bigram Language Model.

### Class Description

#### Constructor

    def __init__(self, path: str, train_val_split: int = 90)

Initialize the LanguageDataset with the provided text data.

##### Parameters

1. `path` (str): The file path to the text data.

2. `train_val_split` (int, optional): The percentage split between training and validation data. Defaults to 90 (90% training, 10% validation).

---

### Methods

---

#### \_load_dataset

    def _load_dataset(self, path: str) -> str

Load the text data from the given file path.

##### Parameters

1. `path` (str): The file path to the text data.

##### Returns

1. `text` (str): The loaded text data as a string.

#### \_encode

    def _encode(self, s: str) -> List[int]

Convert the text data into a list of numerical representations using character encoding.

##### Parameters

1. `s` (str): The input text data to be encoded.

##### Returns

1. `List[int]`: A list of integer token indices representing the encoded text.

#### \_decode

    def _decode(self, l: List[int]) -> str

Convert a list of numerical representations back into the original text.

##### Parameters

1. `l` (List[int]): A list of integer token indices representing the encoded text.

##### Returns

1. `str`: The decoded text as a string.

#### \_calculate_split

    def _calculate_split(self, train_val_split: int) -> float

Calculate the split index for separating training and validation data.

##### Parameters

1. `train_val_split` (int): The percentage split between training and validation data.

##### Returns

1. `float`: The split index as a float between 0 and 1, representing the percentage split.

---

### Properties

---

#### text

**@property**

    def text(self) -> str

Get the loaded text data as a string.

#### chars

**@property**

    def chars(self) -> List[str]

Get the list of unique characters present in the text data.

#### vocab_size

**@property**

    def vocab_size(self) -> int

Get the size of the character vocabulary.

#### stoi

**@property**

    def stoi(self) -> Dict[str, int]

Get a dictionary mapping characters to their corresponding integer token indices.

#### itos

**@property**

    def itos(self) -> Dict[int, str]

Get a dictionary mapping integer token indices to their corresponding characters.

#### train_data

**@property**

    def train_data(self) -> torch.Tensor

Get the training data as a tensor containing numerical representations of the text.

#### val_data

**@property**

    def val_data(self) -> torch.Tensor

Get the validation data as a tensor containing numerical representations of the text.

---

---

## Head

A single head of self-attention used in the Transformer block.

### Constructor

    def __init__(self, head_size, n_embd, block_size, dropout):

Initialize the Head with linear layers and a lower triangular mask.

#### Parameters

1. `head_size` (int): The size of the attention head.

2. `n_embd` (int): The embedding dimension of the input.

3. `block_size` (int): The maximum length of the input sequence.

4. `dropout` (float): The dropout rate.

### Methods

#### forward

    def forward(self, x):

Perform the forward pass of the attention head.

##### Parameters

1. `x` (torch.Tensor): The input tensor of shape `(B, T, C)`, where `B` is the batch size, `T` is the sequence length, and `C` is the embedding dimension.

##### Returns

1. `torch.Tensor`: The output tensor after the self-attention operation. It has the same shape as the input tensor `(B, T, C)`.

---

---

## MultiHeadAttention

Multi-head self-attention module used in the Transformer block.

### Class Description

### Constructor

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):

Initialize the MultiHeadAttention with attention heads and projection layer.

#### Parameters

1. `num_heads` (int): The number of attention heads.

2. `head_size` (int): The size of each attention head.

3. `n_embd` (int): The embedding dimension of the input.

4. `block_size` (int): The maximum length of the input sequence.

5. `dropout` (float): The dropout rate.

---

### Methods

---

#### forward

    def forward(self, x):

Perform the forward pass of the multi-head self-attention module.

##### Parameters

1. `x` (torch.Tensor): The input tensor of shape `(B, T, C)`, where `B` is the batch size, `T` is the sequence length, and `C` is the embedding dimension.

##### Returns

1. `torch.Tensor`: The output tensor after applying the multi-head self-attention. It has the same shape as the input tensor `(B, T, C)`.

---

---

## FeedForward

Feed-forward neural network module used in the Transformer block.

### Constructor

    def __init__(self, n_embd, dropout):

Initialize the FeedForward with linear layers and activation functions.

#### Parameters

1. `n_embd` (int): The embedding dimension of the input.

2. `dropout` (float): The dropout rate.

### Methods

##### forward

    def forward(self, x):

Perform the forward pass of the feed-forward neural network.

##### Parameters

1. `x` (torch.Tensor): The input tensor of shape `(B, T, C)`, where `B` is the batch size, `T` is the sequence length, and `C` is the embedding dimension.

##### Returns

1. `torch.Tensor`: The output tensor after the feed-forward operation. It has the same shape as the input tensor `(B, T, C)`.

---

---

## Block

---

Transformer block module used in the BigramLanguageModel.

### Constructor

    def __init__(self, n_embd, n_head, block_size, dropout):

Initialize the Transformer block with self-attention and feed-forward modules.

#### Parameters

1. `n_embd` (int): The embedding dimension of the input.

2. `n_head` (int): The number of attention heads in the multi-head self-attention.

3. `block_size` (int): The maximum length of the input sequence.

4. `dropout` (float): The dropout rate.

---

### Methods

---

#### forward

    def forward(self, x):

Perform the forward pass of the transformer block.

##### Parameters

1. `x` (torch.Tensor): The input tensor of shape `(B, T, C)`, where `B` is the batch size, `T` is the sequence length, and `C` is the embedding dimension.

##### Returns

1. `torch.Tensor`: The output tensor after applying the transformer block. It has the same shape as the input tensor `(B, T, C)`.

---

---

## BigramLanguageModel

---

### Class Description

#### Constructor

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout)

#### Parameters

1. `vocab_size` (int): The size of the vocabulary, which determines the number of tokens in the language model.

2. `n_embd` (int): The embedding dimension of the input.

3. `n_head` (int): The number of attention heads in the multi-head self-attention.

4. `n_layer` (int): The number of Transformer blocks in the model.

5. `block_size` (int): The maximum length of the input sequence.

6. `dropout` (float): The dropout rate for regularization.

---

### Methods

---

#### \_build_block

    def _build_block(self) -> Block

Creates a single Transformer block with specified embedding dimensions and attention heads.

#### forward

    def forward(self, idx: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]

Performs the forward pass of the BigramLanguageModel.

##### Parameters

1. `idx` (torch.Tensor): The input tensor representing the token indices of shape (B, T), where B is the batch size, and T is the sequence length.

2. `target` (torch.Tensor, optional): The target tensor representing the token indices for computing the loss. It should have the same shape as idx. If None, no loss will be computed. (Default: None)

##### Returns

1. `torch.Tensor`: The logits tensor after the forward pass of the model. It has the shape (B, T, vocab_size).

2. `torch.Tensor`: The loss tensor, computed using F.cross_entropy if the target is provided. Otherwise, None. It has the shape (B \* T,) if target is not None.

#### generate

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> List[int]

Generates new tokens using the trained language model.

##### Parameters

1. `idx` (torch.Tensor): The input tensor representing the token indices of shape (B, T), where B is the batch size, and T is the sequence length.

2. `max_new_tokens` (int): The maximum number of new tokens to generate.

##### Returns

1. `List[int]`: A list of generated token indices as integers.

---

---

## Utility Functions

---

### set_hyperparams

    def set_hyperparams(**kwargs)

Update the hyperparameters dictionary with the provided keyword arguments.

#### Parameters

1. `**kwargs`: Keyword arguments with hyperparameter names as keys and their corresponding values.

### load_dataset

    def load_dataset(path: str, train_val_split: int = 90) -> LanguageDataset

Load a dataset from the given file path and create a **LanguageDataset** object.

#### Parameters

1. `path` (str): The file path of the dataset to load.

2. `train_val_split` (int, optional): Percentage split between training and validation data. Default is 90, which means 90% training and 10% validation.

#### Returns

1. `LanguageDataset` : An instance of the **LanguageDataset** class containing the loaded dataset.

### initialize_model

    def initialize_model(vocab_size: int, n_embd: Optional[int] = None, n_head: Optional[int] = None, n_layer: Optional[int] = None, block_size: Optional[int] = None, dropout: Optional[float] = None, device: Optional[str] = None) -> BigramLanguageModel

Initialize a BigramLanguageModel with the given hyperparameters.

#### Parameters

1. `vocab_size` (int): The size of the vocabulary, which determines the number of tokens in the language model.

2. `n_embd` (int, optional): The embedding dimension of the input. If not provided, it will be set to a default value.

3. `n_head` (int, optional): The number of attention heads in the multi-head self-attention. If not provided, it will be set to a default value.

4. `n_layer` (int, optional): The number of Transformer blocks in the model. If not provided, it will be set to a default value.

5. `block_size` (int, optional): The maximum length of the input sequence. If not provided, it will be set to a default value.

6. `dropout` (float, optional): The dropout rate for regularization. If not provided, it will be set to a default value.

7. `device` (str, optional): The device on which to place the model (e.g., 'cpu' or 'cuda'). If not provided, it will be set based on the availability of a GPU.

#### Returns

1. `BigramLanguageModel`: An instance of the **BigramLanguageModel** class initialized with the given hyperparameters.

### train

    def train(model: BigramLanguageModel, train_data: torch.Tensor, val_data: torch.Tensor, learning_rate: Optional[float] = None, max_iters: Optional[int] = None, eval_interval: Optional[int] = None, device: Optional[str] = None, eval_iters: Optional[int] = None) -> BigramLanguageModel

Train the provided model on the given training data and validate it on the validation data.

#### Parameters

1. `model` (BigramLanguageModel): The language model to train.

2. `train_data` (torch.Tensor): The training data tensor.

3. `val_data` (torch.Tensor): The validation data tensor.

4. `learning_rate` (float, optional): The learning rate for the optimizer. If not provided, it will be set to a default value.

5. `max_iters` (int, optional): The maximum number of training iterations. If not provided, it will be set to a default value.

6. `eval_interval` (int, optional): Interval for printing evaluation results during training. If not provided, it will be set to a default value.

7. `device` (str, optional): The device on which to train the model (e.g., 'cpu' or 'cuda'). If not provided, it will be set based on the availability of a GPU.

8. `eval_iters` (int, optional): Number of iterations for evaluation during training. If not provided, it will be set to a default value.

#### Returns

1. `BigramLanguageModel`: The trained model.

### save

    def save(path: str, model: BigramLanguageModel)

Save the state_dict of the provided model to the specified file path.

#### Parameters

1. `path` (str): The file path where the model state_dict will be saved.

2. `model` (BigramLanguageModel): The model whose state_dict will be saved.

### load

    def load(path: str, model: BigramLanguageModel, device: Optional[str] = None)

Load the model state_dict from the specified file path and assign it to the provided model.

#### Parameters

1. `path` (str): The file path from which the model state_dict will be loaded.

2. `model` (BigramLanguageModel): The model to which the loaded state_dict will be assigned.

3. `device` (str, optional): The device on which to load the model (e.g., 'cpu' or 'cuda'). If not provided, it will be set based on the availability of a GPU.

### fine_tune_model

    def fine_tune_model(model: BigramLanguageModel, learning_rate: float, max_iters: int, eval_interval: int, train_data: torch.Tensor, val_data: torch.Tensor) -> BigramLanguageModel

Fine-tune the provided model using the given training data.

#### Parameters

1. `model` (BigramLanguageModel): The model to be fine-tuned.

2. `learning_rate` (float): The learning rate for the optimizer during fine-tuning.

3. `max_iters` (int): The maximum number of iterations for fine-tuning.

4. `eval_interval` (int): The interval at which to evaluate the model's performance during fine-tuning.

5. `train_data` (torch.Tensor): The training data for fine-tuning.

6. `val_data` (torch.Tensor): The validation data for evaluating the model's performance during fine-tuning.

#### Returns

1. `BigramLanguageModel`: The fine-tuned model.

### gpt2_124M

    def gpt2_124M(training_data_path: str, lr: float, max_iters: int, dropout: float = 0.1, eval_iters: int = 200, train_val_split: int = 85) -> BigramLanguageModel

Train a GPT-2 model with 124 million parameters on the provided training data.

#### Parameters

1. `training_data_path` (str): The path to the training data file.

2. `lr` (float): The learning rate for the optimizer during training.

3. `max_iters` (int): The maximum number of iterations for training.

4. `dropout` (float, optional): The dropout rate to be used in the model. Default is 0.1.

5. `eval_iters` (int, optional): The interval at which to evaluate the model's performance during training. Default is 200.

6. `train_val_split` (int, optional): The percentage of data to be used for training, and the rest for validation. Default is 85.

#### Returns

1. `BigramLanguageModel`: The trained GPT-2 model.

#### NOTE

1. **_`Parameters count depends on the vocab_size.`_**

### gpt2_finetune

    def gpt2_finetune(pretrained_model: BigramLanguageModel, training_data_path: str, lr: float, max_iters: int, eval_iters: int = 200, train_val_split: int = 85) -> BigramLanguageModel

Fine-tune a pretrained GPT-2 model on the provided training data.

#### Parameters

1. `pretrained_model` (BigramLanguageModel): The pretrained GPT-2 model to be fine-tuned.

2. `training_data_path` (str): The path to the training data file.

3. `lr` (float): The learning rate for the optimizer during fine-tuning.

4. `max_iters` (int): The maximum number of iterations for fine-tuning.

5. `eval_iters` (int, optional): The interval at which to evaluate the model's performance during fine-tuning. Default is 200.

6. `train_val_split` (int, optional): The percentage of data to be used for training, and the rest for validation. Default is 85.

#### Returns

1. `BigramLanguageModel`: The fine-tuned model.

### count_parameters

    def count_parameters(model: torch.nn.Module) -> int

Count the total number of trainable parameters in the given PyTorch model.

#### Parameters

1. `model` (torch.nn.Module): The PyTorch model for which the parameters need to be counted.

#### Returns

1. `int`: The total number of trainable parameters in the model.

### get_batch

    def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor, block_size: int = None, batch_size: int = None, device: str = None) -> Tuple[torch.Tensor, torch.Tensor]

Get a batch of data for training or validation from the specified split.

#### Parameters

1. `split` (str): The split to get the data from ('train' or 'val').

2. `train_data` (torch.Tensor): The training data tensor.

3. `val_data` (torch.Tensor): The validation data tensor.

4. `block_size` (int, optional): The size of each input sequence block. Defaults to None.

5. `batch_size` (int, optional): The number of sequences in a batch. Defaults to None.

6. `device` (str, optional): The device to store the batch data on ('cpu' or 'cuda'). Defaults to None.

#### Returns

1. `Tuple`[torch.Tensor, torch.Tensor]: A tuple containing two tensors - the input tensor representing the batch of sequences (x) and the target tensor representing the batch of sequences shifted by one (y).

### estimate_loss

    def estimate_loss(model: torch.nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, eval_iters: int = None, block_size: int = None, batch_size: int = None, device: str = None) -> Dict[str, float]

Estimate the average loss of the model on the training and validation data.

### Parameters

1. `model` (torch.nn.Module): The PyTorch model for which the loss is to be estimated.

2. `train_data` (torch.Tensor): The training data tensor.

3. `val_data` (torch.Tensor): The validation data tensor.

4. `eval_iters` (int, optional): The number of iterations for estimating the loss. Defaults to None.

5. `block_size` (int, optional): The size of each input sequence block. Defaults to None.

6. `batch_size` (int, optional): The number of sequences in a batch. Defaults to None.

7. `device` (str, optional): The device to perform computations on ('cpu' or 'cuda'). Defaults to None.

#### Returns

1. `Dict`[str, float]: A dictionary containing the average loss on the training and validation data.

   _`Keys`_: 'train' and 'val'

   _`Values`_: Average loss values (float)

### default_hyperparams

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

A dictionary containing the default hyperparameters for the GPT-2 model.

### set_hyperparamsdefault

      def set_hyperparamsdefault()

Set the default hyperparameters for the GPT-2 model using the set_hyperparams function.

---

## Example Usage

### Train model on default Hyperparameters

    # Import the required functions and classes

    from GPT2LM import bigram as b

    # Path to your training data file

    training_data_path = 'path_to_your_training_data.txt'

    # Load the dataset and split into training and validation sets (default split: 85% training, 15% validation)

    dataset = b.load_dataset(training_data_path)

    # set default hyperparams

    b.set_hyperparams()

    # Initialize the GPT-2 model with default hyperparameters and train on the dataset

    model = b.initialize_model(vocab_size=dataset.vocab_size) # keep record of the 'vocab_size' to use later.

    # Train the model on the training data and validate it on the validation data

    trained_model = b.train(model, dataset.train_data, dataset.val_data)

    # Save the trained model to a file

    b.save('path_to_save_model.pt', trained_model)

### Train model on custom hyperparameters

    # Import the required functions and classes

    from GPT2LM import bigram as b

    # Path to your training data file

    training_data_path = 'path_to_your_training_data.txt'

    # Load the dataset and split into training and validation sets (default split: 85% training, 15% validation)

    dataset = b.load_dataset(training_data_path)

    # set custom hyperparams

    3 This will assign custom values to default hyperparams

    batch_size=32
    block_size=500
    max_iters=10000
    learning_rate=0.0001
    eval_interval=250
    n_embd=800
    n_head=8
    n_layer=8
    dropout=0.1

    b.set_hyperparams(batch_size=batch_size,
                    block_size=block_size,
                    max_iters=max_iters,
                    learning_rate=learning_rate,
                    eval_interval=eval_interval,
                    n_embd=n_embd,
                    n_head=n_head,
                    n_layer=n_layer,
                    dropout=dropout
    )

    # Initialize the GPT-2 model with custom hyperparameters and train on the dataset

    model = b.initialize_model(vocab_size=dataset.vocab_size) # keep record of the 'vocab_size' to use later.

    # Train the model on the training data and validate it on the validation data

    trained_model = b.train(model, dataset.train_data, dataset.val_data)

    # Save the trained model to a file

    b.save('path_to_save_model.pt', trained_model)

### Generate text

    # initiate context
    context=torch.zeros((1,1),dtype=torch.long,device=device)

    # generate the new tokens and decode them into text
    text=dataset._decode(model.generate(context,max_new_tokens=200))

### Generate text with prompt

    # initiate context
    context=torch.zeros((1,1),dtype=torch.long,device=device)

    # Give prompt

    prompt = "To be or not to be, that is the question."

    # encode prompt into tokens

    prompt_encoded = torch.tensor(dataset._encode(prompt), dtype=torch.long, device=device)
    prompt_encoded = prompt_encoded.unsqueeze(0)

    # initiate context with prompt

    context_with_prompt = torch.cat((context, prompt_encoded), dim=1)

    # generate the new tokens and decode them into text

    text=dataset._decode(model.generate(context_with_prompt,max_new_tokens=200))

## About

This project is designed for educational purposes to help you learn and understand how generative AI works. It demonstrates the implementation of a GPT-2 variant, a popular generative language model, and provides a hands-on experience in training language models on custom datasets. By exploring the code and running the provided examples, you can gain insights into the underlying principles of generative AI and natural language processing.

Please note that while this project serves as a learning resource, it is essential to be mindful of the ethical considerations and potential risks associated with AI models. The use of AI technology, especially in generating text, should be done responsibly and with respect for ethical guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

## Talks

Here are some links for better understanding.

1. **_`Attention Is All You Need`_** [link here](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

2. **_`Textbooks Are All You Need`_** [Link here](https://www.researchgate.net/publication/371729045_Textbooks_Are_All_You_Need)

3. **_`GPT(Generative Pre-trained Transformers)`_** [Link here](https://aws.amazon.com/what-is/gpt/)

4. **_`Language Models are Unsupervised Multitask Learners`_** [Link here](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

5. **_`GPT-2`_** [Link here](https://en.wikipedia.org/wiki/GPT-2)
6. **_`GELU`_** [Link here](https://paperswithcode.com/method/gelu)
7. **_`ReLU`_** [Link here](https://paperswithcode.com/method/relu)
8. **_`ReLU VS GELU`_** [Link here](https://towardsai.net/p/l/is-gelu-the-relu-successor)
9. **_`GIT Link of this project`_** [Link here](https://github.com/YashrajBaila7/GPT2LM)

Code is completely open source so you can play around it and get a better understanding of it.

**_GPT-3_** Hyperparameter are not publicly available yet but you can use your own hyperparameter to train a **_gpt-3_** like base model.

### How GPT works

Generative Pre-trained Transformer (GPT) models, such as GPT-2, are designed to predict the next word or token in a sequence of text. They achieve this by leveraging the power of self-attention mechanisms within the Transformer architecture.

At a high level, here's how GPT models work to predict the next word or token:

1. `Language Modeling`: GPT models are trained on large amounts of text data using a process called language modeling. During training, the model learns the statistical patterns and dependencies present in the input text. It captures the relationships between different words and the likelihood of one word following another in a sentence.

2. `Contextual Understanding`: The key feature of GPT models is their ability to understand the context in which a word or token appears. Instead of treating each word in isolation, the model considers the entire sequence of tokens and assigns higher importance to the tokens that are most relevant for understanding the current word.

3. `Self-Attention`: Self-attention is the mechanism that allows GPT models to weigh the importance of different tokens in the input sequence based on their contextual relevance. Tokens that are semantically related or have strong dependencies receive higher attention scores, while unrelated tokens receive lower attention scores.

4. `Autoregressive Generation`: To predict the next word or token in a sequence, GPT models use an autoregressive generation process. This means that the model predicts each token one by one, conditioned on the tokens that came before it in the sequence. The previously predicted tokens serve as context to guide the prediction of the next token.

5. `Text Completion`: By repeatedly predicting the next token and using the predicted tokens as context, the GPT model can complete a given prompt or generate entirely new text. The generated text is coherent and contextually appropriate because the model has learned from vast amounts of text data during pre-training.

6. `Adaptability`: GPT models are highly adaptable and can be fine-tuned on specific tasks or domains after pre-training. This process allows them to perform well on a wide range of natural language processing tasks, such as text generation, machine translation, question answering, and more.

In summary, GPT models are powerful language models that predict the next word or token in a sequence by understanding the context and dependencies between words. Their ability to generate coherent and contextually relevant text makes them valuable tools for various natural language processing applications.

#### How we trick them to chat with user

As these models are designed to complete given documents, they are not capable of engaging in real-time conversations with users. If you present them with a question, they will simply generate more questions that resemble the input. To overcome this limitation, we use a specific format for interaction.

For instance, we frame the conversation as if it's between a human and an AI. This format helps the AI understand its role and context better. Here's an example prompt:

"AI: What can I do for you today?
\nHuman: ..."

In this format, the human provides their input after "Human:", and the AI responds accordingly. The entire conversation is treated as a single document, with the AI's responses generated by the model and the human's responses provided by an actual person.

This approach allows us to harness the power of the language model to generate creative and contextually relevant AI responses, while human input ensures that the conversation remains meaningful and coherent.

By using this interaction format, we can effectively leverage the capabilities of the language model in completing text documents, making it a useful tool for various natural language processing tasks.

### What is new in this version?

There are no such big changes in bigram, the only difference is the use of the **`GELU`** activation function instead of the **`ReLU`** activation function.

`ReLU(x)` or `Rectified Linear Unit` = max(0,x)

`GELU(x)` or `Gaussian Error Linear Unit` = xΦ(x) = 0.5x(1+tanh[2/π(x+0.044715x^3)])

#### Why GELU is used?

The `Gaussian Error Linear Unit (GELU)` is an activation function used in deep learning. It is defined as `xΦ(x)`, where `Φ(x)` is the `standard Gaussian cumulative distribution function`. The GELU nonlinearity weights inputs by their percentile, rather than gating inputs by their sign as in ReLUs. Consequently, the GELU can be thought of as a smoother ReLU.

It is not known why certain activation functions work better than others in different contexts. So the only answer for “why use GELU instead of ReLu” is "because it works better". However, there are some possible explanations. For example, ReLU can suffer from "problems where a significant amount of neurons in the network become zero and don’t practically do anything". GELU is smoother near zero and “is differentiable in all ranges, and allows to have gradients (although small) in the negative range” which helps with this problem.

In summary, GELU is used in generative AI instead of ReLU because it has been found to work better in practice. It is a smoother version of ReLU that allows for gradients in the negative range, which can help prevent problems with neurons becoming inactive.

#### But why even use activation functions?

Activation functions are used in neural networks to introduce non-linearity into the output of a neuron. Without an activation function, the weights and bias would simply do a linear transformation. A linear equation is simple to solve but is limited in its capacity to solve complex problems and have less power to learn complex functional mappings from data. A neural network without an activation function is just a **`linear regression model`**.

Generally, neural networks use non-linear activation functions, which can help the network learn complex data, compute and learn almost any function representing a question, and provide accurate predictions. Activation functions provide the building blocks that can be used repeatedly in two dimensions of the network structure so that, combined with an attenuation matrix to vary the weight of signaling from layer to layer, is known to be able to approximate an arbitrary and complex function.

In summary, activation functions are used in neural networks to introduce non-linearity into the output of a neuron, which allows the network to learn complex data and provide accurate predictions. They are essential building blocks of neural networks.

### Is the model is GPT (GPT VS Bigram) ?

A bigram is a type of n-gram model that represents text as the joint probability of a sequence of two words. Bigrams, along with other n-grams, are used in most successful language models for speech recognition.

On the other hand, GPT (Generative Pretrained Transformer) is a type of neural language model that uses a transformer architecture to generate text. GPT is a modern Transformer language model, which is more complex than a bigram character-level language model.

In summary, a bigram is a simple statistical model that represents text as the joint probability of a sequence of two words, while GPT is a more complex neural language model that uses a transformer architecture to generate text. The two models have different strengths and weaknesses and can be used for different purposes.

#### But it does not answer my question.

Certainly, let's differentiate between `GPT` (a type of transformer model), a `bigram` model, and the `custom BigramLanguageModel` :

1. **`GPT (Generative Pre-trained Transformer)`**:

   1. `Architecture`: GPT is based on the transformer architecture, which consists of a stack of multiple layers of self-attention and feedforward neural networks.
   2. `Self-Attention`: GPT uses multi-head self-attention to capture relationships between tokens in the input sequence.
   3. `Parameter Sharing`: GPT is a fully autoregressive model, meaning it generates one token at a time based on the previously generated tokens. It doesn't rely on fixed n-grams like bigram models.
   4. `Pre-training`: GPT is pre-trained on a large corpus of text data, learning language patterns, and semantic understanding from diverse text sources.
   5. `Generative`: GPT is designed for generating text and is commonly used for tasks like text completion, text generation, and language translation.

2. **`Bigram Model`**:

   1. `Architecture`: A bigram model is a simple statistical language model that focuses on pairs of consecutive words (bigrams) in a text corpus.
   2. `Scope`: It doesn't have a deep neural network architecture like transformers or GPT. Instead, it relies on counting the frequency of bigrams in a given text corpus.
   3. `Parameterization`: Bigram models have a relatively small number of parameters compared to deep learning models.
   4. `Generative Power`: Bigram models are less expressive than GPT and cannot capture complex language structures or long-range dependencies in the same way.
   5. `Use Case`: Bigram models are often used for basic language modeling tasks but are less suited for generating coherent and contextually rich text compared to GPT.

3. **`Custom Model (BigramLanguageModel)`**:

   1. `Architecture`:This model a custom implementation. It combines elements of self-attention and feedforward neural networks within each block.
   2. `Embeddings`: It uses token embeddings and positional embeddings, which are common in transformer-based models.
   3. `Multi-Head Attention`: Like GPT, it incorporates multi-head self-attention within each block.
   4. `Complexity`: This custom model is more complex and expressive than a simple bigram model but may not have the same generative power or pre-training as GPT.

In summary, GPT is a state-of-the-art transformer-based model designed for generative tasks, bigram models are simpler statistical models focusing on word pairs, and the custom model combines elements of transformers but is a custom implementation tailored for a specific purpose. Each model has its own strengths and weaknesses depending on the task at hand.

### What you have to do.

Go to Colab or Kaggle and train two models, both on the same dataset but with different versions of GPT2LM, to understand how the model works with different activation functions and why generative AI like GPT uses GELU instead of ReLU, probably the most popular activation function.

Also implement your own ReLU and GELU function and plot the graph of them using matplotlib on two or three different ranges to see the actual difference in both functions because on large ranges the graph will overlap each other and appears as the same graph. So on smaller ranges [-1,1] you will see a very small difference in graphs.
