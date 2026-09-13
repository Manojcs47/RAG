# Hugging Face NLP Course — Chapter 2

Source: https://huggingface.co/learn/nlp-course/chapter2


---

<!-- Section 2.1 -->

# Introduction[[introduction]]

<CourseFloatingBanner
    chapter={2}
    classNames="absolute z-10 right-0 top-0"
/>

As you saw in [Chapter 1](/course/chapter1), Transformer models are usually very large. With millions to tens of *billions* of parameters, training and deploying these models is a complicated undertaking. Furthermore, with new models being released on a near-daily basis and each having its own implementation, trying them all out is no easy task.

The 🤗 Transformers library was created to solve this problem. Its goal is to provide a single API through which any Transformer model can be loaded, trained, and saved. The library's main features are:

- **Ease of use**: Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code.
- **Flexibility**: At their core, all models are simple PyTorch `nn.Module` classes and can be handled like any other models in their respective machine learning (ML) frameworks.
- **Simplicity**: Hardly any abstractions are made across the library. The "All in one file" is a core concept: a model's forward pass is entirely defined in a single file, so that the code itself is understandable and hackable.

This last feature makes 🤗 Transformers quite different from other ML libraries. The models are not built on modules 
that are shared across files; instead, each model has its own layers. In addition to making the models more approachable and understandable, this allows you to easily experiment on one model without affecting others.

This chapter will begin with an end-to-end example where we use a model and a tokenizer together to replicate the `pipeline()` function introduced in [Chapter 1](/course/chapter1). Next, we'll discuss the model API: we'll dive into the model and configuration classes, and show you how to load a model and how it processes numerical inputs to output predictions. 

Then we'll look at the tokenizer API, which is the other main component of the `pipeline()` function. Tokenizers take care of the first and last processing steps, handling the conversion from text to numerical inputs for the neural network, and the conversion back to text when it is needed. Finally, we'll show you how to handle sending multiple sentences through a model in a prepared batch, then wrap it all up with a closer look at the high-level `tokenizer()` function.

> [!TIP]
> ⚠️ In order to benefit from all features available with the Model Hub and 🤗 Transformers, we recommend <a href="https://huggingface.co/join">creating an account</a>.

---

<!-- Section 2.2 -->

<FrameworkSwitchCourse {fw} />

# Behind the pipeline[[behind-the-pipeline]]

<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section2_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter2/section2_pt.ipynb"},
]} />

<Youtube id="1pedAIvTWXk"/>

Let's start with a complete example, taking a look at what happened behind the scenes when we executed the following code in [Chapter 1](/course/chapter1):

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

and obtained:

```python out
[
    {"label": "POSITIVE", "score": 0.9598047137260437},
    {"label": "NEGATIVE", "score": 0.9994558095932007},
]
```

As we saw in [Chapter 1](/course/chapter1), this pipeline groups together three steps: preprocessing, passing the inputs through the model, and postprocessing:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline.svg" alt="The full NLP pipeline: tokenization of text, conversion to IDs, and inference through the Transformer model and the model head."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline-dark.svg" alt="The full NLP pipeline: tokenization of text, conversion to IDs, and inference through the Transformer model and the model head."/>
</div>

Let's quickly go over each of these.

## Preprocessing with a tokenizer[[preprocessing-with-a-tokenizer]]

Like other neural networks, Transformer models can't process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a *tokenizer*, which will be responsible for:

- Splitting the input into words, subwords, or symbols (like punctuation) that are called *tokens*
- Mapping each token to an integer
- Adding additional inputs that may be useful to the model

All this preprocessing needs to be done in exactly the same way as when the model was pretrained, so we first need to download that information from the [Model Hub](https://huggingface.co/models). To do this, we use the `AutoTokenizer` class and its `from_pretrained()` method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model's tokenizer and cache it (so it's only downloaded the first time you run the code below).

Since the default checkpoint of the `sentiment-analysis` pipeline is `distilbert-base-uncased-finetuned-sst-2-english` (you can see its model card [here](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)), we run the following:

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Once we have the tokenizer, we can directly pass our sentences to it and we'll get back a dictionary that's ready to feed to our model! The only thing left to do is to convert the list of input IDs to tensors.

You can use 🤗 Transformers without having to worry about which ML framework is used as a backend; it might be PyTorch or Flax for some models. However, Transformer models only accept *tensors* as input. If this is your first time hearing about tensors, you can think of them as NumPy arrays instead. A NumPy array can be a scalar (0D), a vector (1D), a matrix (2D), or have more dimensions. It's effectively a tensor; other ML frameworks' tensors behave similarly, and are usually as simple to instantiate as NumPy arrays.

To specify the type of tensors we want to get back (PyTorch or plain NumPy), we use the `return_tensors` argument:

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

Don't worry about padding and truncation just yet; we'll explain those later. The main things to remember here are that you can pass one sentence or a list of sentences, as well as specifying the type of tensors you want to get back (if no type is passed, you will get a list of lists as a result).

Here's what the results look like as PyTorch tensors:

```python out
{
    "input_ids": tensor(
        [
            [
                101,
                1045,
                1005,
                2310,
                2042,
                3403,
                2005,
                1037,
                17662,
                12172,
                2607,
                2026,
                2878,
                2166,
                1012,
                102,
            ],
            [101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    "attention_mask": tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
}
```

The output itself is a dictionary containing two keys, `input_ids` and `attention_mask`. `input_ids` contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence. We'll explain what the `attention_mask` is later in this chapter. 

## Going through the model[[going-through-the-model]]

We can download our pretrained model the same way we did with our tokenizer. 🤗 Transformers provides an `AutoModel` class which also has a `from_pretrained()` method:

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

In this code snippet, we have downloaded the same checkpoint we used in our pipeline before (it should actually have been cached already) and instantiated a model with it.

This architecture contains only the base Transformer module: given some inputs, it outputs what we'll call *hidden states*, also known as *features*. For each model input, we'll retrieve a high-dimensional vector representing the **contextual understanding of that input by the Transformer model**.

If this doesn't make sense, don't worry about it. We'll explain it all later.

While these hidden states can be useful on their own, they're usually inputs to another part of the model, known as the *head*. In [Chapter 1](/course/chapter1), the different tasks could have been performed with the same architecture, but each of these tasks will have a different head associated with it.

### A high-dimensional vector?[[a-high-dimensional-vector]]

The vector output by the Transformer module is usually large. It generally has three dimensions:

- **Batch size**: The number of sequences processed at a time (2 in our example).
- **Sequence length**: The length of the numerical representation of the sequence (16 in our example).
- **Hidden size**: The vector dimension of each model input.

It is said to be "high dimensional" because of the last value. The hidden size can be very large (768 is common for smaller models, and in larger models this can reach 3072 or more).

We can see this if we feed the inputs we preprocessed to our model:

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

```python out
torch.Size([2, 16, 768])
```

Note that the outputs of 🤗 Transformers models behave like `namedtuple`s or dictionaries. You can access the elements by attributes (like we did) or by key (`outputs["last_hidden_state"]`), or even by index if you know exactly where the thing you are looking for is (`outputs[0]`).

### Model heads: Making sense out of numbers[[model-heads-making-sense-out-of-numbers]]

The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension. They are usually composed of one or a few linear layers:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg" alt="A Transformer network alongside its head."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head-dark.svg" alt="A Transformer network alongside its head."/>
</div>

The output of the Transformer model is sent directly to the model head to be processed.

In this diagram, the model is represented by its embeddings layer and the subsequent layers. The embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token. The subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences.

There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:

- `*Model` (retrieve the hidden states)
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`
- and others 🤗

For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative). So, we won't actually use the `AutoModel` class, but `AutoModelForSequenceClassification`:

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

Now if we look at the shape of our outputs, the dimensionality will be much lower: the model head takes as input the high-dimensional vectors we saw before, and outputs vectors containing two values (one per label):

```python
print(outputs.logits.shape)
```

```python out
torch.Size([2, 2])
```

Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.

## Postprocessing the output[[postprocessing-the-output]]

The values we get as output from our model don't necessarily make sense by themselves. Let's take a look:

```python
print(outputs.logits)
```

```python out
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

Our model predicted `[-1.5607, 1.6123]` for the first sentence and `[ 4.1692, -3.3464]` for the second one. Those are not probabilities but *logits*, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities, they need to go through a [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) layer (all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy):

```py
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```python out
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

Now we can see that the model predicted `[0.0402, 0.9598]` for the first sentence and `[0.9995,  0.0005]` for the second one. These are recognizable probability scores.

To get the labels corresponding to each position, we can inspect the `id2label` attribute of the model config (more on this in the next section):

```python
model.config.id2label
```

```python out
{0: "NEGATIVE", 1: "POSITIVE"}
```

Now we can conclude that the model predicted the following:
 
- First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
- Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005

We have successfully reproduced the three steps of the pipeline: preprocessing with tokenizers, passing the inputs through the model, and postprocessing! Now let's take some time to dive deeper into each of those steps.

> [!TIP]
> ✏️ **Try it out!** Choose two (or more) texts of your own and run them through the `sentiment-analysis` pipeline. Then replicate the steps you saw here yourself and check that you obtain the same results!


---

<!-- Section 2.3 -->

<FrameworkSwitchCourse {fw} />

# Models[[the-models]]

<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section3_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter2/section3_pt.ipynb"},
]} />

<Youtube id="AhChOFRegn4"/>

In this section, we'll take a closer look at creating and using models. We'll use the `AutoModel` class, which is handy when you want to instantiate any model from a checkpoint.

## Creating a Transformer[[creating-a-transformer]]

Let's begin by examining what happens when we instantiate an `AutoModel`:

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

Similar to the tokenizer, the `from_pretrained()` method will download and cache the model data from the Hugging Face Hub. As mentioned previously, the checkpoint name corresponds to a specific model architecture and weights, in this case a BERT model with a basic architecture (12 layers, 768 hidden size, 12 attention heads) and cased inputs (meaning that the uppercase/lowercase distinction is important). There are many checkpoints available on the Hub — you can explore them [here](https://huggingface.co/models).

The `AutoModel` class and its associates are actually simple wrappers designed to fetch the appropriate model architecture for a given checkpoint. It's an "auto" class meaning it will guess the appropriate model architecture for you and instantiate the correct model class. However, if you know the type of model you want to use, you can use the class that defines its architecture directly:

```py
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

## Loading and saving[[loading-and-saving]]

Saving a model is as simple as saving a tokenizer. In fact, the models actually have the same `save_pretrained()` method, which saves the model's weights and architecture configuration:

```py
model.save_pretrained("directory_on_my_computer")
```

This will save two files to your disk:

```
ls directory_on_my_computer

config.json model.safetensors
```

If you look inside the *config.json* file, you'll see all the necessary attributes needed to build the model architecture. This file also contains some metadata, such as where the checkpoint originated and what 🤗 Transformers version you were using when you last saved the checkpoint.

The *pytorch_model.safetensors* file is known as the state dictionary; it contains all your model's weights. The two files work together: the configuration file is needed to know about the model architecture, while the model weights are the parameters of the model.

To reuse a saved model, use the `from_pretrained()` method again:

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")
```

A wonderful feature of the 🤗 Transformers library is the ability to easily share models and tokenizers with the community. To do this, make sure you have an account on [Hugging Face](https://huggingface.co). If you're using a notebook, you can easily log in with this:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Otherwise, at your terminal run:

```bash
huggingface-cli login
```

Then you can push the model to the Hub with the `push_to_hub()` method:

```py
model.push_to_hub("my-awesome-model")
```

This will upload the model files to the Hub, in a repository under your namespace named *my-awesome-model*. Then, anyone can load your model with the `from_pretrained()` method!

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("your-username/my-awesome-model")
```

You can do a lot more with the Hub API:
- Push a model from a local repository
- Update specific files without re-uploading everything
- Add model cards to document the model's abilities, limitations, known biases, etc.

See [the documentation](https://huggingface.co/docs/huggingface_hub/how-to-upstream) for a complete tutorial on this, or check out the advanced [Chapter 4](/course/chapter4).

## Encoding text[[encoding-text]]

Transformer models handle text by turning the inputs into numbers. Here we will look at exactly what happens when your text is processed by the tokenizer. We've already seen in [Chapter 1](/course/chapter1) that tokenizers split the text into tokens and then convert these tokens into numbers. We can see this conversion through a simple tokenizer:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
```

```python out
{
    "input_ids": [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102],
    "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
```

We get a dictionary with the following fields:
- input_ids: numerical representations of your tokens
- token_type_ids: these tell the model which part of the input is sentence A and which is sentence B (discussed more in the next section)
- attention_mask: this indicates which tokens should be attended to and which should not (discussed more in a bit)

We can decode the input IDs to get back the original text:

```py
tokenizer.decode(encoded_input["input_ids"])
```

```python out
"[CLS] Hello, I'm a single sentence! [SEP]"
```

You'll notice that the tokenizer has added special tokens — `[CLS]` and `[SEP]` — required by the model. Not all models need special tokens; they're utilized when a model was pretrained with them, in which case the tokenizer needs to add them as that model expects these tokens.

You can encode multiple sentences at once, either by batching them together (we'll discuss this soon) or by passing a list:

```py
encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)
```

```python out
{
    "input_ids": [
        [101, 1731, 1132, 1128, 136, 102],
        [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102],
    ],
    "token_type_ids": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
}
```

Note that when passing multiple sentences, the tokenizer returns a list for each sentence for each dictionary value. We can also ask the tokenizer to return tensors directly from PyTorch:

```py
encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)
```

```python out
{
    "input_ids": tensor(
        [
            [101, 1731, 1132, 1128, 136, 102],
            [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102],
        ]
    ),
    "token_type_ids": tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
}
```

But there's a problem: the two lists don't have the same length! Arrays and tensors need to be rectangular, so we can't simply convert these lists to a PyTorch tensor (or NumPy array). The tokenizer provides an option for that: padding.

### Padding inputs[[padding-inputs]]

If we ask the tokenizer to pad the inputs, it will make all sentences the same length by adding a special padding token to the sentences that are shorter than the longest one:

```py
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)
```

```python out
{
    "input_ids": tensor(
        [
            [101, 1731, 1132, 1128, 136, 102, 0, 0, 0, 0],
            [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102],
        ]
    ),
    "token_type_ids": tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
}
```

Now we have rectangular tensors! Note that the padding tokens have been encoded into input IDs with ID 0, and they have an attention mask value of 0 as well. This is because those padding tokens shouldn't be analyzed by the model: they're not part of the actual sentence.

### Truncating inputs[[truncating-inputs]]

The tensors might get too big to be processed by the model. For instance, BERT was only pretrained with sequences up to 512 tokens, so it cannot process longer sequences. If you have sequences longer than the model can handle, you'll need to truncate them with the `truncation` parameter:

```py
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])
```

```python out
[
    101,
    1188,
    1110,
    170,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1505,
    1179,
    5650,
    119,
    102,
]
```

By combining the padding and truncation arguments, you can make sure your tensors have the exact size you need:

```py
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)
```

```python out
{
    "input_ids": tensor([[101, 1731, 1132, 1128, 102], [101, 1045, 1005, 1049, 102]]),
    "token_type_ids": tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
}
```

### Adding special tokens

Special tokens (or at least the concept of them) is particularly important to BERT and derived models. These tokens are added to better represent the sentence boundaries, such as the beginning of a sentence (`[CLS]`) or separator between sentences (`[SEP]`). Let's look at a simple example:

```py
encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])
```

```python out
[101, 1731, 1132, 1128, 136, 102]
"[CLS] How are you? [SEP]"
```

These special tokens are automatically added by the tokenizer. Not all models need special tokens; they are primarily used when a model was pretrained with them, in which case the tokenizer will add them since the model expects them.

### Why is all of this necessary?

Here's a concrete example. Consider these encoded sequences:

```py
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
```

Once tokenized, we have:

```python
encoded_sequences = [
    [
        101,
        1045,
        1005,
        2310,
        2042,
        3403,
        2005,
        1037,
        17662,
        12172,
        2607,
        2026,
        2878,
        2166,
        1012,
        102,
    ],
    [101, 1045, 5223, 2023, 2061, 2172, 999, 102],
]
```

This is a list of encoded sequences: a list of lists. Tensors only accept rectangular shapes (think matrices). This "array" is already of rectangular shape, so converting it to a tensor is easy:

```py
import torch

model_inputs = torch.tensor(encoded_sequences)
```

### Using the tensors as inputs to the model[[using-the-tensors-as-inputs-to-the-model]]

Making use of the tensors with the model is extremely simple — we just call the model with the inputs:

```py
output = model(model_inputs)
```

While the model accepts a lot of different arguments, only the input IDs are necessary. We'll explain what the other arguments do and when they are required later, 
but first we need to take a closer look at the tokenizers that build the inputs that a Transformer model can understand.


---

<!-- Section 2.4 -->

<FrameworkSwitchCourse {fw} />

# Tokenizers[[tokenizers]]

<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section4_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter2/section4_pt.ipynb"},
]} />

<Youtube id="VFp38yj8h3A"/>

Tokenizers are one of the core components of the NLP pipeline. They serve one purpose: to translate text into data that can be processed by the model. Models can only process numbers, so tokenizers need to convert our text inputs to numerical data. In this section, we'll explore exactly what happens in the tokenization pipeline. 

In NLP tasks, the data that is generally processed is raw text. Here's an example of such text:

```
Jim Henson was a puppeteer
```

However, models can only process numbers, so we need to find a way to convert the raw text to numbers. That's what the tokenizers do, and there are a lot of ways to go about this. The goal is to find the most meaningful representation — that is, the one that makes the most sense to the model — and, if possible, the smallest representation.

Let's take a look at some examples of tokenization algorithms, and try to answer some of the questions you may have about tokenization.

## Word-based[[word-based]]

<Youtube id="nhJxYji1aho"/>

The first type of tokenizer that comes to mind is _word-based_. It's generally very easy to set up and use with only a few rules, and it often yields decent results. For example, in the image below, the goal is to split the raw text into words and find a numerical representation for each of them:

<div class="flex justify-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/word_based_tokenization.svg" alt="An example of word-based tokenization."/>
  <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/word_based_tokenization-dark.svg" alt="An example of word-based tokenization."/>
</div>

There are different ways to split the text. For example, we could use whitespace to tokenize the text into words by applying Python's `split()` function:

```py
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
```

```python out
["Jim", "Henson", "was", "a", "puppeteer"]
```

There are also variations of word tokenizers that have extra rules for punctuation. With this kind of tokenizer, we can end up with some pretty large "vocabularies," where a vocabulary is defined by the total number of independent tokens that we have in our corpus.

Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. The model uses these IDs to identify each word.

If we want to completely cover a language with a word-based tokenizer, we'll need to have an identifier for each word in the language, which will generate a huge amount of tokens. For example, there are over 500,000 words in the English language, so to build a map from each word to an input ID we'd need to keep track of that many IDs. Furthermore, words like "dog" are represented differently from words like "dogs", and the model will initially have no way of knowing that "dog" and "dogs" are similar: it will identify the two words as unrelated. The same applies to other similar words, like "run" and "running", which the model will not see as being similar initially.

Finally, we need a custom token to represent words that are not in our vocabulary. This is known as the "unknown" token, often represented as "[UNK]" or "&lt;unk&gt;". It's generally a bad sign if you see that the tokenizer is producing a lot of these tokens, as it wasn't able to retrieve a sensible representation of a word and you're losing information along the way. The goal when crafting the vocabulary is to do it in such a way that the tokenizer tokenizes as few words as possible into the unknown token.

One way to reduce the amount of unknown tokens is to go one level deeper, using a _character-based_ tokenizer.

## Character-based[[character-based]]

<Youtube id="ssLq_EK2jLE"/>

Character-based tokenizers split the text into characters, rather than words. This has two primary benefits:

- The vocabulary is much smaller.
- There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.

But here too some questions arise concerning spaces and punctuation:

<div class="flex justify-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/character_based_tokenization.svg" alt="An example of character-based tokenization."/>
  <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/character_based_tokenization-dark.svg" alt="An example of character-based tokenization."/>
</div>

This approach isn't perfect either. Since the representation is now based on characters rather than words, one could argue that, intuitively, it's less meaningful: each character doesn't mean a lot on its own, whereas that is the case with words. However, this again differs according to the language; in Chinese, for example, each character carries more information than a character in a Latin language.

Another thing to consider is that we'll end up with a very large amount of tokens to be processed by our model: whereas a word would only be a single token with a word-based tokenizer, it can easily turn into 10 or more tokens when converted into characters.

To get the best of both worlds, we can use a third technique that combines the two approaches: *subword tokenization*.

## Subword tokenization[[subword-tokenization]]

<Youtube id="zHvTiHr506c"/>

Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.

For instance, "annoyingly" might be considered a rare word and could be decomposed into "annoying" and "ly". These are both likely to appear more frequently as standalone subwords, while at the same time the meaning of "annoyingly" is kept by the composite meaning of "annoying" and "ly".

Here is an example showing how a subword tokenization algorithm would tokenize the sequence "Let's do tokenization!":

<div class="flex justify-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/bpe_subword.svg" alt="A subword tokenization algorithm."/>
  <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/bpe_subword-dark.svg" alt="A subword tokenization algorithm."/>
</div>

These subwords end up providing a lot of semantic meaning: for instance, in the example above "tokenization" was split into "token" and "ization", two tokens that have a semantic meaning while being space-efficient (only two tokens are needed to represent a long word). This allows us to have relatively good coverage with small vocabularies, and close to no unknown tokens.

This approach is especially useful in agglutinative languages such as Turkish, where you can form (almost) arbitrarily long complex words by stringing together subwords.

### And more![[and-more]]

Unsurprisingly, there are many more techniques out there. To name a few:

- Byte-level BPE, as used in GPT-2
- WordPiece, as used in BERT
- SentencePiece or Unigram, as used in several multilingual models

You should now have sufficient knowledge of how tokenizers work to get started with the API.

## Loading and saving[[loading-and-saving]]

Loading and saving tokenizers is as simple as it is with models. Actually, it's based on the same two methods: `from_pretrained()` and `save_pretrained()`. These methods will load or save the algorithm used by the tokenizer (a bit like the *architecture* of the model) as well as its vocabulary (a bit like the *weights* of the model).

Loading the BERT tokenizer trained with the same checkpoint as BERT is done the same way as loading the model, except we use the `BertTokenizer` class:

```py
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

Similar to `AutoModel`, the `AutoTokenizer` class will grab the proper tokenizer class in the library based on the checkpoint name, and can be used directly with any checkpoint:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

We can now use the tokenizer as shown in the previous section:

```python
tokenizer("Using a Transformer network is simple")
```

```python out
{
    "input_ids": [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
    "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],
}
```

Saving a tokenizer is identical to saving a model:

```py
tokenizer.save_pretrained("directory_on_my_computer")
```

We'll talk more about `token_type_ids` in [Chapter 3](/course/chapter3), and we'll explain the `attention_mask` key a little later. First, let's see how the `input_ids` are generated. To do this, we'll need to look at the intermediate methods of the tokenizer.

## Encoding[[encoding]]

<Youtube id="Yffk5aydLzg"/>

Translating text to numbers is known as _encoding_. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.

As we've seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called *tokens*. There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.

The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model. To do this, the tokenizer has a *vocabulary*, which is the part we download when we instantiate it with the `from_pretrained()` method. Again, we need to use the same vocabulary used when the model was pretrained.

To get a better understanding of the two steps, we'll explore them separately. Note that we will use some methods that perform parts of the tokenization pipeline separately to show you the intermediate results of those steps, but in practice, you should call the tokenizer directly on your inputs (as shown in the section 2).

### Tokenization[[tokenization]]

The tokenization process is done by the `tokenize()` method of the tokenizer:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

The output of this method is a list of strings, or tokens:

```python out
["Using", "a", "transform", "##er", "network", "is", "simple"]
```

This tokenizer is a subword tokenizer: it splits the words until it obtains tokens that can be represented by its vocabulary. That's the case here with `transformer`, which is split into two tokens: `transform` and `##er`.

### From tokens to input IDs[[from-tokens-to-input-ids]]

The conversion to input IDs is handled by the `convert_tokens_to_ids()` tokenizer method:

```py
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```python out
[7993, 170, 11303, 1200, 2443, 1110, 3014]
```

These outputs, once converted to the appropriate framework tensor, can then be used as inputs to a model as seen earlier in this chapter.

> [!TIP]
> ✏️ **Try it out!** Replicate the two last steps (tokenization and conversion to input IDs) on the input sentences we used in section 2 ("I've been waiting for a HuggingFace course my whole life." and "I hate this so much!"). Check that you get the same input IDs we got earlier!

## Decoding[[decoding]]

*Decoding* is going the other way around: from vocabulary indices, we want to get a string. This can be done with the `decode()` method as follows:

```py
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

```python out
"Using a Transformer network is simple"
```

Note that the `decode` method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence. This behavior will be extremely useful when we use models that predict new text (either text generated from a prompt, or for sequence-to-sequence problems like translation or summarization).

By now you should understand the atomic operations a tokenizer can handle: tokenization, conversion to IDs, and converting IDs back to a string. However, we've just scraped the tip of the iceberg. In the following section, we'll take our approach to its limits and take a look at how to overcome them.


---

<!-- Section 2.5 -->

<FrameworkSwitchCourse {fw} />

# Handling multiple sequences[[handling-multiple-sequences]]

<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section5_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter2/section5_pt.ipynb"},
]} />

<Youtube id="M6adb1j2jPI"/>

In the previous section, we explored the simplest of use cases: doing inference on a single sequence of a small length. However, some questions emerge already:

- How do we handle multiple sequences?
- How do we handle multiple sequences *of different lengths*?
- Are vocabulary indices the only inputs that allow a model to work well?
- Is there such a thing as too long a sequence?

Let's see what kinds of problems these questions pose, and how we can solve them using the 🤗 Transformers API.

## Models expect a batch of inputs[[models-expect-a-batch-of-inputs]]

In the previous exercise you saw how sequences get translated into lists of numbers. Let's convert this list of numbers to a tensor and send it to the model:

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# This line will fail.
model(input_ids)
```

```python out
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

Oh no! Why did this fail? We followed the steps from the pipeline in section 2.

The problem is that we sent a single sequence to the model, whereas 🤗 Transformers models expect multiple sentences by default. Here we tried to do everything the tokenizer did behind the scenes when we applied it to a `sequence`. But if you look closely, you'll see that the tokenizer didn't just convert the list of input IDs into a tensor, it added a dimension on top of it:

```py
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

```python out
tensor(
    [
        [
            101,
            1045,
            1005,
            2310,
            2042,
            3403,
            2005,
            1037,
            17662,
            12172,
            2607,
            2026,
            2878,
            2166,
            1012,
            102,
        ]
    ]
)
```

Let's try again and add a new dimension:

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

We print the input IDs as well as the resulting logits — here's the output:

```python out
Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]
```

*Batching* is the act of sending multiple sentences through the model, all at once. If you only have one sentence, you can just build a batch with a single sequence: 

```
batched_ids = [ids, ids]
```

This is a batch of two identical sequences!

> [!TIP]
> ✏️ **Try it out!** Convert this `batched_ids` list into a tensor and pass it through your model. Check that you obtain the same logits as before (but twice)!

Batching allows the model to work when you feed it multiple sentences. Using multiple sequences is just as simple as building a batch with a single sequence. There's a second issue, though. When you're trying to batch together two (or more) sentences, they might be of different lengths. If you've ever worked with tensors before, you know that they need to be of rectangular shape, so you won't be able to convert the list of input IDs into a tensor directly. To work around this problem, we usually *pad* the inputs.

## Padding the inputs[[padding-the-inputs]]

The following list of lists cannot be converted to a tensor:

```py no-format
batched_ids = [[200, 200, 200], [200, 200]]
```

In order to work around this, we'll use *padding* to make our tensors have a rectangular shape. Padding makes sure all our sentences have the same length by adding a special word called the *padding token* to the sentences with fewer values. For example, if you have 10 sentences with 10 words and 1 sentence with 20 words, padding will ensure all the sentences have 20 words. In our example, the resulting tensor looks like this:

```py no-format
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

The padding token ID can be found in `tokenizer.pad_token_id`. Let's use it and send our two sentences through the model individually and batched together:

```py no-format
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

```python out
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
```

There's something wrong with the logits in our batched predictions: the second row should be the same as the logits for the second sentence, but we've got completely different values!

This is because the key feature of Transformer models is attention layers that *contextualize* each token. These will take into account the padding tokens since they attend to all of the tokens of a sequence. To get the same result when passing individual sentences of different lengths through the model or when passing a batch with the same sentences and padding applied, we need to tell those attention layers to ignore the padding tokens. This is done by using an attention mask.

## Attention masks[[attention-masks]]

*Attention masks* are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).

Let's complete the previous example with an attention mask:

```py no-format
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

```python out
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
```

Now we get the same logits for the second sentence in the batch.

Notice how the last value of the second sequence is a padding ID, which is a 0 value in the attention mask.

> [!TIP]
> ✏️ **Try it out!** Apply the tokenization manually on the two sentences used in section 2 ("I've been waiting for a HuggingFace course my whole life." and "I hate this so much!"). Pass them through the model and check that you get the same logits as in section 2. Now batch them together using the padding token, then create the proper attention mask. Check that you obtain the same results when going through the model!

## Longer sequences[[longer-sequences]]

With Transformer models, there is a limit to the lengths of the sequences we can pass the models. Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences. There are two solutions to this problem:

- Use a model with a longer supported sequence length.
- Truncate your sequences.

Models have different supported sequence lengths, and some specialize in handling very long sequences. [Longformer](https://huggingface.co/docs/transformers/model_doc/longformer) is one example, and another is [LED](https://huggingface.co/docs/transformers/model_doc/led). If you're working on a task that requires very long sequences, we recommend you take a look at those models.

Otherwise, we recommend you truncate your sequences by specifying the `max_sequence_length` parameter:

```py
sequence = sequence[:max_sequence_length]
```


---

<!-- Section 2.6 -->

<FrameworkSwitchCourse {fw} />

# Putting it all together[[putting-it-all-together]]

<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter2/section6_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter2/section6_pt.ipynb"},
]} />

In the last few sections, we've been trying our best to do most of the work by hand. We've explored how tokenizers work and looked at tokenization, conversion to input IDs, padding, truncation, and attention masks.

However, as we saw in section 2, the 🤗 Transformers API can handle all of this for us with a high-level function that we'll dive into here. When you call your `tokenizer` directly on the sentence, you get back inputs that are ready to pass through your model:

```py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

Here, the `model_inputs` variable contains everything that's necessary for a model to operate well. For DistilBERT, that includes the input IDs as well as the attention mask. Other models that accept additional inputs will also have those output by the `tokenizer` object.

As we'll see in some examples below, this method is very powerful. First, it can tokenize a single sequence:

```py
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

It also handles multiple sequences at a time, with no change in the API:

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
```

It can pad according to several objectives:

```py
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

It can also truncate sequences:

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

The `tokenizer` object can handle the conversion to specific framework tensors, which can then be directly sent to the model. For example, in the following code sample we are prompting the tokenizer to return tensors from the different frameworks — `"pt"` returns PyTorch tensors and `"np"` returns NumPy arrays:

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## Special tokens[[special-tokens]]

If we take a look at the input IDs returned by the tokenizer, we will see they are a tiny bit different from what we had earlier:

```py
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

```python out
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
```

One token ID was added at the beginning, and one at the end. Let's decode the two sequences of IDs above to see what this is about:

```py
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```

```python out
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"

"i've been waiting for a huggingface course my whole life."
```

The tokenizer added the special word `[CLS]` at the beginning and the special word `[SEP]` at the end. This is because the model was pretrained with those, so to get the same results for inference we need to add them as well. Note that some models don't add special words, or add different ones; models may also add these special words only at the beginning, or only at the end. In any case, the tokenizer knows which ones are expected and will deal with this for you.

## Wrapping up: From tokenizer to model[[wrapping-up-from-tokenizer-to-model]]

Now that we've seen all the individual steps the `tokenizer` object uses when applied on texts, let's see one final time how it can handle multiple sequences (padding!), very long sequences (truncation!), and multiple types of tensors with its main API:

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```


---

<!-- Section 2.7 -->

# Basic usage completed![[basic-usage-completed]]

<CourseFloatingBanner
    chapter={2}
    classNames="absolute z-10 right-0 top-0"
/>

Great job following the course up to here! To recap, in this chapter you:

- Learned the basic building blocks of a Transformer model.
- Learned what makes up a tokenization pipeline.
- Saw how to use a Transformer model in practice.
- Learned how to leverage a tokenizer to convert text to tensors that are understandable by the model.
- Set up a tokenizer and a model together to get from text to predictions.
- Learned the limitations of input IDs, and learned about attention masks.
- Played around with versatile and configurable tokenizer methods.

From now on, you should be able to freely navigate the 🤗 Transformers docs: the vocabulary will sound familiar, and you've already seen the methods that you'll use the majority of the time.


---

<!-- Section 2.8 -->

# Optimized Inference Deployment

In this section, we'll explore advanced frameworks for optimizing LLM deployments: Text Generation Inference (TGI), vLLM, and llama.cpp. These applications are primarily used in production environments to serve LLMs to users. This section focuses on how to deploy these frameworks in production rather than how to use them for inference on a single machine.

We'll cover how these tools maximize inference efficiency and simplify production deployments of Large Language Models.

## Framework Selection Guide

TGI, vLLM, and llama.cpp serve similar purposes but have distinct characteristics that make them better suited for different use cases. Let's look at the key differences between them, focusing on performance and integration.

### Memory Management and Performance

**TGI** is designed to be stable and predictable in production, using fixed sequence lengths to keep memory usage consistent. TGI manages memory using Flash Attention 2 and continuous batching techniques. This means it can process attention calculations very efficiently and keep the GPU busy by constantly feeding it work. The system can move parts of the model between CPU and GPU when needed, which helps handle larger models. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/flash-attn.png" alt="Flash Attention" />

<Tip title="How Flash Attention Works">

Flash Attention is a technique that optimizes the attention mechanism in transformer models by addressing memory bandwidth bottlenecks. As discussed earlier in [Chapter 1.8](/course/chapter1/8), the attention mechanism has quadratic complexity and memory usage, making it inefficient for long sequences.

The key innovation is in how it manages memory transfers between High Bandwidth Memory (HBM) and faster SRAM cache. Traditional attention repeatedly transfers data between HBM and SRAM, creating bottlenecks by leaving the GPU idle. Flash Attention loads data once into SRAM and performs all calculations there, minimizing expensive memory transfers. 

While the benefits are most significant during training, Flash Attention's reduced VRAM usage and improved efficiency make it valuable for inference as well, enabling faster and more scalable LLM serving.

</Tip>

**vLLM** takes a different approach by using PagedAttention. Just like how a computer manages its memory in pages, vLLM splits the model's memory into smaller blocks. This clever system means it can handle different-sized requests more flexibly and doesn't waste memory space. It's particularly good at sharing memory between different requests and reduces memory fragmentation, which makes the whole system more efficient.

<Tip title="How PagedAttention Works">

PagedAttention is a technique that addresses another critical bottleneck in LLM inference: KV cache memory management. As discussed in [Chapter 1.8](/course/chapter1/8), during text generation, the model stores attention keys and values (KV cache) for each generated token to reduce redundant computations. The KV cache can become enormous, especially with long sequences or multiple concurrent requests.

vLLM's key innovation lies in how it manages this cache:

1. **Memory Paging**: Instead of treating the KV cache as one large block, it's divided into fixed-size "pages" (similar to virtual memory in operating systems).
2. **Non-contiguous Storage**: Pages don't need to be stored contiguously in GPU memory, allowing for more flexible memory allocation.
3. **Page Table Management**: A page table tracks which pages belong to which sequence, enabling efficient lookup and access.
4. **Memory Sharing**: For operations like parallel sampling, pages storing the KV cache for the prompt can be shared across multiple sequences.

The PagedAttention approach can lead to up to 24x higher throughput compared to traditional methods, making it a game-changer for production LLM deployments. If you want to go really deep into how PagedAttention works, you can read the [the guide from the vLLM documentation](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html).

</Tip>

**llama.cpp** is a highly optimized C/C++ implementation originally designed for running LLaMA models on consumer hardware. It focuses on CPU efficiency with optional GPU acceleration and is ideal for resource-constrained environments. llama.cpp uses quantization techniques to reduce model size and memory requirements while maintaining good performance. It implements optimized kernels for various CPU architectures and supports basic KV cache management for efficient token generation.

<Tip title="How llama.cpp Quantization Works">

Quantization in llama.cpp reduces the precision of model weights from 32-bit or 16-bit floating point to lower precision formats like 8-bit integers (INT8), 4-bit, or even lower. This significantly reduces memory usage and improves inference speed with minimal quality loss.

Key quantization features in llama.cpp include:
1. **Multiple Quantization Levels**: Supports 8-bit, 4-bit, 3-bit, and even 2-bit quantization
2. **GGML/GGUF Format**: Uses custom tensor formats optimized for quantized inference
3. **Mixed Precision**: Can apply different quantization levels to different parts of the model
4. **Hardware-Specific Optimizations**: Includes optimized code paths for various CPU architectures (AVX2, AVX-512, NEON)

This approach enables running billion-parameter models on consumer hardware with limited memory, making it perfect for local deployments and edge devices.

</Tip>

### Deployment and Integration

Let's move on to the deployment and integration differences between the frameworks.

**TGI** excels in enterprise-level deployment with its production-ready features. It comes with built-in Kubernetes support and includes everything you need for running in production, like monitoring through Prometheus and Grafana, automatic scaling, and comprehensive safety features. The system also includes enterprise-grade logging and various protective measures like content filtering and rate limiting to keep your deployment secure and stable.

**vLLM** takes a more flexible, developer-friendly approach to deployment. It's built with Python at its core and can easily replace OpenAI's API in your existing applications. The framework focuses on delivering raw performance and can be customized to fit your specific needs. It works particularly well with Ray for managing clusters, making it a great choice when you need high performance and adaptability.

**llama.cpp** prioritizes simplicity and portability. Its server implementation is lightweight and can run on a wide range of hardware, from powerful servers to consumer laptops and even some high-end mobile devices. With minimal dependencies and a simple C/C++ core, it's easy to deploy in environments where installing Python frameworks would be challenging. The server provides an OpenAI-compatible API while maintaining a much smaller resource footprint than other solutions.

## Getting Started

Let's explore how to use these frameworks for deploying LLMs, starting with installation and basic setup.

### Installation and Basic Setup

<hfoptions id="inference-frameworks" >

<hfoption value="tgi" label="TGI">

TGI is easy to install and use, with deep integration into the Hugging Face ecosystem.

First, launch the TGI server using Docker:

```sh
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
```

Then interact with it using Hugging Face's InferenceClient:

```python
from huggingface_hub import InferenceClient

# Initialize client pointing to TGI endpoint
client = InferenceClient(
    model="http://localhost:8080",  # URL to the TGI server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

Alternatively, you can use the OpenAI client:

```python
from openai import OpenAI

# Initialize client pointing to TGI endpoint
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Make sure to include /v1
    api_key="not-needed",  # TGI doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

</hfoption>

<hfoption value="llama.cpp" label="llama.cpp">

llama.cpp is easy to install and use, requiring minimal dependencies and supporting both CPU and GPU inference.

First, install and build llama.cpp:

```sh
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the project
make

# Download the SmolLM2-1.7B-Instruct-GGUF model
curl -L -O https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.Q4_K_M.gguf
```

Then, launch the server (with OpenAI API compatibility):

```sh
# Start the server
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \
    --n-gpu-layers 0  # Set to a higher number to use GPU
```

Interact with the server using Hugging Face's InferenceClient:

```python
from huggingface_hub import InferenceClient

# Initialize client pointing to llama.cpp server
client = InferenceClient(
    model="http://localhost:8080/v1",  # URL to the llama.cpp server
    token="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

Alternatively, you can use the OpenAI client:

```python
from openai import OpenAI

# Initialize client pointing to llama.cpp server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Chat completion
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",  # Model identifier can be anything as server only loads one model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

</hfoption>

<hfoption value="vllm" label="vLLM">

vLLM is easy to install and use, with both OpenAI API compatibility and a native Python interface.

First, launch the vLLM OpenAI-compatible server:

```sh
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Then interact with it using Hugging Face's InferenceClient:

```python
from huggingface_hub import InferenceClient

# Initialize client pointing to vLLM endpoint
client = InferenceClient(
    model="http://localhost:8000/v1",  # URL to the vLLM server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

Alternatively, you can use the OpenAI client:

```python
from openai import OpenAI

# Initialize client pointing to vLLM endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

</hfoption>

</hfoptions>

### Basic Text Generation

Let's look at examples of text generation with the frameworks:

<hfoptions id="inference-frameworks" >

<hfoption value="tgi" label="TGI">

First, deploy TGI with advanced parameters:
```sh
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
    --max-total-tokens 4096 \
    --max-input-length 3072 \
    --max-batch-total-tokens 8192 \
    --waiting-served-ratio 1.2
```

Use the InferenceClient for flexible text generation:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Raw text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    details=True,
)
print(response.generated_text)
```

Or use the OpenAI client:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
)
print(response.choices[0].message.content)
```

</hfoption>

<hfoption value="llama.cpp" label="llama.cpp">

For llama.cpp, you can set advanced parameters when launching the server:

```sh
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \            # Context size
    --threads 8 \        # CPU threads to use
    --batch-size 512 \   # Batch size for prompt evaluation
    --n-gpu-layers 0     # GPU layers (0 = CPU only)
```

Use the InferenceClient:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080/v1", token="sk-no-key-required")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    details=True,
)
print(response.generated_text)
```

Or use the OpenAI client for generation with control over the sampling parameters:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

# Advanced parameters example
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Nucleus sampling probability
    frequency_penalty=0.5,  # Reduce repetition of frequent tokens
    presence_penalty=0.5,  # Reduce repetition by penalizing tokens already present
    max_tokens=200,  # Maximum generation length
)
print(response.choices[0].message.content)
```

You can also use llama.cpp's native library for even more control:

```python
# Using llama-cpp-python package for direct model access
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="smollm2-1.7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,  # Context window size
    n_threads=8,  # CPU threads
    n_gpu_layers=0,  # GPU layers (0 = CPU only)
)

# Format prompt according to the model's expected format
prompt = """<|im_start|>system
You are a creative storyteller.
<|im_end|>
<|im_start|>user
Write a creative story
<|im_end|>
<|im_start|>assistant
"""

# Generate response with precise parameter control
output = llm(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["<|im_end|>"],
)

print(output["choices"][0]["text"])
```

</hfoption>

<hfoption value="vllm" label="vLLM">

For advanced usage with vLLM, you can use the InferenceClient:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8000/v1")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    details=True,
)
print(response.generated_text)
```

You can also use the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=200,
)
print(response.choices[0].message.content)
```

vLLM also provides a native Python interface with fine-grained control:

```python
from vllm import LLM, SamplingParams

# Initialize the model with advanced parameters
llm = LLM(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
    block_size=16,
)

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    max_tokens=100,  # Maximum length
    presence_penalty=1.1,  # Reduce repetition
    frequency_penalty=1.1,  # Reduce repetition
    stop=["\n\n", "###"],  # Stop sequences
)

# Generate text
prompt = "Write a creative story"
outputs = llm.generate(prompt, sampling_params)
print(outputs[0].outputs[0].text)

# For chat-style interactions
chat_prompt = [
    {"role": "system", "content": "You are a creative storyteller."},
    {"role": "user", "content": "Write a creative story"},
]
formatted_prompt = llm.get_chat_template()(chat_prompt)  # Uses model's chat template
outputs = llm.generate(formatted_prompt, sampling_params)
print(outputs[0].outputs[0].text)
```

</hfoption>

</hfoptions>

## Advanced Generation Control

### Token Selection and Sampling

The process of generating text involves selecting the next token at each step. This selection process can be controlled through various parameters:

1. **Raw Logits**: The initial output probabilities for each token
2. **Temperature**: Controls randomness in selection (higher = more creative)
3. **Top-p (Nucleus) Sampling**: Filters to top tokens making up X% of probability mass
4. **Top-k Filtering**: Limits selection to k most likely tokens

Here's how to configure these parameters:

<hfoptions id="inference-frameworks" >

<hfoption value="tgi" label="TGI">

```python
client.generate(
    "Write a creative story",
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    top_k=50,  # Consider top 50 tokens
    max_new_tokens=100,  # Maximum length
    repetition_penalty=1.1,  # Reduce repetition
)
```

</hfoption>

<hfoption value="llama.cpp" label="llama.cpp">

```python
# Via OpenAI API compatibility
response = client.completions.create(
    model="smollm2-1.7b-instruct",  # Model name (can be any string for llama.cpp server)
    prompt="Write a creative story",
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    frequency_penalty=1.1,  # Reduce repetition
    presence_penalty=0.1,  # Reduce repetition
    max_tokens=100,  # Maximum length
)

# Via llama-cpp-python direct access
output = llm(
    "Write a creative story",
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=100,
    repeat_penalty=1.1,
)
```

</hfoption>

<hfoption value="vllm" label="vLLM">

```python
params = SamplingParams(
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    top_k=50,  # Consider top 50 tokens
    max_tokens=100,  # Maximum length
    presence_penalty=0.1,  # Reduce repetition
)
llm.generate("Write a creative story", sampling_params=params)
```

</hfoption>

</hfoptions>

### Controlling Repetition

Both frameworks provide ways to prevent repetitive text generation:

<hfoptions id="inference-frameworks" >

<hfoption value="tgi" label="TGI">

```python
client.generate(
    "Write a varied text",
    repetition_penalty=1.1,  # Penalize repeated tokens
    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
)
```

</hfoption>

<hfoption value="llama.cpp" label="llama.cpp">

```python
# Via OpenAI API
response = client.completions.create(
    model="smollm2-1.7b-instruct",
    prompt="Write a varied text",
    frequency_penalty=1.1,  # Penalize frequent tokens
    presence_penalty=0.8,  # Penalize tokens already present
)

# Via direct library
output = llm(
    "Write a varied text",
    repeat_penalty=1.1,  # Penalize repeated tokens
    frequency_penalty=0.5,  # Additional frequency penalty
    presence_penalty=0.5,  # Additional presence penalty
)
```

</hfoption>

<hfoption value="vllm" label="vLLM">

```python
params = SamplingParams(
    presence_penalty=0.1,  # Penalize token presence
    frequency_penalty=0.1,  # Penalize token frequency
)
```

</hfoption>

</hfoptions>

### Length Control and Stop Sequences

You can control generation length and specify when to stop:

<hfoptions id="inference-frameworks" >

<hfoption value="tgi" label="TGI">

```python
client.generate(
    "Generate a short paragraph",
    max_new_tokens=100,
    min_new_tokens=10,
    stop_sequences=["\n\n", "###"],
)
```

</hfoption>

<hfoption value="llama.cpp" label="llama.cpp">

```python
# Via OpenAI API
response = client.completions.create(
    model="smollm2-1.7b-instruct",
    prompt="Generate a short paragraph",
    max_tokens=100,
    stop=["\n\n", "###"],
)

# Via direct library
output = llm("Generate a short paragraph", max_tokens=100, stop=["\n\n", "###"])
```

</hfoption>

<hfoption value="vllm" label="vLLM">

```python
params = SamplingParams(
    max_tokens=100,
    min_tokens=10,
    stop=["###", "\n\n"],
    ignore_eos=False,
    skip_special_tokens=True,
)
```

</hfoption>

</hfoptions>

## Memory Management

Both frameworks implement advanced memory management techniques for efficient inference.

<hfoptions id="inference-frameworks" >

<hfoption value="tgi" label="TGI">

TGI uses Flash Attention 2 and continuous batching:

```sh
# Docker deployment with memory optimization
docker run --gpus all -p 8080:80 \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --max-batch-total-tokens 8192 \
    --max-input-length 4096
```

</hfoption>

<hfoption value="llama.cpp" label="llama.cpp">

llama.cpp uses quantization and optimized memory layout:

```sh
# Server with memory optimizations
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 2048 \               # Context size
    --threads 4 \           # CPU threads
    --n-gpu-layers 32 \     # Use more GPU layers for larger models
    --mlock \               # Lock memory to prevent swapping
    --cont-batching         # Enable continuous batching
```

For models too large for your GPU, you can use CPU offloading:

```sh
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --n-gpu-layers 20 \     # Keep first 20 layers on GPU
    --threads 8             # Use more CPU threads for CPU layers
```

</hfoption>

<hfoption value="vllm" label="vLLM">

vLLM uses PagedAttention for optimal memory management:

```python
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=8192,
    block_size=16,
)

llm = LLM(engine_args=engine_args)
```

</hfoption>

</hfoptions>

## Resources

- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python Repository](https://github.com/abetlen/llama-cpp-python)


---

<!-- Section 2.9 -->

<FrameworkSwitchCourse {fw} />

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={2}
    classNames="absolute z-10 right-0 top-0"
/>

### 1. What is the order of the language modeling pipeline?

<Question
	choices={[
		{
			text: "First, the model, which handles text and returns raw predictions. The tokenizer then makes sense of these predictions and converts them back to text when needed.",
			explain: "The model cannot understand text! The tokenizer must first tokenize the text and convert it to IDs so that it is understandable by the model."
		},
		{
			text: "First, the tokenizer, which handles text and returns IDs. The model handles these IDs and outputs a prediction, which can be some text.",
			explain: "The model's prediction cannot be text straight away. The tokenizer has to be used in order to convert the prediction back to text!"
		},
		{
			text: "The tokenizer handles text and returns IDs. The model handles these IDs and outputs a prediction. The tokenizer can then be used once again to convert these predictions back to some text.",
			explain: "The tokenizer can be used for both tokenizing and de-tokenizing.",
            correct: true
		}
	]}
/>

### 2. How many dimensions does the tensor output by the base Transformer model have, and what are they?

<Question
	choices={[
		{
			text: "2: The sequence length and the batch size",
			explain: "False! The tensor output by the model has a third dimension: hidden size."
		},
		{
			text: "2: The sequence length and the hidden size",
			explain: "False! All Transformer models handle batches, even with a single sequence; that would be a batch size of 1!"
		},
		{
			text: "3: The sequence length, the batch size, and the hidden size",
			explain: "Nicely done!",
            correct: true
		}
	]}
/>

### 3. Which of the following is an example of subword tokenization?

<Question
	choices={[
		{
			text: "WordPiece",
			explain: "Yes, that's one example of subword tokenization!",
            correct: true
		},
		{
			text: "Character-based tokenization",
			explain: "Character-based tokenization is not a type of subword tokenization."
		},
		{
			text: "Splitting on whitespace and punctuation",
			explain: "That's a word-based tokenization scheme!"
		},
		{
			text: "BPE",
			explain: "Yes, that's one example of subword tokenization!",
            correct: true
        },
		{
			text: "Unigram",
			explain: "Yes, that's one example of subword tokenization!",
            correct: true
        },
		{
			text: "None of the above",
			explain: "Wrong!"
        }
	]}
/>

### 4. What is a model head?

<Question
	choices={[
		{
			text: "A component of the base Transformer network that redirects tensors to their correct layers",
			explain: "There's no such component."
		},
		{
			text: "Also known as the self-attention mechanism, it adapts the representation of a token according to the other tokens of the sequence",
			explain: "The self-attention layer does contain attention \"heads,\" but these are not adaptation heads."
		},
		{
			text: "An additional component, usually made up of one or a few layers, to convert the transformer predictions to a task-specific output",
			explain: "That's right. Adaptation heads, also known simply as heads, come up in different forms: language modeling heads, question answering heads, sequence classification heads... ",
			correct: true
		} 
	]}
/>

### 5. What is an AutoModel?

<Question
	choices={[
		{
			text: "A model that automatically trains on your data",
			explain: "Are you mistaking this with our <a href='https://huggingface.co/autotrain'>AutoTrain</a> product?"
		},
		{
			text: "An object that returns the correct architecture based on the checkpoint",
			explain: "Exactly: the <code>AutoModel</code> only needs to know the checkpoint from which to initialize to return the correct architecture.",
			correct: true
		},
		{
			text: "A model that automatically detects the language used for its inputs to load the correct weights",
			explain: "While some checkpoints and models are capable of handling multiple languages, there are no built-in tools for automatic checkpoint selection according to language. You should head over to the <a href='https://huggingface.co/models'>Model Hub</a> to find the best checkpoint for your task!"
		} 
	]}
/>

### 6. What are the techniques to be aware of when batching sequences of different lengths together?

<Question
	choices={[
		{
			text: "Truncating",
			explain: "Yes, truncation is a correct way of evening out sequences so that they fit in a rectangular shape. Is it the only one, though?",
			correct: true
		},
		{
			text: "Returning tensors",
			explain: "While the other techniques allow you to return rectangular tensors, returning tensors isn't helpful when batching sequences together."
		},
		{
			text: "Padding",
			explain: "Yes, padding is a correct way of evening out sequences so that they fit in a rectangular shape. Is it the only one, though?",
			correct: true
		}, 
		{
			text: "Attention masking",
			explain: "Absolutely! Attention masks are of prime importance when handling sequences of different lengths. That's not the only technique to be aware of, however.",
			correct: true
		} 
	]}
/>

### 7. What is the point of applying a SoftMax function to the logits output by a sequence classification model?

<Question
	choices={[
		{
			text: "It softens the logits so that they're more reliable.",
			explain: "No, the SoftMax function does not affect the reliability of results."
		},
		{
			text: "It applies a lower and upper bound so that they're understandable.",
			explain: "The resulting values are bound between 0 and 1. That's not the only reason we use a SoftMax function, though.",
            correct: true
		},
		{
			text: "The total sum of the output is then 1, resulting in a possible probabilistic interpretation.",
			explain: "Correct! That's not the only reason we use a SoftMax function, though.",
            correct: true
		}
	]}
/>

### 8. What method is most of the tokenizer API centered around?

<Question
	choices={[
		{
			text: "<code>encode</code>, as it can encode text into IDs and IDs into predictions",
			explain: "Wrong! While the <code>encode</code> method does exist on tokenizers, it does not exist on models."
		},
		{
			text: "Calling the tokenizer object directly.",
			explain: "Exactly! The <code>__call__</code> method of the tokenizer is a very powerful method which can handle pretty much anything. It is also the method used to retrieve predictions from a model.",
			correct: true
		},
		{
			text: "<code>pad</code>",
			explain: "Wrong! Padding is very useful, but it's just one part of the tokenizer API."
		},
		{
			text: "<code>tokenize</code>",
			explain: "The <code>tokenize</code> method is arguably one of the most useful methods, but it isn't the core of the tokenizer API."
		}
	]}
/>

### 9. What does the `result` variable contain in this code sample?

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
```

<Question
	choices={[
		{
			text: "A list of strings, each string being a token",
			explain: "Absolutely! Convert this to IDs, and send them to a model!",
            correct: true
		},
		{
			text: "A list of IDs",
			explain: "Incorrect; that's what the <code>__call__</code> or <code>convert_tokens_to_ids</code> method is for!"
		},
		{
			text: "A string containing all of the tokens",
			explain: "This would be suboptimal, as the goal is to split the string into multiple tokens."
		}
	]}
/>

### 10. Is there something wrong with the following code?

```py
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```

<Question
	choices={[
		{
			text: "No, it seems correct.",
			explain: "Unfortunately, coupling a model with a tokenizer that was trained with a different checkpoint is rarely a good idea. The model was not trained to make sense out of this tokenizer's output, so the model output (if it can even run!) will not make any sense."
		},
		{
			text: "The tokenizer and model should always be from the same checkpoint.",
			explain: "Right!",
            correct: true
		},
		{
			text: "It's good practice to pad and truncate with the tokenizer as every input is a batch.",
			explain: "It's true that every model input needs to be a batch. However, truncating or padding this sequence wouldn't necessarily make sense as there is only one of it, and those are techniques to batch together a list of sentences."
		}
	]}
/>

