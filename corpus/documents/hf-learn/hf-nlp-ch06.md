# Hugging Face NLP Course — Chapter 6

Source: https://huggingface.co/learn/nlp-course/chapter6


---

<!-- Section 6.1 -->

# Introduction[[introduction]]

<CourseFloatingBanner
    chapter={6}
    classNames="absolute z-10 right-0 top-0"
/>

In [Chapter 3](/course/chapter3), we looked at how to fine-tune a model on a given task. When we do that, we use the same tokenizer that the model was pretrained with -- but what do we do when we want to train a model from scratch? In these cases, using a tokenizer that was pretrained on a corpus from another domain or language is typically suboptimal. For example, a tokenizer that's trained on an English corpus will perform poorly on a corpus of Japanese texts because the use of spaces and punctuation is very different in the two languages.

In this chapter, you will learn how to train a brand new tokenizer on a corpus of texts, so it can then be used to pretrain a language model. This will all be done with the help of the [🤗 Tokenizers](https://github.com/huggingface/tokenizers) library, which provides the "fast" tokenizers in the [🤗 Transformers](https://github.com/huggingface/transformers) library. We'll take a close look at the features that this library provides, and explore how the fast tokenizers differ from the "slow" versions.

Topics we will cover include:

* How to train a new tokenizer similar to the one used by a given checkpoint on a new corpus of texts
* The special features of fast tokenizers
* The differences between the three main subword tokenization algorithms used in NLP today
* How to build a tokenizer from scratch with the 🤗 Tokenizers library and train it on some data

The techniques introduced in this chapter will prepare you for the section in [Chapter 7](/course/chapter7/6) where we look at creating a language model for Python source code. Let's start by looking at what it means to "train" a tokenizer in the first place.

---

<!-- Section 6.2 -->

# Training a new tokenizer from an old one[[training-a-new-tokenizer-from-an-old-one]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section2.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section2.ipynb"},
]} />

If a language model is not available in the language you are interested in, or if your corpus is very different from the one your language model was trained on, you will most likely want to retrain the model from scratch using a tokenizer adapted to your data. That will require training a new tokenizer on your dataset. But what exactly does that mean? When we first looked at tokenizers in [Chapter 2](/course/chapter2), we saw that most Transformer models use a _subword tokenization algorithm_. To identify which subwords are of interest and occur most frequently in the corpus at hand, the tokenizer needs to take a hard look at all the texts in the corpus -- a process we call *training*. The exact rules that govern this training depend on the type of tokenizer used, and we'll go over the three main algorithms later in this chapter.

<Youtube id="DJimQynXZsQ"/>

> [!WARNING]
> ⚠️ Training a tokenizer is not the same as training a model! Model training uses stochastic gradient descent to make the loss a little bit smaller for each batch. It's randomized by nature (meaning you have to set some seeds to get the same results when doing the same training twice). Training a tokenizer is a statistical process that tries to identify which subwords are the best to pick for a given corpus, and the exact rules used to pick them depend on the tokenization algorithm. It's deterministic, meaning you always get the same results when training with the same algorithm on the same corpus.

## Assembling a corpus[[assembling-a-corpus]]

There's a very simple API in 🤗 Transformers that you can use to train a new tokenizer with the same characteristics as an existing one: `AutoTokenizer.train_new_from_iterator()`. To see this in action, let’s say we want to train GPT-2 from scratch, but in a language other than English. Our first task will be to gather lots of data in that language in a training corpus. To provide examples everyone will be able to understand, we won't use a language like Russian or Chinese here, but rather a specialized English language: Python code.

The [🤗 Datasets](https://github.com/huggingface/datasets) library can help us assemble a corpus of Python source code. We'll use the usual `load_dataset()` function to download and cache the [CodeSearchNet](https://huggingface.co/datasets/code_search_net) dataset. This dataset was created for the [CodeSearchNet challenge](https://wandb.ai/github/CodeSearchNet/benchmark) and contains millions of functions from open source libraries on GitHub in several programming languages. Here, we will load the Python part of this dataset:

```py
from datasets import load_dataset

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")
```

We can have a look at the training split to see which columns we have access to:

```py
raw_datasets["train"]
```

```python out
Dataset(
    {
        features: [
            "repository_name",
            "func_path_in_repository",
            "func_name",
            "whole_func_string",
            "language",
            "func_code_string",
            "func_code_tokens",
            "func_documentation_string",
            "func_documentation_tokens",
            "split_name",
            "func_code_url",
        ],
        num_rows: 412178,
    }
)
```

We can see the dataset separates docstrings from code and suggests a tokenization of both. Here. we'll just use the `whole_func_string` column to train our tokenizer. We can look at an example of one these functions by indexing into the `train` split:

```py
print(raw_datasets["train"][123456]["whole_func_string"])
```

which should print the following:

```out
def handle_simple_responses(
      self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
    """Accepts normal responses from the device.

    Args:
      timeout_ms: Timeout in milliseconds to wait for each response.
      info_cb: Optional callback for text sent from the bootloader.

    Returns:
      OKAY packet's message.
    """
    return self._accept_responses('OKAY', info_cb, timeout_ms=timeout_ms)
```

The first thing we need to do is transform the dataset into an _iterator_ of lists of texts -- for instance, a list of list of texts. Using lists of texts will enable our tokenizer to go faster (training on batches of texts instead of processing individual texts one by one), and it should be an iterator if we want to avoid having everything in memory at once. If your corpus is huge, you will want to take advantage of the fact that 🤗 Datasets does not load everything into RAM but stores the elements of the dataset on disk. 

Doing the following would create a list of lists of 1,000 texts each, but would load everything in memory:

```py
# Don't uncomment the following line unless your dataset is small!
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)]
```

Using a Python generator, we can avoid Python loading anything into memory until it's actually necessary. To create such a generator, you just to need to replace the brackets with parentheses:

```py
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```

This line of code doesn't fetch any elements of the dataset; it just creates an object you can use in a Python `for` loop. The texts will only be loaded when you need them (that is, when you're at the step of the `for` loop that requires them), and only 1,000 texts at a time will be loaded. This way you won't exhaust all your memory even if you are processing a huge dataset.

The problem with a generator object is that it can only be used once. So, instead of this giving us the list of the first 10 digits twice:

```py
gen = (i for i in range(10))
print(list(gen))
print(list(gen))
```

we get them once and then an empty list:

```python out
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[]
```

That's why we define a function that returns a generator instead:

```py
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()
```

You can also define your generator inside a `for` loop by using the `yield` statement:

```py
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
```

which will produce the exact same generator as before, but allows you to use more complex logic than you can in a list comprehension.

## Training a new tokenizer[[training-a-new-tokenizer]]

Now that we have our corpus in the form of an iterator of batches of texts, we are ready to train a new tokenizer. To do this, we first need to load the tokenizer we want to pair with our model (here, GPT-2):

```py
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Even though we are going to train a new tokenizer, it's a good idea to do this to avoid starting entirely from scratch. This way, we won't have to specify anything about the tokenization algorithm or the special tokens we want to use; our new tokenizer will be exactly the same as GPT-2, and the only thing that will change is the vocabulary, which will be determined by the training on our corpus.

First let's have a look at how this tokenizer would treat an example function:

```py
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
tokens
```

```python out
[
    "def",
    "Ġadd",
    "_",
    "n",
    "umbers",
    "(",
    "a",
    ",",
    "Ġb",
    "):",
    "Ċ",
    "Ġ",
    "Ġ",
    "Ġ",
    'Ġ"""',
    "Add",
    "Ġthe",
    "Ġtwo",
    "Ġnumbers",
    "Ġ`",
    "a",
    "`",
    "Ġand",
    "Ġ`",
    "b",
    "`",
    '."',
    '""',
    "Ċ",
    "Ġ",
    "Ġ",
    "Ġ",
    "Ġreturn",
    "Ġa",
    "Ġ+",
    "Ġb",
]
```

This tokenizer has a few special symbols, like `Ġ` and `Ċ`, which denote spaces and newlines, respectively. As we can see, this is not too efficient: the tokenizer returns individual tokens for each space, when it could group together indentation levels (since having sets of four or eight spaces is going to be very common in code). It also split the function name a bit weirdly, not being used to seeing words with the `_` character.

Let's train a new tokenizer and see if it solves those issues. For this, we'll use the method `train_new_from_iterator()`:

```py
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

This command might take a bit of time if your corpus is very large, but for this dataset of 1.6 GB of texts it's  blazing fast (1 minute 16 seconds on an AMD Ryzen 9 3900X CPU with 12 cores).

Note that `AutoTokenizer.train_new_from_iterator()` only works if the tokenizer you are using is a "fast" tokenizer. As you'll see in the next section, the 🤗 Transformers library contains two types of tokenizers: some are written purely in Python and others (the fast ones) are backed by the 🤗 Tokenizers library, which is written in the [Rust](https://www.rust-lang.org) programming language. Python is the language most often used for data science and deep learning applications, but when anything needs to be parallelized to be fast, it has to be written in another language. For instance, the matrix multiplications that are at the core of the model computation are written in CUDA, an optimized C library for GPUs.

Training a brand new tokenizer in pure Python would be excruciatingly slow, which is why we developed the 🤗 Tokenizers library. Note that just as you didn't have to learn the CUDA language to be able to execute your model on a batch of inputs on a GPU, you won't need to learn Rust to use a fast tokenizer. The 🤗 Tokenizers library provides Python bindings for many methods that internally call some piece of code in Rust; for example, to parallelize the training of your new tokenizer or, as we saw in [Chapter 3](/course/chapter3), the tokenization of a batch of inputs.

Most of the Transformer models have a fast tokenizer available (there are some exceptions that you can check [here](https://huggingface.co/transformers/#supported-frameworks)), and the `AutoTokenizer` API always selects the fast tokenizer for you if it's available. In the next section we'll take a look at some of the other special features fast tokenizers have, which will be really useful for tasks like token classification and question answering. Before diving into that, however, let's try our brand new tokenizer on the previous example:

```py
tokens = tokenizer.tokenize(example)
tokens
```

```python out
[
    "def",
    "Ġadd",
    "_",
    "numbers",
    "(",
    "a",
    ",",
    "Ġb",
    "):",
    "ĊĠĠĠ",
    'Ġ"""',
    "Add",
    "Ġthe",
    "Ġtwo",
    "Ġnumbers",
    "Ġ`",
    "a",
    "`",
    "Ġand",
    "Ġ`",
    "b",
    '`."""',
    "ĊĠĠĠ",
    "Ġreturn",
    "Ġa",
    "Ġ+",
    "Ġb",
]
```

Here we again see the special symbols `Ġ` and `Ċ` that denote spaces and newlines, but we can also see that our tokenizer learned some tokens that are highly specific to a corpus of Python functions: for example, there is a `ĊĠĠĠ` token that represents an indentation, and a `Ġ"""` token that represents the three quotes that start a docstring. The tokenizer also correctly split the function name on `_`. This is quite a compact representation; comparatively, using the plain English tokenizer on the same example will give us a longer sentence:

```py
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))
```

```python out
27
36
```

Let's look at another example:

```python
example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
tokenizer.tokenize(example)
```

```python out
[
    "class",
    "ĠLinear",
    "Layer",
    "():",
    "ĊĠĠĠ",
    "Ġdef",
    "Ġ__",
    "init",
    "__(",
    "self",
    ",",
    "Ġinput",
    "_",
    "size",
    ",",
    "Ġoutput",
    "_",
    "size",
    "):",
    "ĊĠĠĠĠĠĠĠ",
    "Ġself",
    ".",
    "weight",
    "Ġ=",
    "Ġtorch",
    ".",
    "randn",
    "(",
    "input",
    "_",
    "size",
    ",",
    "Ġoutput",
    "_",
    "size",
    ")",
    "ĊĠĠĠĠĠĠĠ",
    "Ġself",
    ".",
    "bias",
    "Ġ=",
    "Ġtorch",
    ".",
    "zeros",
    "(",
    "output",
    "_",
    "size",
    ")",
    "ĊĊĠĠĠ",
    "Ġdef",
    "Ġ__",
    "call",
    "__(",
    "self",
    ",",
    "Ġx",
    "):",
    "ĊĠĠĠĠĠĠĠ",
    "Ġreturn",
    "Ġx",
    "Ġ@",
    "Ġself",
    ".",
    "weights",
    "Ġ+",
    "Ġself",
    ".",
    "bias",
    "ĊĠĠĠĠ",
]
```

In addition to the token corresponding to an indentation, here we can also see a token for a double indentation: `ĊĠĠĠĠĠĠĠ`. The special Python words like `class`, `init`, `call`, `self`, and `return` are each tokenized as one token, and we can see that as well as splitting on `_` and `.` the tokenizer correctly splits even camel-cased names: `LinearLayer` is tokenized as `["ĠLinear", "Layer"]`.

## Saving the tokenizer[[saving-the-tokenizer]]

To make sure we can use it later, we need to save our new tokenizer. Like for models, this is done with the `save_pretrained()` method:

```py
tokenizer.save_pretrained("code-search-net-tokenizer")
```

This will create a new folder named *code-search-net-tokenizer*, which will contain all the files the tokenizer needs to be reloaded. If you want to share this tokenizer with your colleagues and friends, you can upload it to the Hub by logging into your account. If you're working in a notebook, there's a convenience function to help you with this:

```python
from huggingface_hub import notebook_login

notebook_login()
```

This will display a widget where you can enter your Hugging Face login credentials. If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

Once you've logged in, you can push your tokenizer by executing the following command:

```py
tokenizer.push_to_hub("code-search-net-tokenizer")
```

This will create a new repository in your namespace with the name `code-search-net-tokenizer`, containing the tokenizer file. You can then load the tokenizer from anywhere with the `from_pretrained()` method:

```py
# Replace "huggingface-course" below with your actual namespace to use your own tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
```

You're now all set for training a language model from scratch and fine-tuning it on your task at hand! We'll get to that in [Chapter 7](/course/chapter7), but first, in the rest of this chapter we'll take a closer look at fast tokenizers and explore in detail what actually happens when we call the method `train_new_from_iterator()`.


---

<!-- Section 6.3 -->

<FrameworkSwitchCourse {fw} />

# Fast tokenizers' special powers[[fast-tokenizers-special-powers]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section3_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section3_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section3_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section3_tf.ipynb"},
]} />

{/if}

In this section we will take a closer look at the capabilities of the tokenizers in 🤗 Transformers. Up to now we have only used them to tokenize inputs or decode IDs back into text, but tokenizers -- especially those backed by the 🤗 Tokenizers library -- can do a lot more. To illustrate these additional features, we will explore how to reproduce the results of the `token-classification` (that we called `ner`) and `question-answering` pipelines that we first encountered in [Chapter 1](/course/chapter1).

<Youtube id="g8quOxoqhHQ"/>

In the following discussion, we will often make the distinction between "slow" and "fast" tokenizers. Slow tokenizers are those written in Python inside the 🤗 Transformers library, while the fast versions are the ones provided by 🤗 Tokenizers, which are written in Rust. If you remember the table from [Chapter 5](/course/chapter5/3) that reported how long it took a fast and a slow tokenizer to tokenize the Drug Review Dataset, you should have an idea of why we call them fast and slow:

|               | Fast tokenizer | Slow tokenizer
:--------------:|:--------------:|:-------------:
`batched=True`  | 10.8s          | 4min41s
`batched=False` | 59.2s          | 5min3s

> [!WARNING]
> ⚠️ When tokenizing a single sentence, you won't always see a difference in speed between the slow and fast versions of the same tokenizer. In fact, the fast version might actually be slower! It's only when tokenizing lots of texts in parallel at the same time that you will be able to clearly see the difference.

## Batch encoding[[batch-encoding]]

<Youtube id="3umI3tm27Vw"/>

The output of a tokenizer isn't a simple Python dictionary; what we get is actually a special `BatchEncoding` object. It's a subclass of a dictionary (which is why we were able to index into that result without any problem before), but with additional methods that are mostly used by fast tokenizers.

Besides their parallelization capabilities, the key functionality of fast tokenizers is that they always keep track of the original span of texts the final tokens come from -- a feature we call *offset mapping*. This in turn unlocks features like mapping each word to the tokens it generated or mapping each character of the original text to the token it's inside, and vice versa.

Let's take a look at an example:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
```

As mentioned previously, we get a `BatchEncoding` object in the tokenizer's output:

```python out
<class 'transformers.tokenization_utils_base.BatchEncoding'>
```

Since the `AutoTokenizer` class picks a fast tokenizer by default, we can use the additional methods this `BatchEncoding` object provides. We have two ways to check if our tokenizer is a fast or a slow one. We can either check the attribute `is_fast` of the `tokenizer`:

```python
tokenizer.is_fast
```

```python out
True
```

or check the same attribute of our `encoding`:

```python
encoding.is_fast
```

```python out
True
```

Let's see what a fast tokenizer enables us to do. First, we can access the tokens without having to convert the IDs back to tokens:

```py
encoding.tokens()
```

```python out
[
    "[CLS]",
    "My",
    "name",
    "is",
    "S",
    "##yl",
    "##va",
    "##in",
    "and",
    "I",
    "work",
    "at",
    "Hu",
    "##gging",
    "Face",
    "in",
    "Brooklyn",
    ".",
    "[SEP]",
]
```

In this case the token at index 5 is `##yl`, which is part of the word "Sylvain" in the original sentence. We can also use the `word_ids()` method to get the index of the word each token comes from:

```py
encoding.word_ids()
```

```python out
[None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None]
```

We can see that the tokenizer's special tokens `[CLS]` and `[SEP]` are mapped to `None`, and then each token is mapped to the word it originates from. This is especially useful to determine if a token is at the start of a word or if two tokens are in the same word. We could rely on the `##` prefix for that, but it only works for BERT-like tokenizers; this method works for any type of tokenizer as long as it's a fast one. In the next chapter, we'll see how we can use this capability to apply the labels we have for each word properly to the tokens in tasks like named entity recognition (NER) and part-of-speech (POS) tagging. We can also use it to mask all the tokens coming from the same word in masked language modeling (a technique called _whole word masking_).

> [!TIP]
> The notion of what a word is complicated. For instance, does "I'll" (a contraction of "I will") count as one or two words? It actually depends on the tokenizer and the pre-tokenization operation it applies. Some tokenizers just split on spaces, so they will consider this as one word. Others use punctuation on top of spaces, so will consider it two words.
>
> ✏️ **Try it out!** Create a tokenizer from the `bert-base-cased` and `roberta-base` checkpoints and tokenize "81s" with them. What do you observe? What are the word IDs?

Similarly, there is a `sentence_ids()` method that we can use to map a token to the sentence it came from (though in this case, the `token_type_ids` returned by the tokenizer can give us the same information).

Lastly, we can map any word or token to characters in the original text, and vice versa, via the `word_to_chars()` or `token_to_chars()` and `char_to_word()` or `char_to_token()` methods. For instance, the `word_ids()` method told us that `##yl` is part of the word at index 3, but which word is it in the sentence? We can find out like this:

```py
start, end = encoding.word_to_chars(3)
example[start:end]
```

```python out
Sylvain
```

As we mentioned previously, this is all powered by the fact the fast tokenizer keeps track of the span of text each token comes from in a list of *offsets*. To illustrate their use, next we'll show you how to replicate the results of the `token-classification` pipeline manually.

> [!TIP]
> ✏️ **Try it out!** Create your own example text and see if you can understand which tokens are associated with word ID, and also how to extract the character spans for a single word. For bonus points, try using two sentences as input and see if the sentence IDs make sense to you.

## Inside the `token-classification` pipeline[[inside-the-token-classification-pipeline]]

In [Chapter 1](/course/chapter1) we got our first taste of applying NER -- where the task is to identify which parts of the text correspond to entities like persons, locations, or organizations -- with the 🤗 Transformers `pipeline()` function. Then, in [Chapter 2](/course/chapter2), we saw how a pipeline groups together the three stages necessary to get the predictions from a raw text: tokenization, passing the inputs through the model, and post-processing. The first two steps in the `token-classification` pipeline are the same as in any other pipeline, but the post-processing is a little more complex -- let's see how!

{#if fw === 'pt'}

<Youtube id="0E7ltQB7fM8"/>

{:else}

<Youtube id="PrX4CjrVnNc"/>

{/if}

### Getting the base results with the pipeline[[getting-the-base-results-with-the-pipeline]]

First, let's grab a token classification pipeline so we can get some results to compare manually. The model used by default is [`dbmdz/bert-large-cased-finetuned-conll03-english`](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english); it performs NER on sentences:

```py
from transformers import pipeline

token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[
    {"entity": "I-PER", "score": 0.9993828, "index": 4, "word": "S", "start": 11, "end": 12},
    {"entity": "I-PER", "score": 0.99815476, "index": 5, "word": "##yl", "start": 12, "end": 14},
    {"entity": "I-PER", "score": 0.99590725, "index": 6, "word": "##va", "start": 14, "end": 16},
    {"entity": "I-PER", "score": 0.9992327, "index": 7, "word": "##in", "start": 16, "end": 18},
    {"entity": "I-ORG", "score": 0.97389334, "index": 12, "word": "Hu", "start": 33, "end": 35},
    {"entity": "I-ORG", "score": 0.976115, "index": 13, "word": "##gging", "start": 35, "end": 40},
    {"entity": "I-ORG", "score": 0.98879766, "index": 14, "word": "Face", "start": 41, "end": 45},
    {
        "entity": "I-LOC",
        "score": 0.99321055,
        "index": 16,
        "word": "Brooklyn",
        "start": 49,
        "end": 57,
    },
]
```

The model properly identified each token generated by "Sylvain" as a person, each token generated by "Hugging Face" as an organization, and the token "Brooklyn" as a location. We can also ask the pipeline to group together the tokens that correspond to the same entity:

```py
from transformers import pipeline

token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[
    {"entity_group": "PER", "score": 0.9981694, "word": "Sylvain", "start": 11, "end": 18},
    {"entity_group": "ORG", "score": 0.97960204, "word": "Hugging Face", "start": 33, "end": 45},
    {"entity_group": "LOC", "score": 0.99321055, "word": "Brooklyn", "start": 49, "end": 57},
]
```

The `aggregation_strategy` picked will change the scores computed for each grouped entity. With `"simple"` the score is just the mean of the scores of each token in the given entity: for instance, the score of "Sylvain" is the mean of the scores we saw in the previous example for the tokens `S`, `##yl`, `##va`, and `##in`. Other strategies available are:

- `"first"`, where the score of each entity is the score of the first token of that entity (so for "Sylvain" it would be 0.993828, the score of the token `S`)
- `"max"`, where the score of each entity is the maximum score of the tokens in that entity (so for "Hugging Face" it would be 0.98879766, the score of "Face")
- `"average"`, where the score of each entity is the average of the scores of the words composing that entity (so for "Sylvain" there would be no difference from the `"simple"` strategy, but "Hugging Face" would have a score of 0.9819, the average of the scores for "Hugging", 0.975, and "Face", 0.98879)

Now let's see how to obtain these results without using the `pipeline()` function!

### From inputs to predictions[[from-inputs-to-predictions]]

{#if fw === 'pt'}

First we need to tokenize our input and pass it through the model. This is done exactly as in [Chapter 2](/course/chapter2); we instantiate the tokenizer and the model using the `AutoXxx` classes and then use them on our example:

```py
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
```

Since we're using `AutoModelForTokenClassification` here, we get one set of logits for each token in the input sequence:

```py
print(inputs["input_ids"].shape)
print(outputs.logits.shape)
```

```python out
torch.Size([1, 19])
torch.Size([1, 19, 9])
```

{:else}

First we need to tokenize our input and pass it through the model. This is done exactly as in [Chapter 2](/course/chapter2); we instantiate the tokenizer and the model using the `TFAutoXxx` classes and then use them on our example:

```py
from transformers import AutoTokenizer, TFAutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="tf")
outputs = model(**inputs)
```

Since we're using `TFAutoModelForTokenClassification` here, we get one set of logits for each token in the input sequence:

```py
print(inputs["input_ids"].shape)
print(outputs.logits.shape)
```

```python out
(1, 19)
(1, 19, 9)
```

{/if}

We have a batch with 1 sequence of 19 tokens and the model has 9 different labels, so the output of the model has a shape of 1 x 19 x 9. Like for the text classification pipeline, we use a softmax function to convert those logits to probabilities, and we take the argmax to get predictions (note that we can take the argmax on the logits because the softmax does not change the order):

{#if fw === 'pt'}

```py
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
```

{:else}

```py
import tensorflow as tf

probabilities = tf.math.softmax(outputs.logits, axis=-1)[0]
probabilities = probabilities.numpy().tolist()
predictions = tf.math.argmax(outputs.logits, axis=-1)[0]
predictions = predictions.numpy().tolist()
print(predictions)
```

{/if}

```python out
[0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
```

The `model.config.id2label` attribute contains the mapping of indexes to labels that we can use to make sense of the predictions:

```py
model.config.id2label
```

```python out
{
    0: "O",
    1: "B-MISC",
    2: "I-MISC",
    3: "B-PER",
    4: "I-PER",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-LOC",
    8: "I-LOC",
}
```

As we saw earlier, there are 9 labels: `O` is the label for the tokens that are not in any named entity (it stands for "outside"), and we then have two labels for each type of entity (miscellaneous, person, organization, and location). The label `B-XXX` indicates the token is at the beginning of an entity `XXX` and the label `I-XXX` indicates the token is inside the entity `XXX`. For instance, in the current example we would expect our model to classify the token `S` as `B-PER` (beginning of a person entity) and the tokens `##yl`, `##va` and `##in` as `I-PER` (inside a person entity). 

You might think the model was wrong in this case as it gave the label `I-PER` to all four of these tokens, but that's not entirely true. There are actually two formats for those `B-` and `I-` labels: *IOB1* and *IOB2*. The IOB2 format (in pink below), is the one we introduced whereas in the IOB1 format (in blue), the labels beginning with `B-` are only ever used to separate two adjacent entities of the same type. The model we are using was fine-tuned on a dataset using that format, which is why it assigns the label `I-PER` to the `S` token.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/IOB_versions.svg" alt="IOB1 vs IOB2 format"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/IOB_versions-dark.svg" alt="IOB1 vs IOB2 format"/>
</div>

With this map, we are ready to reproduce (almost entirely) the results of the first pipeline -- we can just grab the score and label of each token that was not classified as `O`:

```py
results = []
tokens = inputs.tokens()

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append({"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]})

print(results)
```

```python out
[
    {"entity": "I-PER", "score": 0.9993828, "index": 4, "word": "S"},
    {"entity": "I-PER", "score": 0.99815476, "index": 5, "word": "##yl"},
    {"entity": "I-PER", "score": 0.99590725, "index": 6, "word": "##va"},
    {"entity": "I-PER", "score": 0.9992327, "index": 7, "word": "##in"},
    {"entity": "I-ORG", "score": 0.97389334, "index": 12, "word": "Hu"},
    {"entity": "I-ORG", "score": 0.976115, "index": 13, "word": "##gging"},
    {"entity": "I-ORG", "score": 0.98879766, "index": 14, "word": "Face"},
    {"entity": "I-LOC", "score": 0.99321055, "index": 16, "word": "Brooklyn"},
]
```

This is very similar to what we had before, with one exception: the pipeline also gave us information about the `start` and `end` of each entity in the original sentence. This is where our offset mapping will come into play. To get the offsets, we just have to set `return_offsets_mapping=True` when we apply the tokenizer to our inputs:

```py
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]
```

```python out
[
    (0, 0),
    (0, 2),
    (3, 7),
    (8, 10),
    (11, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (19, 22),
    (23, 24),
    (25, 29),
    (30, 32),
    (33, 35),
    (35, 40),
    (41, 45),
    (46, 48),
    (49, 57),
    (57, 58),
    (0, 0),
]
```

Each tuple is the span of text corresponding to each token, where `(0, 0)` is reserved for the special tokens. We saw before that the token at index 5 is `##yl`, which has `(12, 14)` as offsets here. If we grab the corresponding slice in our example:


```py
example[12:14]
```

we get the proper span of text without the `##`:

```python out
yl
```

Using this, we can now complete the previous results:

```py
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)
```

```python out
[
    {"entity": "I-PER", "score": 0.9993828, "index": 4, "word": "S", "start": 11, "end": 12},
    {"entity": "I-PER", "score": 0.99815476, "index": 5, "word": "##yl", "start": 12, "end": 14},
    {"entity": "I-PER", "score": 0.99590725, "index": 6, "word": "##va", "start": 14, "end": 16},
    {"entity": "I-PER", "score": 0.9992327, "index": 7, "word": "##in", "start": 16, "end": 18},
    {"entity": "I-ORG", "score": 0.97389334, "index": 12, "word": "Hu", "start": 33, "end": 35},
    {"entity": "I-ORG", "score": 0.976115, "index": 13, "word": "##gging", "start": 35, "end": 40},
    {"entity": "I-ORG", "score": 0.98879766, "index": 14, "word": "Face", "start": 41, "end": 45},
    {
        "entity": "I-LOC",
        "score": 0.99321055,
        "index": 16,
        "word": "Brooklyn",
        "start": 49,
        "end": 57,
    },
]
```

This is the same as what we got from the first pipeline!

### Grouping entities[[grouping-entities]]

Using the offsets to determine the start and end keys for each entity is handy, but that information isn't strictly necessary. When we want to group the entities together, however, the offsets will save us a lot of messy code. For example, if we wanted to group together the tokens `Hu`, `##gging`, and `Face`, we could make special rules that say the first two should be attached while removing the `##`, and the `Face` should be added with a space since it does not begin with `##` -- but that would only work for this particular type of tokenizer. We would have to write another set of rules for a SentencePiece or a Byte-Pair-Encoding tokenizer (discussed later in this chapter).

With the offsets, all that custom code goes away: we just can take the span in the original text that begins with the first token and ends with the last token. So, in the case of the tokens `Hu`, `##gging`, and `Face`, we should start at character 33 (the beginning of `Hu`) and end before character 45 (the end of `Face`):

```py
example[33:45]
```

```python out
Hugging Face
```

To write the code that post-processes the predictions while grouping entities, we will group together entities that are consecutive and labeled with `I-XXX`, except for the first one, which can be labeled as `B-XXX` or `I-XXX` (so, we stop grouping an entity when we get a `O`, a new type of entity, or a `B-XXX` that tells us an entity of the same type is starting):

```py
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while idx < len(predictions) and model.config.id2label[predictions[idx]] == f"I-{label}":
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
```

And we get the same results as with our second pipeline!

```python out
[
    {"entity_group": "PER", "score": 0.9981694, "word": "Sylvain", "start": 11, "end": 18},
    {"entity_group": "ORG", "score": 0.97960204, "word": "Hugging Face", "start": 33, "end": 45},
    {"entity_group": "LOC", "score": 0.99321055, "word": "Brooklyn", "start": 49, "end": 57},
]
```

Another example of a task where these offsets are extremely useful is question answering. Diving into that pipeline, which we'll do in the next section, will also enable us to take a look at one last feature of the tokenizers in the 🤗 Transformers library: dealing with overflowing tokens when we truncate an input to a given length.


---

<!-- Section 6.4 -->

# Normalization and pre-tokenization[[normalization-and-pre-tokenization]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section4.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section4.ipynb"},
]} />

Before we dive more deeply into the three most common subword tokenization algorithms used with Transformer models (Byte-Pair Encoding [BPE], WordPiece, and Unigram), we'll first take a look at the preprocessing that each tokenizer applies to text. Here's a high-level overview of the steps in the tokenization pipeline:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg" alt="The tokenization pipeline.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline-dark.svg" alt="The tokenization pipeline.">
</div>

Before splitting a text into subtokens (according to its model), the tokenizer performs two steps: _normalization_ and _pre-tokenization_.

## Normalization[[normalization]]

<Youtube id="4IIC2jI9CaU"/>

The normalization step involves some general cleanup, such as removing needless whitespace, lowercasing, and/or removing accents. If you're familiar with [Unicode normalization](http://www.unicode.org/reports/tr15/) (such as NFC or NFKC), this is also something the tokenizer may apply.

The 🤗 Transformers `tokenizer` has an attribute called `backend_tokenizer` that provides access to the underlying tokenizer from the 🤗 Tokenizers library:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
```

```python out
<class 'tokenizers.Tokenizer'>
```

The `normalizer` attribute of the `tokenizer` object has a `normalize_str()` method that we can use to see how the normalization is performed:

```py
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
```

```python out
"hello how are u?"
```

In this example, since we picked the `bert-base-uncased` checkpoint, the normalization applied lowercasing and removed the accents. 

> [!TIP]
> ✏️ **Try it out!** Load a tokenizer from the `bert-base-cased` checkpoint and pass the same example to it. What are the main differences you can see between the cased and uncased versions of the tokenizer?

## Pre-tokenization[[pre-tokenization]]

<Youtube id="grlLV8AIXug"/>

As we will see in the next sections, a tokenizer cannot be trained on raw text alone. Instead, we first need to split the texts into small entities, like words. That's where the pre-tokenization step comes in. As we saw in [Chapter 2](/course/chapter2), a word-based tokenizer can simply split a raw text into words on whitespace and punctuation. Those words will be the boundaries of the subtokens the tokenizer can learn during its training.

To see how a fast tokenizer performs pre-tokenization, we can use the `pre_tokenize_str()` method of the `pre_tokenizer` attribute of the `tokenizer` object:

```py
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python out
[
    ("Hello", (0, 5)),
    (",", (5, 6)),
    ("how", (7, 10)),
    ("are", (11, 14)),
    ("you", (16, 19)),
    ("?", (19, 20)),
]
```

Notice how the tokenizer is already keeping track of the offsets, which is how it can give us the offset mapping we used in the previous section. Here the tokenizer ignores the two spaces and replaces them with just one, but the offset jumps between `are` and `you` to account for that.

Since we're using a BERT tokenizer, the pre-tokenization involves splitting on whitespace and punctuation. Other tokenizers can have different rules for this step. For example, if we use the GPT-2 tokenizer:

```py
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

it will split on whitespace and punctuation as well, but it will keep the spaces and replace them with a `Ġ` symbol, enabling it to recover the original spaces if we decode the tokens:

```python out
[
    ("Hello", (0, 5)),
    (",", (5, 6)),
    ("Ġhow", (6, 10)),
    ("Ġare", (10, 14)),
    ("Ġ", (14, 15)),
    ("Ġyou", (15, 19)),
    ("?", (19, 20)),
]
```

Also note that unlike the BERT tokenizer, this tokenizer does not ignore the double space.

For a last example, let's have a look at the T5 tokenizer, which is based on the SentencePiece algorithm:

```py
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

```python out
[("▁Hello,", (0, 6)), ("▁how", (7, 10)), ("▁are", (11, 14)), ("▁you?", (16, 20))]
```

Like the GPT-2 tokenizer, this one keeps spaces and replaces them with a specific token (`_`), but the T5 tokenizer only splits on whitespace, not punctuation. Also note that it added a space by default at the beginning of the sentence (before `Hello`) and ignored the double space between `are` and `you`.

Now that we've seen a little of how some different tokenizers process text, we can start to explore the underlying algorithms themselves. We'll begin with a quick look at the broadly widely applicable SentencePiece; then, over the next three sections, we'll examine how the three main algorithms used for subword tokenization work.

## SentencePiece[[sentencepiece]]

[SentencePiece](https://github.com/google/sentencepiece) is a tokenization algorithm for the preprocessing of text that you can use with any of the models we will see in the next three sections. It considers the text as a sequence of Unicode characters, and replaces spaces with a special character, `▁`. Used in conjunction with the Unigram algorithm (see [section 7](/course/chapter6/7)), it doesn't even require a pre-tokenization step, which is very useful for languages where the space character is not used (like Chinese or Japanese).

The other main feature of SentencePiece is *reversible tokenization*: since there is no special treatment of spaces, decoding the tokens is done simply by concatenating them and replacing the `_`s with spaces -- this results in the normalized text. As we saw earlier, the BERT tokenizer removes repeating spaces, so its tokenization is not reversible.

## Algorithm overview[[algorithm-overview]]

In the following sections, we'll dive into the three main subword tokenization algorithms: BPE (used by GPT-2 and others), WordPiece (used for example by BERT), and Unigram (used by T5 and others). Before we get started, here's a quick overview of how they each work. Don't hesitate to come back to this table after reading each of the next sections if it doesn't make sense to you yet.


Model | BPE | WordPiece | Unigram
:----:|:---:|:---------:|:------:
Training | Starts from a small vocabulary and learns rules to merge tokens |  Starts from a small vocabulary and learns rules to merge tokens | Starts from a large vocabulary and learns rules to remove tokens
Training step | Merges the tokens corresponding to the most common pair | Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent | Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus
Learns | Merge rules and a vocabulary | Just a vocabulary | A vocabulary with a score for each token
Encoding | Splits a word into characters and applies the merges learned during training | Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word | Finds the most likely split into tokens, using the scores learned during training

Now let's dive into BPE!


---

<!-- Section 6.5 -->

# Byte-Pair Encoding tokenization[[byte-pair-encoding-tokenization]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section5.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section5.ipynb"},
]} />

Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model. It's used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.

<Youtube id="HEikzVL-lZU"/>

> [!TIP]
> 💡 This section covers BPE in depth, going as far as showing a full implementation. You can skip to the end if you just want a general overview of the tokenization algorithm.

## Training algorithm[[training-algorithm]]

BPE training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words. As a very simple example, let's say our corpus uses these five words:

```
"hug", "pug", "pun", "bun", "hugs"
```

The base vocabulary will then be `["b", "g", "h", "n", "p", "s", "u"]`. For real-world cases, that base vocabulary will contain all the ASCII characters, at the very least, and probably some Unicode characters as well. If an example you are tokenizing uses a character that is not in the training corpus, that character will be converted to the unknown token. That's one reason why lots of NLP models are very bad at analyzing content with emojis, for instance.

> [!TIP]
> The GPT-2 and RoBERTa tokenizers (which are pretty similar) have a clever way to deal with this: they don't look at words as being written with Unicode characters, but with bytes. This way the base vocabulary has a small size (256), but every character you can think of will still be included and not end up being converted to the unknown token. This trick is called *byte-level BPE*.

After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning *merges*, which are rules to merge two elements of the existing vocabulary together into a new one. So, at the beginning these merges will create tokens with two characters, and then, as training progresses, longer subwords.

At any step during the tokenizer training, the BPE algorithm will search for the most frequent pair of existing tokens (by "pair," here we mean two consecutive tokens in a word). That most frequent pair is the one that will be merged, and we rinse and repeat for the next step.

Going back to our previous example, let's assume the words had the following frequencies:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

meaning `"hug"` was present 10 times in the corpus, `"pug"` 5 times, `"pun"` 12 times, `"bun"` 4 times, and `"hugs"` 5 times. We start the training by splitting each word into characters (the ones that form our initial vocabulary) so we can see each word as a list of tokens:

```
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

Then we look at pairs. The pair `("h", "u")` is present in the words `"hug"` and `"hugs"`, so 15 times total in the corpus. It's not the most frequent pair, though: that honor belongs to `("u", "g")`, which is present in `"hug"`, `"pug"`, and `"hugs"`, for a grand total of 20 times in the vocabulary.

Thus, the first merge rule learned by the tokenizer is `("u", "g") -> "ug"`, which means that `"ug"` will be added to the vocabulary, and the pair should be merged in all the words of the corpus. At the end of this stage, the vocabulary and corpus look like this:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

Now we have some pairs that result in a token longer than two characters: the pair `("h", "ug")`, for instance (present 15 times in the corpus). The most frequent pair at this stage is `("u", "n")`, however, present 16 times in the corpus, so the second merge rule learned is `("u", "n") -> "un"`. Adding that to the vocabulary and merging all existing occurrences leads us to:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

Now the most frequent pair is `("h", "ug")`, so we learn the merge rule `("h", "ug") -> "hug"`, which gives us our first three-letter token. After the merge, the corpus looks like this:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]
Corpus: ("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```

And we continue like this until we reach the desired vocabulary size.

> [!TIP]
> ✏️ **Now your turn!** What do you think the next merge rule will be?

## Tokenization algorithm[[tokenization-algorithm]]

Tokenization follows the training process closely, in the sense that new inputs are tokenized by applying the following steps:

1. Normalization
2. Pre-tokenization
3. Splitting the words into individual characters
4. Applying the merge rules learned in order on those splits

Let's take the example we used during training, with the three merge rules learned:

```
("u", "g") -> "ug"
("u", "n") -> "un"
("h", "ug") -> "hug"
```

The word `"bug"` will be tokenized as `["b", "ug"]`. `"mug"`, however, will be tokenized as `["[UNK]", "ug"]` since the letter `"m"` was not in the base vocabulary. Likewise, the word `"thug"` will be tokenized as `["[UNK]", "hug"]`: the letter `"t"` is not in the base vocabulary, and applying the merge rules results first in `"u"` and `"g"` being merged and then `"h"` and `"ug"` being merged.

> [!TIP]
> ✏️ **Now your turn!** How do you think  the word `"unhug"` will be tokenized?

## Implementing BPE[[implementing-bpe]]

Now let's take a look at an implementation of the BPE algorithm. This won't be an optimized version you can actually use on a big corpus; we just want to show you the code so you can understand the algorithm a little bit better.

First we need a corpus, so let's create a simple one with a few sentences:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

Next, we need to pre-tokenize that corpus into words. Since we are replicating a BPE tokenizer (like GPT-2), we will use the `gpt2` tokenizer for the pre-tokenization:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Then we compute the frequencies of each word in the corpus as we do the pre-tokenization:

```python
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
```

```python out
defaultdict(
    int,
    {
        "This": 3,
        "Ġis": 2,
        "Ġthe": 1,
        "ĠHugging": 1,
        "ĠFace": 1,
        "ĠCourse": 1,
        ".": 4,
        "Ġchapter": 1,
        "Ġabout": 1,
        "Ġtokenization": 1,
        "Ġsection": 1,
        "Ġshows": 1,
        "Ġseveral": 1,
        "Ġtokenizer": 1,
        "Ġalgorithms": 1,
        "Hopefully": 1,
        ",": 1,
        "Ġyou": 1,
        "Ġwill": 1,
        "Ġbe": 1,
        "Ġable": 1,
        "Ġto": 1,
        "Ġunderstand": 1,
        "Ġhow": 1,
        "Ġthey": 1,
        "Ġare": 1,
        "Ġtrained": 1,
        "Ġand": 1,
        "Ġgenerate": 1,
        "Ġtokens": 1,
    },
)
```

The next step is to compute the base vocabulary, formed by all the characters used in the corpus:

```python
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)
```

```python out
[
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "Ġ",
]
```

We also add the special tokens used by the model at the beginning of that vocabulary. In the case of GPT-2, the only special token is `"<|endoftext|>"`:

```python
vocab = ["<|endoftext|>"] + alphabet.copy()
```

We now need to split each word into individual characters, to be able to start training:

```python
splits = {word: [c for c in word] for word in word_freqs.keys()}
```

Now that we are ready for training, let's write a function that computes the frequency of each pair. We'll need to use this at each step of the training:

```python
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs
```

Let's have a look at a part of this dictionary after the initial splits:

```python
pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break
```

```python out
('T', 'h'): 3
('h', 'i'): 3
('i', 's'): 5
('Ġ', 'i'): 2
('Ġ', 't'): 7
('t', 'h'): 3
```

Now, finding the most frequent pair only takes a quick loop:

```python
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)
```

```python out
('Ġ', 't') 7
```

So the first merge to learn is `('Ġ', 't') -> 'Ġt'`, and we add `'Ġt'` to the vocabulary:

```python
merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")
```

To continue, we need to apply that merge in our `splits` dictionary. Let's write another function for this:

```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

And we can have a look at the result of the first merge:

```py
splits = merge_pair("Ġ", "t", splits)
print(splits["Ġtrained"])
```

```python out
["Ġt", "r", "a", "i", "n", "e", "d"]
```

Now we have everything we need to loop until we have learned all the merges we want. Let's aim for a vocab size of 50:

```python
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
```

As a result, we've learned 19 merge rules (the initial vocabulary had a size of 31 -- 30 characters in the alphabet, plus the special token):

```py
print(merges)
```

```python out
{
    ("Ġ", "t"): "Ġt",
    ("i", "s"): "is",
    ("e", "r"): "er",
    ("Ġ", "a"): "Ġa",
    ("Ġt", "o"): "Ġto",
    ("e", "n"): "en",
    ("T", "h"): "Th",
    ("Th", "is"): "This",
    ("o", "u"): "ou",
    ("s", "e"): "se",
    ("Ġto", "k"): "Ġtok",
    ("Ġtok", "en"): "Ġtoken",
    ("n", "d"): "nd",
    ("Ġ", "is"): "Ġis",
    ("Ġt", "h"): "Ġth",
    ("Ġth", "e"): "Ġthe",
    ("i", "n"): "in",
    ("Ġa", "b"): "Ġab",
    ("Ġtoken", "i"): "Ġtokeni",
}
```

And the vocabulary is composed of the special token, the initial alphabet, and all the results of the merges:

```py
print(vocab)
```

```python out
[
    "<|endoftext|>",
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "Ġ",
    "Ġt",
    "is",
    "er",
    "Ġa",
    "Ġto",
    "en",
    "Th",
    "This",
    "ou",
    "se",
    "Ġtok",
    "Ġtoken",
    "nd",
    "Ġis",
    "Ġth",
    "Ġthe",
    "in",
    "Ġab",
    "Ġtokeni",
]
```

> [!TIP]
> 💡 Using `train_new_from_iterator()` on the same corpus won't result in the exact same vocabulary. This is because when there is a choice of the most frequent pair, we selected the first one encountered, while the 🤗 Tokenizers library selects the first one based on its inner IDs.

To tokenize a new text, we pre-tokenize it, split it, then apply all the merge rules learned:

```python
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
```

We can try this on any text composed of characters in the alphabet:

```py
tokenize("This is not a token.")
```

```python out
["This", "Ġis", "Ġ", "n", "o", "t", "Ġa", "Ġtoken", "."]
```

> [!WARNING]
> ⚠️ Our implementation will throw an error if there is an unknown character since we didn't do anything to handle them. GPT-2 doesn't actually have an unknown token (it's impossible to get an unknown character when using byte-level BPE), but this could happen here because we did not include all the possible bytes in the initial vocabulary. This aspect of BPE is beyond the scope of this section, so we've left the details out.

That's it for the BPE algorithm! Next, we'll have a look at WordPiece.

---

<!-- Section 6.6 -->

# WordPiece tokenization[[wordpiece-tokenization]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section6.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section6.ipynb"},
]} />

WordPiece is the tokenization algorithm Google developed to pretrain BERT. It has since been reused in quite a few Transformer models based on BERT, such as DistilBERT, MobileBERT, Funnel Transformers, and MPNET. It's very similar to BPE in terms of the training, but the actual tokenization is done differently.

<Youtube id="qpv6ms_t_1A"/>

> [!TIP]
> 💡 This section covers WordPiece in depth, going as far as showing a full implementation. You can skip to the end if you just want a general overview of the tokenization algorithm.

## Training algorithm[[training-algorithm]]

> [!WARNING]
> ⚠️ Google never open-sourced its implementation of the training algorithm of WordPiece, so what follows is our best guess based on the published literature. It may not be 100% accurate.

Like BPE, WordPiece starts from a small vocabulary including the special tokens used by the model and the initial alphabet. Since it identifies subwords by adding a prefix (like `##` for BERT), each word is initially split by adding that prefix to all the characters inside the word. So, for instance, `"word"` gets split like this:

```
w ##o ##r ##d
```

Thus, the initial alphabet contains all the characters present at the beginning of a word and the characters present inside a word preceded by the WordPiece prefix.

Then, again like BPE, WordPiece learns merge rules. The main difference is the way the pair to be merged is selected. Instead of selecting the most frequent pair, WordPiece computes a score for each pair, using the following formula:

$$\mathrm{score} = (\mathrm{freq\_of\_pair}) / (\mathrm{freq\_of\_first\_element} \times \mathrm{freq\_of\_second\_element})$$

By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary. For instance, it won't necessarily merge `("un", "##able")` even if that pair occurs very frequently in the vocabulary, because the two pairs `"un"` and `"##able"` will likely each appear in a lot of other words and have a high frequency. In contrast, a pair like `("hu", "##gging")` will probably be merged faster (assuming the word "hugging" appears often in the vocabulary) since `"hu"` and `"##gging"` are likely to be less frequent individually.

Let's look at the same vocabulary we used in the BPE training example:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

The splits here will be:

```
("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##g" "##s", 5)
```

so the initial vocabulary will be `["b", "h", "p", "##g", "##n", "##s", "##u"]` (if we forget about special tokens for now). The most frequent pair is `("##u", "##g")` (present 20 times), but the individual frequency of `"##u"` is very high, so its score is not the highest (it's 1 / 36). All pairs with a `"##u"` actually have that same score (1 / 36), so the best score goes to the pair `("##g", "##s")` -- the only one without a `"##u"` -- at 1 / 20, and the first merge learned is `("##g", "##s") -> ("##gs")`.

Note that when we merge, we remove the `##` between the two tokens, so we add `"##gs"` to the vocabulary and apply the merge in the words of the corpus:

```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
Corpus: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)
```

At this point, `"##u"` is in all the possible pairs, so they all end up with the same score. Let's say that in this case, the first pair is merged, so `("h", "##u") -> "hu"`. This takes us to:

```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu"]
Corpus: ("hu" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```

Then the next best score is shared by `("hu", "##g")` and `("hu", "##gs")` (with 1/15, compared to 1/21 for all the other pairs), so the first pair with the biggest score is merged:

```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu", "hug"]
Corpus: ("hug", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```

and we continue like this until we reach the desired vocabulary size.

> [!TIP]
> ✏️ **Now your turn!** What will the next merge rule be?

## Tokenization algorithm[[tokenization-algorithm]]

Tokenization differs in WordPiece and BPE in that WordPiece only saves the final vocabulary, not the merge rules learned. Starting from the word to tokenize, WordPiece finds the longest subword that is in the vocabulary, then splits on it. For instance, if we use the vocabulary learned in the example above, for the word `"hugs"` the longest subword starting from the beginning that is inside the vocabulary is `"hug"`, so we split there and get `["hug", "##s"]`. We then continue with `"##s"`, which is in the vocabulary, so the tokenization of `"hugs"` is `["hug", "##s"]`.

With BPE, we would have applied the merges learned in order and tokenized this as `["hu", "##gs"]`, so the encoding is different.

As another example, let's see how the word `"bugs"` would be tokenized. `"b"` is the longest subword starting at the beginning of the word that is in the vocabulary, so we split there and get `["b", "##ugs"]`. Then `"##u"` is the longest subword starting at the beginning of `"##ugs"` that is in the vocabulary, so we split there and get `["b", "##u, "##gs"]`. Finally, `"##gs"` is in the vocabulary, so this last list is the tokenization of `"bugs"`.

When the tokenization gets to a stage where it's not possible to find a subword in the vocabulary, the whole word is tokenized as unknown -- so, for instance, `"mug"` would be tokenized as `["[UNK]"]`, as would `"bum"` (even if we can begin with `"b"` and `"##u"`, `"##m"` is not the vocabulary, and the resulting tokenization will just be `["[UNK]"]`, not `["b", "##u", "[UNK]"]`). This is another difference from BPE, which would only classify the individual characters not in the vocabulary as unknown.

> [!TIP]
> ✏️ **Now your turn!** How will the word `"pugs"` be tokenized?

## Implementing WordPiece[[implementing-wordpiece]]

Now let's take a look at an implementation of the WordPiece algorithm. Like with BPE, this is just pedagogical, and you won't able to use this on a big corpus.

We will use the same corpus as in the BPE example:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

First, we need to pre-tokenize the corpus into words. Since we are replicating a WordPiece tokenizer (like BERT), we will use the `bert-base-cased` tokenizer for the pre-tokenization:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Then we compute the frequencies of each word in the corpus as we do the pre-tokenization:

```python
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs
```

```python out
defaultdict(
    int,
    {
        "This": 3,
        "is": 2,
        "the": 1,
        "Hugging": 1,
        "Face": 1,
        "Course": 1,
        ".": 4,
        "chapter": 1,
        "about": 1,
        "tokenization": 1,
        "section": 1,
        "shows": 1,
        "several": 1,
        "tokenizer": 1,
        "algorithms": 1,
        "Hopefully": 1,
        ",": 1,
        "you": 1,
        "will": 1,
        "be": 1,
        "able": 1,
        "to": 1,
        "understand": 1,
        "how": 1,
        "they": 1,
        "are": 1,
        "trained": 1,
        "and": 1,
        "generate": 1,
        "tokens": 1,
    },
)
```

As we saw before, the alphabet is the unique set composed of all the first letters of words, and all the other letters that appear in words prefixed by `##`:

```python
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
alphabet

print(alphabet)
```

```python out
[
    "##a",
    "##b",
    "##c",
    "##d",
    "##e",
    "##f",
    "##g",
    "##h",
    "##i",
    "##k",
    "##l",
    "##m",
    "##n",
    "##o",
    "##p",
    "##r",
    "##s",
    "##t",
    "##u",
    "##v",
    "##w",
    "##y",
    "##z",
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "g",
    "h",
    "i",
    "s",
    "t",
    "u",
    "w",
    "y",
]
```

We also add the special tokens used by the model at the beginning of that vocabulary. In the case of BERT, it's the list `["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]`:

```python
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
```

Next we need to split each word, with all the letters that are not the first prefixed by `##`:

```python
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)] for word in word_freqs.keys()
}
```

Now that we are ready for training, let's write a function that computes the score of each pair. We'll need to use this at each step of the training:

```python
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores
```

Let's have a look at a part of this dictionary after the initial splits:

```python
pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break
```

```python out
('T', '##h'): 0.125
('##h', '##i'): 0.03409090909090909
('##i', '##s'): 0.02727272727272727
('i', '##s'): 0.1
('t', '##h'): 0.03571428571428571
('##h', '##e'): 0.011904761904761904
```

Now, finding the pair with the best score only takes a quick loop:

```python
best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score

print(best_pair, max_score)
```

```python out
('a', '##b') 0.2
```

So the first merge to learn is `('a', '##b') -> 'ab'`, and we add `'ab'` to the vocabulary:

```python
vocab.append("ab")
```

To continue, we need to apply that merge in our `splits` dictionary. Let's write another function for this:

```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

And we can have a look at the result of the first merge:

```py
splits = merge_pair("a", "##b", splits)
splits["about"]
```

```python out
["ab", "##o", "##u", "##t"]
```

Now we have everything we need to loop until we have learned all the merges we want. Let's aim for a vocab size of 70:

```python
vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)
```

We can then look at the generated vocabulary:

```py
print(vocab)
```

```python out
[
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "##a",
    "##b",
    "##c",
    "##d",
    "##e",
    "##f",
    "##g",
    "##h",
    "##i",
    "##k",
    "##l",
    "##m",
    "##n",
    "##o",
    "##p",
    "##r",
    "##s",
    "##t",
    "##u",
    "##v",
    "##w",
    "##y",
    "##z",
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "g",
    "h",
    "i",
    "s",
    "t",
    "u",
    "w",
    "y",
    "ab",
    "##fu",
    "Fa",
    "Fac",
    "##ct",
    "##ful",
    "##full",
    "##fully",
    "Th",
    "ch",
    "##hm",
    "cha",
    "chap",
    "chapt",
    "##thm",
    "Hu",
    "Hug",
    "Hugg",
    "sh",
    "th",
    "is",
    "##thms",
    "##za",
    "##zat",
    "##ut",
]
```

As we can see, compared to BPE, this tokenizer learns parts of words as tokens a bit faster.

> [!TIP]
> 💡 Using `train_new_from_iterator()` on the same corpus won't result in the exact same vocabulary. This is because the 🤗 Tokenizers library does not implement WordPiece for the training (since we are not completely sure of its internals), but uses BPE instead.

To tokenize a new text, we pre-tokenize it, split it, then apply the tokenization algorithm on each word. That is, we look for the biggest subword starting at the beginning of the first word and split it, then we repeat the process on the second part, and so on for the rest of that word and the following words in the text:

```python
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
```

Let's test it on one word that's in the vocabulary, and another that isn't:

```python
print(encode_word("Hugging"))
print(encode_word("HOgging"))
```

```python out
["Hugg", "##i", "##n", "##g"]
["[UNK]"]
```

Now, let's write a function that tokenizes a text:

```python
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])
```

We can try it on any text:

```python
tokenize("This is the Hugging Face course!")
```

```python out
[
    "Th",
    "##i",
    "##s",
    "is",
    "th",
    "##e",
    "Hugg",
    "##i",
    "##n",
    "##g",
    "Fac",
    "##e",
    "c",
    "##o",
    "##u",
    "##r",
    "##s",
    "##e",
    "[UNK]",
]
```

That's it for the WordPiece algorithm! Now let's take a look at Unigram.


---

<!-- Section 6.7 -->

# Unigram tokenization[[unigram-tokenization]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section7.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section7.ipynb"},
]} />

The Unigram algorithm is used in combination with [SentencePiece](https://huggingface.co/papers/1808.06226), which is the tokenization algorithm used by models like AlBERT, T5, mBART, Big Bird, and XLNet.

SentencePiece addresses the fact that not all languages use spaces to separate words. Instead, SentencePiece treats the input as a raw input stream which includes the space in the set of characters to use. Then it can use the Unigram algorithm to construct the appropriate vocabulary.

<Youtube id="TGZfZVuF9Yc"/>

> [!TIP]
> 💡 This section covers Unigram in depth, going as far as showing a full implementation. You can skip to the end if you just want a general overview of the tokenization algorithm.

## Training algorithm[[training-algorithm]]

Compared to BPE and WordPiece, Unigram works in the other direction: it starts from a big vocabulary and removes tokens from it until it reaches the desired vocabulary size. There are several options to use to build that base vocabulary: we can take the most common substrings in pre-tokenized words, for instance, or apply BPE on the initial corpus with a large vocabulary size.

At each step of the training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was removed, and looks for the symbols that would increase it the least. Those symbols have a lower effect on the overall loss over the corpus, so in a sense they are "less needed" and are the best candidates for removal.

This is all a very costly operation, so we don't just remove the single symbol associated with the lowest loss increase, but the \\(p\\) (\\(p\\) being a hyperparameter you can control, usually 10 or 20) percent of the symbols associated with the lowest loss increase. This process is then repeated until the vocabulary has reached the desired size.

Note that we never remove the base characters, to make sure any word can be tokenized.

Now, this is still a bit vague: the main part of the algorithm is to compute a loss over the corpus and see how it changes when we remove some tokens from the vocabulary, but we haven't explained how to do this yet. This step relies on the tokenization algorithm of a Unigram model, so we'll dive into this next.

We'll reuse the corpus from the previous examples:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

and for this example, we will take all strict substrings for the initial vocabulary :

```
["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]
```

## Tokenization algorithm[[tokenization-algorithm]]

A Unigram model is a type of language model that considers each token to be independent of the tokens before it. It's the simplest language model, in the sense that the probability of token X given the previous context is just the probability of token X. So, if we used a Unigram language model to generate text, we would always predict the most common token.

The probability of a given token is its frequency (the number of times we find it) in the original corpus, divided by the sum of all frequencies of all tokens in the vocabulary (to make sure the probabilities sum up to 1). For instance, `"ug"` is present in `"hug"`, `"pug"`, and `"hugs"`, so it has a frequency of 20 in our corpus.

Here are the frequencies of all the possible subwords in the vocabulary:

```
("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16)
("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)
```

So, the sum of all frequencies is 210, and the probability of the subword `"ug"` is thus 20/210.

> [!TIP]
> ✏️ **Now your turn!** Write the code to compute the frequencies above and double-check that the results shown are correct, as well as the total sum.

Now, to tokenize a given word, we look at all the possible segmentations into tokens and compute the probability of each according to the Unigram model. Since all tokens are considered independent, this probability is just the product of the probability of each token. For instance, the tokenization `["p", "u", "g"]` of `"pug"` has the probability:

$$P([``p", ``u", ``g"]) = P(``p") \times P(``u") \times P(``g") = \frac{5}{210} \times \frac{36}{210} \times \frac{20}{210} = 0.000389$$

Comparatively, the tokenization `["pu", "g"]` has the probability:

$$P([``pu", ``g"]) = P(``pu") \times P(``g") = \frac{5}{210} \times \frac{20}{210} = 0.0022676$$

so that one is way more likely. In general, tokenizations with the least tokens possible will have the highest probability (because of that division by 210 repeated for each token), which corresponds to what we want intuitively: to split a word into the least number of tokens possible.

The tokenization of a word with the Unigram model is then the tokenization with the highest probability. In the example of `"pug"`, here are the probabilities we would get for each possible segmentation:

```
["p", "u", "g"] : 0.000389
["p", "ug"] : 0.0022676
["pu", "g"] : 0.0022676
```

So, `"pug"` would be tokenized as `["p", "ug"]` or `["pu", "g"]`, depending on which of those segmentations is encountered first (note that in a larger corpus, equality cases like this will be rare).

In this case, it was easy to find all the possible segmentations and compute their probabilities, but in general it's going to be a bit harder. There is a classic algorithm used for this, called the *Viterbi algorithm*. Essentially, we can build a graph to detect the possible segmentations of a given word by saying there is a branch from character _a_ to character _b_ if the subword from _a_ to _b_ is in the vocabulary, and attribute to that branch the probability of the subword.

To find the path in that graph that is going to have the best score the Viterbi algorithm determines, for each position in the word, the segmentation with the best score that ends at that position. Since we go from the beginning to the end, that best score can be found by looping through all subwords ending at the current position and then using the best tokenization score from the position this subword begins at. Then, we just have to unroll the path taken to arrive at the end.

Let's take a look at an example using our vocabulary and the word `"unhug"`. For each position, the subwords with the best scores ending there are the following:

```
Character 0 (u): "u" (score 0.171429)
Character 1 (n): "un" (score 0.076191)
Character 2 (h): "un" "h" (score 0.005442)
Character 3 (u): "un" "hu" (score 0.005442)
Character 4 (g): "un" "hug" (score 0.005442)
```

Thus `"unhug"` would be tokenized as `["un", "hug"]`.

> [!TIP]
> ✏️ **Now your turn!** Determine the tokenization of the word `"huggun"`, and its score.

## Back to training[[back-to-training]]

Now that we have seen how the tokenization works, we can dive a little more deeply into the loss used during training. At any given stage, this loss is computed by tokenizing every word in the corpus, using the current vocabulary and the Unigram model determined by the frequencies of each token in the corpus (as seen before).

Each word in the corpus has a score, and the loss is the negative log likelihood of those scores -- that is, the sum for all the words in the corpus of all the `-log(P(word))`.

Let's go back to our example with the following corpus:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

The tokenization of each word with their respective scores is:

```
"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)
```

So the loss is:

```
10 * (-log(0.071428)) + 5 * (-log(0.007710)) + 12 * (-log(0.006168)) + 4 * (-log(0.001451)) + 5 * (-log(0.001701)) = 169.8
```

Now we need to compute how removing each token affects the loss. This is rather tedious, so we'll just do it for two tokens here and save the whole process for when we have code to help us. In this (very) particular case, we had two equivalent tokenizations of all the words: as we saw earlier, for example, `"pug"` could be tokenized `["p", "ug"]` with the same score. Thus, removing the `"pu"` token from the vocabulary will give the exact same loss.

On the other hand, removing `"hug"` will make the loss worse, because the tokenization of `"hug"` and `"hugs"` will become:

```
"hug": ["hu", "g"] (score 0.006802)
"hugs": ["hu", "gs"] (score 0.001701)
```

These changes will cause the loss to rise by:

```
- 10 * (-log(0.071428)) + 10 * (-log(0.006802)) = 23.5
```

Therefore, the token `"pu"` will probably be removed from the vocabulary, but not `"hug"`.

## Implementing Unigram[[implementing-unigram]]

Now let's implement everything we've seen so far in code. Like with BPE and WordPiece, this is not an efficient implementation of the Unigram algorithm (quite the opposite), but it should help you understand it a bit better.

We will use the same corpus as before as an example:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

This time, we will use `xlnet-base-cased` as our model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
```

Like for BPE and WordPiece, we begin by counting the number of occurrences of each word in the corpus:

```python
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs
```

Then, we need to initialize our vocabulary to something larger than the vocab size we will want at the end. We have to include all the basic characters (otherwise we won't be able to tokenize every word), but for the bigger substrings we'll only keep the most common ones, so we sort them by frequency:

```python
char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # Loop through the subwords of length at least 2
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# Sort subwords by frequency
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
sorted_subwords[:10]
```

```python out
[
    ("▁t", 7),
    ("is", 5),
    ("er", 5),
    ("▁a", 5),
    ("▁to", 4),
    ("to", 4),
    ("en", 4),
    ("▁T", 3),
    ("▁Th", 3),
    ("▁Thi", 3),
]
```

We group the characters with the best subwords to arrive at an initial vocabulary of size 300:

```python
token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}
```

> [!TIP]
> 💡 SentencePiece uses a more efficient algorithm called Enhanced Suffix Array (ESA) to create the initial vocabulary.

Next, we compute the sum of all frequencies, to convert the frequencies into probabilities. For our model we will store the logarithms of the probabilities, because it's more numerically stable to add logarithms than to multiply small numbers, and this will simplify the computation of the loss of the model:

```python
from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```

Now the main function is the one that tokenizes words using the Viterbi algorithm. As we saw before, that algorithm computes the best segmentation of each substring of the word, which we will store in a variable named `best_segmentations`. We will store one dictionary per position in the word (from 0 to its total length), with two keys: the index of the start of the last token in the best segmentation, and the score of the best segmentation. With the index of the start of the last token, we will be able to retrieve the full segmentation once the list is completely populated.

Populating the list is done with just two loops: the main loop goes over each start position, and the second loop tries all substrings beginning at that start position. If the substring is in the vocabulary, we have a new segmentation of the word up until that end position, which we compare to what is in `best_segmentations`.

Once the main loop is finished, we just start from the end and hop from one start position to the next, recording the tokens as we go, until we reach the start of the word:

```python
def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score
```

We can already try our initial model on some words:

```python
print(encode_word("Hopefully", model))
print(encode_word("This", model))
```

```python out
(["H", "o", "p", "e", "f", "u", "ll", "y"], 41.5157494601402)
(["This"], 6.288267030694535)
```

Now it's easy to compute the loss of the model on the corpus!

```python
def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss
```

We can check it works on the model we have:

```python
compute_loss(model)
```

```python out
413.10377642940875
```

Computing the scores for each token is not very hard either; we just have to compute the loss for the models obtained by deleting each token:

```python
import copy


def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores
```

We can try it on a given token:

```python
scores = compute_scores(model)
print(scores["ll"])
print(scores["his"])
```

Since `"ll"` is used in the tokenization of `"Hopefully"`, and removing it will probably make us use the token `"l"` twice instead, we expect it will have a positive loss. `"his"` is only used inside the word `"This"`, which is tokenized as itself, so we expect it to have a zero loss. Here are the results:

```python out
6.376412403623874
0.0
```

> [!TIP]
> 💡 This approach is very inefficient, so SentencePiece uses an approximation of the loss of the model without token X: instead of starting from scratch, it just replaces token X by its segmentation in the vocabulary that is left. This way, all the scores can be computed at once at the same time as the model loss.

With all of this in place, the last thing we need to do is add the special tokens used by the model to the vocabulary, then loop until we have pruned enough tokens from the vocabulary to reach our desired size:

```python
percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # Remove percent_to_remove tokens with the lowest scores.
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```

Then, to tokenize some text, we just need to apply the pre-tokenization and then use our `encode_word()` function:

```python
def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


tokenize("This is the Hugging Face course.", model)
```

```python out
["▁This", "▁is", "▁the", "▁Hugging", "▁Face", "▁", "c", "ou", "r", "s", "e", "."]
```

> [!TIP]
> The XLNetTokenizer uses SentencePiece which is why the `"_"` character is included. To decode with SentencePiece, concatenate all the tokens and replace `"_"` with a space.

That's it for Unigram! Hopefully by now you're feeling like an expert in all things tokenizer. In the next section, we will delve into the building blocks of the 🤗 Tokenizers library, and show you how you can use them to build your own tokenizer.


---

<!-- Section 6.8 -->

# Building a tokenizer, block by block[[building-a-tokenizer-block-by-block]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section8.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section8.ipynb"},
]} />

As we've seen in the previous sections, tokenization comprises several steps:

- Normalization (any cleanup of the text that is deemed necessary, such as removing spaces or accents, Unicode normalization, etc.)
- Pre-tokenization (splitting the input into words)
- Running the input through the model (using the pre-tokenized words to produce a sequence of tokens)
- Post-processing (adding the special tokens of the tokenizer, generating the attention mask and token type IDs)

As a reminder, here's another look at the overall process:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg" alt="The tokenization pipeline.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline-dark.svg" alt="The tokenization pipeline.">
</div>

The 🤗 Tokenizers library has been built to provide several options for each of those steps, which you can mix and match together. In this section we'll see how we can build a tokenizer from scratch, as opposed to training a new tokenizer from an old one as we did in [section 2](/course/chapter6/2). You'll then be able to build any kind of tokenizer you can think of!

<Youtube id="MR8tZm5ViWU"/>

More precisely, the library is built around a central `Tokenizer` class with the building blocks regrouped in submodules:

- `normalizers` contains all the possible types of `Normalizer` you can use (complete list [here](https://huggingface.co/docs/tokenizers/api/normalizers)).
- `pre_tokenizers` contains all the possible types of `PreTokenizer` you can use (complete list [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)).
- `models` contains the various types of `Model` you can use, like `BPE`, `WordPiece`, and `Unigram` (complete list [here](https://huggingface.co/docs/tokenizers/api/models)).
- `trainers` contains all the different types of `Trainer` you can use to train your model on a corpus (one per type of model; complete list [here](https://huggingface.co/docs/tokenizers/api/trainers)).
- `post_processors` contains the various types of `PostProcessor` you can use (complete list [here](https://huggingface.co/docs/tokenizers/api/post-processors)).
- `decoders` contains the various types of `Decoder` you can use to decode the outputs of tokenization (complete list [here](https://huggingface.co/docs/tokenizers/components#decoders)).

You can find the whole list of building blocks [here](https://huggingface.co/docs/tokenizers/components).

## Acquiring a corpus[[acquiring-a-corpus]]

To train our new tokenizer, we will use a small corpus of text (so the examples run fast). The steps for acquiring the corpus are similar to the ones we took at the [beginning of this chapter](/course/chapter6/2), but this time we'll use the [WikiText-2](https://huggingface.co/datasets/wikitext) dataset:

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```

The function `get_training_corpus()` is a generator that will yield batches of 1,000 texts, which we will use to train the tokenizer. 

🤗 Tokenizers can also be trained on text files directly. Here's how we can generate a text file containing all the texts/inputs from WikiText-2 that we can use locally:

```python
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")
```

Next we'll show you how to build your own BERT, GPT-2, and XLNet tokenizers, block by block. That will give us an example of each of the three main tokenization algorithms: WordPiece, BPE, and Unigram. Let's start with BERT!

## Building a WordPiece tokenizer from scratch[[building-a-wordpiece-tokenizer-from-scratch]]

To build a tokenizer with the 🤗 Tokenizers library, we start by instantiating a `Tokenizer` object with a `model`, then set its `normalizer`, `pre_tokenizer`, `post_processor`, and `decoder` attributes to the values we want.

For this example, we'll create a `Tokenizer` with a WordPiece model:

```python
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```

We have to specify the `unk_token` so the model knows what to return when it encounters characters it hasn't seen before. Other arguments we can set here include the `vocab` of our model (we're going to train the model, so we don't need to set this) and `max_input_chars_per_word`, which specifies a maximum length for each word (words longer than the value passed will be split).

The first step of tokenization is normalization, so let's begin with that. Since BERT is widely used, there is a `BertNormalizer` with the classic options we can set for BERT: `lowercase` and `strip_accents`, which are self-explanatory; `clean_text` to remove all control characters and replace repeating spaces with a single one; and `handle_chinese_chars`, which places spaces around Chinese characters. To replicate the `bert-base-uncased` tokenizer, we can just set this normalizer:

```python
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
```

Generally speaking, however, when building a new tokenizer you won't have access to such a handy normalizer already implemented in the 🤗 Tokenizers library -- so let's see how to create the BERT normalizer by hand. The library provides a `Lowercase` normalizer and a `StripAccents` normalizer, and you can compose several normalizers using a `Sequence`:

```python
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```

We're also using an `NFD` Unicode normalizer, as otherwise the `StripAccents` normalizer won't properly recognize the accented characters and thus won't strip them out.

As we've seen before, we can use the `normalize_str()` method of the `normalizer` to check out the effects it has on a given text:

```python
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
```

```python out
hello how are u?
```

> [!TIP]
> **To go further** If you test the two versions of the previous normalizers on a string containing the unicode character `u"\u0085"` you will surely notice that these two normalizers are not exactly equivalent. 
> To not over-complicate the version with `normalizers.Sequence` too much , we haven't included the Regex replacements that the `BertNormalizer` requires when the `clean_text` argument is set to `True` - which is the default behavior. But don't worry: it is possible to get exactly the same normalization without using the handy `BertNormalizer` by adding two `normalizers.Replace`'s to the normalizers sequence.

Next is the pre-tokenization step. Again, there is a prebuilt `BertPreTokenizer` that we can use:

```python
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
```

Or we can build it from scratch:

```python
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
```

Note that the `Whitespace` pre-tokenizer splits on whitespace and all characters that are not letters, digits, or the underscore character, so it technically splits on whitespace and punctuation:

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
```

```python out
[
    ("Let", (0, 3)),
    ("'", (3, 4)),
    ("s", (4, 5)),
    ("test", (6, 10)),
    ("my", (11, 13)),
    ("pre", (14, 17)),
    ("-", (17, 18)),
    ("tokenizer", (18, 27)),
    (".", (27, 28)),
]
```

If you only want to split on whitespace, you should use the `WhitespaceSplit` pre-tokenizer instead:

```python
pre_tokenizer = pre_tokenizers.WhitespaceSplit()
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
```

```python out
[("Let's", (0, 5)), ("test", (6, 10)), ("my", (11, 13)), ("pre-tokenizer.", (14, 28))]
```

Like with normalizers, you can use a `Sequence` to compose several pre-tokenizers:

```python
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
```

```python out
[
    ("Let", (0, 3)),
    ("'", (3, 4)),
    ("s", (4, 5)),
    ("test", (6, 10)),
    ("my", (11, 13)),
    ("pre", (14, 17)),
    ("-", (17, 18)),
    ("tokenizer", (18, 27)),
    (".", (27, 28)),
]
```

The next step in the tokenization pipeline is running the inputs through the model. We already specified our model in the initialization, but we still need to train it, which will require a `WordPieceTrainer`. The main thing to remember when instantiating a trainer in 🤗 Tokenizers is that you need to pass it all the special tokens you intend to use -- otherwise it won't add them to the vocabulary, since they are not in the training corpus:

```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
```

As well as specifying the `vocab_size` and `special_tokens`, we can set the `min_frequency` (the number of times a token must appear to be included in the vocabulary) or change the `continuing_subword_prefix` (if we want to use something different from `##`).

To train our model using the iterator we defined earlier, we just have to execute this command:

```python
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

We can also use text files to train our tokenizer, which would look like this (we reinitialize the model with an empty `WordPiece` beforehand):

```python
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

In both cases, we can then test the tokenizer on a text by calling the `encode()` method:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```

```python out
["let", "'", "s", "test", "this", "tok", "##eni", "##zer", "."]
```

The `encoding` obtained is an `Encoding`, which contains all the necessary outputs of the tokenizer in its various attributes: `ids`, `type_ids`, `tokens`, `offsets`, `attention_mask`, `special_tokens_mask`, and `overflowing`.

The last step in the tokenization pipeline is post-processing. We need to add the `[CLS]` token at the beginning and the `[SEP]` token at the end (or after each sentence, if we have a pair of sentences). We will use a `TemplateProcessor` for this, but first we need to know the IDs of the `[CLS]` and `[SEP]` tokens in the vocabulary:

```python
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)
```

```python out
(2, 3)
```

To write the template for the `TemplateProcessor`, we have to specify how to treat a single sentence and a pair of sentences. For both, we write the special tokens we want to use; the first (or single) sentence is represented by `$A`, while the second sentence (if encoding a pair) is represented by `$B`. For each of these (special tokens and sentences), we also specify the corresponding token type ID after a colon. 

The classic BERT template is thus defined as follows:

```python
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
```

Note that we need to pass along the IDs of the special tokens, so the tokenizer can properly convert them to their IDs.

Once this is added, going back to our previous example will give:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```

```python out
["[CLS]", "let", "'", "s", "test", "this", "tok", "##eni", "##zer", ".", "[SEP]"]
```

And on a pair of sentences, we get the proper result:

```python
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)
```

```python out
[
    "[CLS]",
    "let",
    "'",
    "s",
    "test",
    "this",
    "tok",
    "##eni",
    "##zer",
    "...",
    "[SEP]",
    "on",
    "a",
    "pair",
    "of",
    "sentences",
    ".",
    "[SEP]",
]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

We've almost finished building this tokenizer from scratch -- the last step is to include a decoder: 

```python
tokenizer.decoder = decoders.WordPiece(prefix="##")
```

Let's test it on our previous `encoding`:

```python
tokenizer.decode(encoding.ids)
```

```python out
"let's test this tokenizer... on a pair of sentences."
```

Great! We can save our tokenizer in a single JSON file like this:

```python
tokenizer.save("tokenizer.json")
```

We can then reload that file in a `Tokenizer` object with the `from_file()` method:

```python
new_tokenizer = Tokenizer.from_file("tokenizer.json")
```

To use this tokenizer in 🤗 Transformers, we have to wrap it in a `PreTrainedTokenizerFast`. We can either use the generic class or, if our tokenizer corresponds to an existing model, use that class (here, `BertTokenizerFast`). If you apply this lesson to build a brand new tokenizer, you will have to use the first option.

To wrap the tokenizer in a `PreTrainedTokenizerFast`, we can either pass the tokenizer we built as a `tokenizer_object` or pass the tokenizer file we saved as `tokenizer_file`. The key thing to remember is that we have to manually set all the special tokens, since that class can't infer from the `tokenizer` object which token is the mask token, the `[CLS]` token, etc.:

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```

If you are using a specific tokenizer class (like `BertTokenizerFast`), you will only need to specify the special tokens that are different from the default ones (here, none):

```python
from transformers import BertTokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```

You can then use this tokenizer like any other 🤗 Transformers tokenizer. You can save it with the `save_pretrained()` method, or upload it to the Hub with the `push_to_hub()` method.

Now that we've seen how to build a WordPiece tokenizer, let's do the same for a BPE tokenizer. We'll go a bit faster since you know all the steps, and only highlight the differences.

## Building a BPE tokenizer from scratch[[building-a-bpe-tokenizer-from-scratch]]

Let's now build a GPT-2 tokenizer. Like for the BERT tokenizer, we start by initializing a `Tokenizer` with a BPE model:

```python
tokenizer = Tokenizer(models.BPE())
```

Also like for BERT, we could initialize this model with a vocabulary if we had one (we would need to pass the `vocab` and `merges` in this case), but since we will train from scratch, we don't need to do that. We also don't need to specify an `unk_token` because GPT-2 uses byte-level BPE, which doesn't require it.

GPT-2 does not use a normalizer, so we skip that step and go directly to the pre-tokenization:

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

The option we added to `ByteLevel` here is to not add a space at the beginning of a sentence (which is the default otherwise). We can have a look at the pre-tokenization of an example text like before:

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
```

```python out
[
    ("Let", (0, 3)),
    ("'s", (3, 5)),
    ("Ġtest", (5, 10)),
    ("Ġpre", (10, 14)),
    ("-", (14, 15)),
    ("tokenization", (15, 27)),
    ("!", (27, 28)),
]
```

Next is the model, which needs training. For GPT-2, the only special token is the end-of-text token:

```python
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

Like with the `WordPieceTrainer`, as well as the `vocab_size` and `special_tokens`, we can specify the `min_frequency` if we want to, or if we have an end-of-word suffix (like `</w>`), we can set it with `end_of_word_suffix`. 

This tokenizer can also be trained on text files:

```python
tokenizer.model = models.BPE()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

Let's have a look at the tokenization of a sample text:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```

```python out
["L", "et", "'", "s", "Ġtest", "Ġthis", "Ġto", "ken", "izer", "."]
```

We apply the byte-level post-processing for the GPT-2 tokenizer as follows:

```python
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
```

The `trim_offsets = False` option indicates to the post-processor that we should leave the offsets of tokens that begin with 'Ġ' as they are: this way the start of the offsets will point to the space before the word, not the first character of the word (since the space is technically part of the token). Let's have a look at the result with the text we just encoded, where `'Ġtest'` is the token at index 4:

```python
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
sentence[start:end]
```

```python out
"test"
```

Finally, we add a byte-level decoder:

```python
tokenizer.decoder = decoders.ByteLevel()
```

and we can double-check it works properly:

```python
tokenizer.decode(encoding.ids)
```

```python out
"Let's test this tokenizer."
```

Great! Now that we're done, we can save the tokenizer like before, and wrap it in a `PreTrainedTokenizerFast` or `GPT2TokenizerFast` if we want to use it in 🤗 Transformers:

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
```

or:

```python
from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
```

As the last example, we'll show you how to build a Unigram tokenizer from scratch.

## Building a Unigram tokenizer from scratch[[building-a-unigram-tokenizer-from-scratch]]

Let's now build an XLNet tokenizer. Like for the previous tokenizers, we start by initializing a `Tokenizer` with a Unigram model:

```python
tokenizer = Tokenizer(models.Unigram())
```

Again, we could initialize this model with a vocabulary if we had one.

For the normalization, XLNet uses a few replacements (which come from SentencePiece):

```python
from tokenizers import Regex

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)
```

This replaces <code>``</code> and <code>''</code> with <code>"</code> and any sequence of two or more spaces with a single space, as well as removing the accents in the texts to tokenize.

The pre-tokenizer to use for any SentencePiece tokenizer is `Metaspace`:

```python
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
```

We can have a look at the pre-tokenization of an example text like before:

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")
```

```python out
[("▁Let's", (0, 5)), ("▁test", (5, 10)), ("▁the", (10, 14)), ("▁pre-tokenizer!", (14, 29))]
```

Next is the model, which needs training. XLNet has quite a few special tokens:

```python
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

A very important argument not to forget for the `UnigramTrainer` is the `unk_token`. We can also pass along other arguments specific to the Unigram algorithm, such as the `shrinking_factor` for each step where we remove tokens (defaults to 0.75) or the `max_piece_length` to specify the maximum length of a given token (defaults to 16).

This tokenizer can also be trained on text files:

```python
tokenizer.model = models.Unigram()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

Let's have a look at the tokenization of a sample text:

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```

```python out
["▁Let", "'", "s", "▁test", "▁this", "▁to", "ken", "izer", "."]
```

A peculiarity of XLNet is that it puts the `<cls>` token at the end of the sentence, with a type ID of 2 (to distinguish it from the other tokens). It's padding on the left, as a result. We can deal with all the special tokens and token type IDs with a template, like for BERT, but first we have to get the IDs of the `<cls>` and `<sep>` tokens:

```python
cls_token_id = tokenizer.token_to_id("<cls>")
sep_token_id = tokenizer.token_to_id("<sep>")
print(cls_token_id, sep_token_id)
```

```python out
0 1
```

The template looks like this:

```python
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)
```

And we can test it works by encoding a pair of sentences:

```python
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences!")
print(encoding.tokens)
print(encoding.type_ids)
```

```python out
[
    "▁Let",
    "'",
    "s",
    "▁test",
    "▁this",
    "▁to",
    "ken",
    "izer",
    ".",
    ".",
    ".",
    "<sep>",
    "▁",
    "on",
    "▁",
    "a",
    "▁pair",
    "▁of",
    "▁sentence",
    "s",
    "!",
    "<sep>",
    "<cls>",
]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
```

Finally, we add a `Metaspace` decoder:

```python
tokenizer.decoder = decoders.Metaspace()
```

and we're done with this tokenizer! We can save the tokenizer like before, and wrap it in a `PreTrainedTokenizerFast` or `XLNetTokenizerFast` if we want to use it in 🤗 Transformers. One thing to note when using `PreTrainedTokenizerFast` is that on top of the special tokens, we need to tell the 🤗 Transformers library to pad on the left:

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)
```

Or alternatively:

```python
from transformers import XLNetTokenizerFast

wrapped_tokenizer = XLNetTokenizerFast(tokenizer_object=tokenizer)
```

Now that you have seen how the various building blocks are used to build existing tokenizers, you should be able to write any tokenizer you want with the 🤗 Tokenizers library and be able to use it in 🤗 Transformers.


---

<!-- Section 6.9 -->

# Tokenizers, check![[tokenizers-check]]

<CourseFloatingBanner
    chapter={6}
    classNames="absolute z-10 right-0 top-0"
/>

Great job finishing this chapter!

After this deep dive into tokenizers, you should:

- Be able to train a new tokenizer using an old one as a template
- Understand how to use offsets to map tokens' positions to their original span of text
- Know the differences between BPE, WordPiece, and Unigram
- Be able to mix and match the blocks provided by the 🤗 Tokenizers library to build your own tokenizer
- Be able to use that tokenizer inside the 🤗 Transformers library


---

<!-- Section 6.10 -->

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={6}
    classNames="absolute z-10 right-0 top-0"
/>

Let's test what you learned in this chapter!

### 1. When should you train a new tokenizer?

<Question
	choices={[
		{
			text: "When your dataset is similar to that used by an existing pretrained model, and you want to pretrain a new model",
			explain: "In this case, to save time and compute resources, a better choice would be to use the same tokenizer as the pretrained model and fine-tune that model instead."
		},
		{
			text: "When your dataset is similar to that used by an existing pretrained model, and you want to fine-tune a new model using this pretrained model",
			explain: "To fine-tune a model from a pretrained model, you should always use the same tokenizer."
		},
		{
			text: "When your dataset is different from the one used by an existing pretrained model, and you want to pretrain a new model",
			explain: "Correct! In this case there's no advantage to using the same tokenizer.",
            correct: true
		},
        {
			text: "When your dataset is different from the one used by an existing pretrained model, but you want to fine-tune a new model using this pretrained model",
			explain: "To fine-tune a model from a pretrained model, you should always use the same tokenizer."
		}
	]}
/>

### 2. What is the advantage of using a generator of lists of texts compared to a list of lists of texts when using `train_new_from_iterator()`?

<Question
	choices={[
		{
			text: "That's the only type the method <code>train_new_from_iterator()</code> accepts.",
			explain: "A list of lists of texts is a particular kind of generator of lists of texts, so the method will accept this too. Try again!"
		},
		{
			text: "You will avoid loading the whole dataset into memory at once.",
			explain: "Right! Each batch of texts will be released from memory when you iterate, and the gain will be especially visible if you use 🤗 Datasets to store your texts.",
			correct: true
		},
		{
			text: "This will allow the 🤗 Tokenizers library to use multiprocessing.",
			explain: "No, it will use multiprocessing either way."
		},
        {
			text: "The tokenizer you train will generate better texts.",
			explain: "The tokenizer does not generate text -- are you confusing it with a language model?"
		}
	]}
/>

### 3. What are the advantages of using a "fast" tokenizer?

<Question
	choices={[
		{
			text: "It can process inputs faster than a slow tokenizer when you batch lots of inputs together.",
			explain: "Correct! Thanks to parallelism implemented in Rust, it will be faster on batches of inputs. What other benefit can you think of?",
			correct: true
		},
		{
			text: "Fast tokenizers always tokenize faster than their slow counterparts.",
			explain: "A fast tokenizer can actually be slower when you only give it one or very few texts, since it can't use parallelism."
		},
		{
			text: "It can apply padding and truncation.",
			explain: "True, but slow tokenizers also do that."
		},
        {
			text: "It has some additional features allowing you to map tokens to the span of text that created them.",
			explain: "Indeed -- those are called offset mappings. That's not the only advantage, though.",
			correct: true
		}
	]}
/>

### 4. How does the `token-classification` pipeline handle entities that span over several tokens?

<Question
	choices={[
		{
			text: "The entities with the same label are merged into one entity.",
			explain: "That's oversimplifying things a little. Try again!"
		},
		{
			text: "There is a label for the beginning of an entity and a label for the continuation of an entity.",
			explain: "Correct!",
			correct: true
		},
		{
			text: "In a given word, as long as the first token has the label of the entity, the whole word is considered labeled with that entity.",
			explain: "That's one strategy to handle entities. What other answers here apply?",
			correct: true
		},
        {
			text: "When a token has the label of a given entity, any other following token with the same label is considered part of the same entity, unless it's labeled as the start of a new entity.",
			explain: "That's the most common way to group entities together -- it's not the only right answer, though.",
			correct: true
		}
	]}
/>

### 5. How does the `question-answering` pipeline handle long contexts?

<Question
	choices={[
		{
			text: "It doesn't really, as it truncates the long context at the maximum length accepted by the model.",
			explain: "There is a trick you can use to handle long contexts. Do you remember what it is?"
		},
		{
			text: "It splits the context into several parts and averages the results obtained.",
			explain: "No, it wouldn't make sense to average the results, as some parts of the context won't include the answer."
		},
		{
			text: "It splits the context into several parts (with overlap) and finds the maximum score for an answer in each part.",
			explain: "That's the correct answer!",
			correct: true
		},
        {
			text: "It splits the context into several parts (without overlap, for efficiency) and finds the maximum score for an answer in each part.",
			explain: "No, it includes some overlap between the parts to avoid a situation where the answer would be split across two parts."
		}
	]}
/>

### 6. What is normalization?

<Question
	choices={[
		{
			text: "It's any cleanup the tokenizer performs on the texts in the initial stages.",
			explain: "That's correct -- for instance, it might involve removing accents or whitespace, or lowercasing the inputs.",
			correct: true
		},
		{
			text: "It's a data augmentation technique that involves making the text more normal by removing rare words.",
			explain: "That's incorrect! Try again."
		},
		{
			text: "It's the final post-processing step where the tokenizer adds the special tokens.",
			explain: "That stage is simply called post-processing."
		},
        {
			text: "It's when the embeddings are made with mean 0 and standard deviation 1, by subtracting the mean and dividing by the std.",
			explain: "That process is commonly called normalization when applied to pixel values in computer vision, but it's not what normalization means in NLP."
		}
	]}
/>

### 7. What is pre-tokenization for a subword tokenizer?

<Question
	choices={[
		{
			text: "It's the step before the tokenization, where data augmentation (like random masking) is applied.",
			explain: "No, that step is part of the preprocessing."
		},
		{
			text: "It's the step before the tokenization, where the desired cleanup operations are applied to the text.",
			explain: "No, that's the normalization step."
		},
		{
			text: "It's the step before the tokenizer model is applied, to split the input into words.",
			explain: "That's the correct answer!",
			correct: true
		},
        {
			text: "It's the step before the tokenizer model is applied, to split the input into tokens.",
			explain: "No, splitting into tokens is the job of the tokenizer model."
		}
	]}
/>

### 8. Select the sentences that apply to the BPE model of tokenization.

<Question
	choices={[
		{
			text: "BPE is a subword tokenization algorithm that starts with a small vocabulary and learns merge rules.",
			explain: "That's the case indeed!",
			correct: true
		},
		{
			text: "BPE is a subword tokenization algorithm that starts with a big vocabulary and progressively removes tokens from it.",
			explain: "No, that's the approach taken by a different tokenization algorithm."
		},
		{
			text: "BPE tokenizers learn merge rules by merging the pair of tokens that is the most frequent.",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "A BPE tokenizer learns a merge rule by merging the pair of tokens that maximizes a score that privileges frequent pairs with less frequent individual parts.",
			explain: "No, that's the strategy applied by another tokenization algorithm."
		},
		{
			text: "BPE tokenizes words into subwords by splitting them into characters and then applying the merge rules.",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "BPE tokenizes words into subwords by finding the longest subword starting from the beginning that is in the vocabulary, then repeating the process for the rest of the text.",
			explain: "No, that's another tokenization algorithm's way of doing things."
		},
	]}
/>

### 9. Select the sentences that apply to the WordPiece model of tokenization.

<Question
	choices={[
		{
			text: "WordPiece is a subword tokenization algorithm that starts with a small vocabulary and learns merge rules.",
			explain: "That's the case indeed!",
			correct: true
		},
		{
			text: "WordPiece is a subword tokenization algorithm that starts with a big vocabulary and progressively removes tokens from it.",
			explain: "No, that's the approach taken by a different tokenization algorithm."
		},
		{
			text: "WordPiece tokenizers learn merge rules by merging the pair of tokens that is the most frequent.",
			explain: "No, that's the strategy applied by another tokenization algorithm."
		},
		{
			text: "A WordPiece tokenizer learns a merge rule by merging the pair of tokens that maximizes a score that privileges frequent pairs with less frequent individual parts.",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "WordPiece tokenizes words into subwords by finding the most likely segmentation into tokens, according to the model.",
			explain: "No, that's how another tokenization algorithm works."
		},
		{
			text: "WordPiece tokenizes words into subwords by finding the longest subword starting from the beginning that is in the vocabulary, then repeating the process for the rest of the text.",
			explain: "Yes, this is how WordPiece proceeds for the encoding.",
			correct: true
		},
	]}
/>

### 10. Select the sentences that apply to the Unigram model of tokenization.

<Question
	choices={[
		{
			text: "Unigram is a subword tokenization algorithm that starts with a small vocabulary and learns merge rules.",
			explain: "No, that's the approach taken by a different tokenization algorithm."
		},
		{
			text: "Unigram is a subword tokenization algorithm that starts with a big vocabulary and progressively removes tokens from it.",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "Unigram adapts its vocabulary by minimizing a loss computed over the whole corpus.",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "Unigram adapts its vocabulary by keeping the most frequent subwords.",
			explain: "No, this incorrect."
		},
		{
			text: "Unigram tokenizes words into subwords by finding the most likely segmentation into tokens, according to the model.",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "Unigram tokenizes words into subwords by splitting them into characters, then applying the merge rules.",
			explain: "No, that's how another tokenization algorithm works."
		},
	]}
/>

