# Hugging Face NLP Course — Chapter 3

Source: https://huggingface.co/learn/nlp-course/chapter3


---

<!-- Section 3.1 -->

<FrameworkSwitchCourse {fw} />

# Introduction[[introduction]]

<CourseFloatingBanner
    chapter={3}
    classNames="absolute z-10 right-0 top-0"
/>

In [Chapter 2](/course/chapter2) we explored how to use tokenizers and pretrained models to make predictions. But what if you want to fine-tune a pretrained model to solve a specific task? That's the topic of this chapter! You will learn:

* How to prepare a large dataset from the Hub using the latest 🤗 Datasets features
* How to use the high-level `Trainer` API to fine-tune a model with modern best practices
* How to implement a custom training loop with optimization techniques
* How to leverage the 🤗 Accelerate library to easily run distributed training on any setup
* How to apply current fine-tuning best practices for maximum performance

> [!TIP]
> 📚 **Essential Resources**: Before starting, you might want to review the [🤗 Datasets documentation](https://huggingface.co/docs/datasets/) for data processing.

This chapter will also serve as an introduction to some Hugging Face libraries beyond the 🤗 Transformers library! We'll see how libraries like 🤗 Datasets, 🤗 Tokenizers, 🤗 Accelerate, and 🤗 Evaluate can help you train models more efficiently and effectively.

Each of the main sections in this chapter will teach you something different:
- **Section 2**: Learn modern data preprocessing techniques and efficient dataset handling
- **Section 3**: Master the powerful Trainer API with all its latest features
- **Section 4**: Implement training loops from scratch and understand distributed training with Accelerate

By the end of this chapter, you'll be able to fine-tune models on your own datasets using both high-level APIs and custom training loops, applying the latest best practices in the field.

> [!TIP]
> 🎯 **What You'll Build**: By the end of this chapter, you'll have fine-tuned a BERT model for text classification and understand how to adapt the techniques to your own datasets and tasks.

This chapter focuses exclusively on **PyTorch**, as it has become the standard framework for modern deep learning research and production. We'll use the latest APIs and best practices from the Hugging Face ecosystem.

To upload your trained models to the Hugging Face Hub, you will need a Hugging Face account: [create an account](https://huggingface.co/join)

---

<!-- Section 3.2 -->

# Processing the data[[processing-the-data]]

<CourseFloatingBanner chapter={3}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter3/section2.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter3/section2.ipynb"},
]} />

Continuing with the example from the [previous chapter](/course/chapter2), here is how we would train a sequence classifier on one batch:

```python
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

Of course, just training the model on two sentences is not going to yield very good results. To get better results, you will need to prepare a bigger dataset.

In this section we will use as an example the MRPC (Microsoft Research Paraphrase Corpus) dataset, introduced in a [paper](https://www.aclweb.org/anthology/I05-5002.pdf) by William B. Dolan and Chris Brockett. The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing). We've selected it for this chapter because it's a small dataset, so it's easy to experiment with training on it.

### Loading a dataset from the Hub[[loading-a-dataset-from-the-hub]]

<Youtube id="_BZearw7f0w"/>

The Hub doesn't just contain models; it also has multiple datasets in lots of different languages. You can browse the datasets [here](https://huggingface.co/datasets), and we recommend you try to load and process a new dataset once you have gone through this section (see the general documentation [here](https://huggingface.co/docs/datasets/loading)). But for now, let's focus on the MRPC dataset! This is one of the 10 datasets composing the [GLUE benchmark](https://gluebenchmark.com/), which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.

The 🤗 Datasets library provides a very simple command to download and cache a dataset on the Hub. We can download the MRPC dataset like this:

> [!TIP]
> 💡 **Additional Resources**: For more dataset loading techniques and examples, check out the [🤗 Datasets documentation](https://huggingface.co/docs/datasets/). 

```py
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

As you can see, we get a `DatasetDict` object which contains the training set, the validation set, and the test set. Each of those contains several columns (`sentence1`, `sentence2`, `label`, and `idx`) and a variable number of rows, which are the number of elements in each set (so, there are 3,668 pairs of sentences in the training set, 408 in the validation set, and 1,725 in the test set).

> [!TIP]
> This command downloads and caches the dataset, by default in *~/.cache/huggingface/datasets*. Recall from Chapter 2 that you can customize your cache folder by setting the `HF_HOME` environment variable.

We can access each pair of sentences in our `raw_datasets` object by indexing, like with a dictionary:

```py
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

```python out
{
    "idx": 0,
    "label": 1,
    "sentence1": 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    "sentence2": 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
}
```

We can see the labels are already integers, so we won't have to do any preprocessing there. To know which integer corresponds to which label, we can inspect the `features` of our `raw_train_dataset`. This will tell us the type of each column:

```py
raw_train_dataset.features
```

```python out
{
    "sentence1": Value(dtype="string", id=None),
    "sentence2": Value(dtype="string", id=None),
    "label": ClassLabel(
        num_classes=2, names=["not_equivalent", "equivalent"], names_file=None, id=None
    ),
    "idx": Value(dtype="int32", id=None),
}
```

Behind the scenes, `label` is of type `ClassLabel`, and the mapping of integers to label name is stored in the *names* folder. `0` corresponds to `not_equivalent`, and `1` corresponds to `equivalent`.

> [!TIP]
> ✏️ **Try it out!** Look at element 15 of the training set and element 87 of the validation set. What are their labels?

### Preprocessing a dataset[[preprocessing-a-dataset]]

<Youtube id="0u3ioSwev3s"/>

To preprocess the dataset, we need to convert the text to numbers the model can make sense of. As you saw in the [previous chapter](/course/chapter2), this is done with a tokenizer. We can feed the tokenizer one sentence or a list of sentences, so we can directly tokenize all the first sentences and all the second sentences of each pair like this:

```py
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```

> [!TIP]
> 💡 **Deep Dive**: For more advanced tokenization techniques and understanding how different tokenizers work, explore the [🤗 Tokenizers documentation](https://huggingface.co/docs/transformers/main/en/tokenizer_summary) and the [tokenization guide in the cookbook](https://huggingface.co/learn/cookbook/en/advanced_rag#tokenization-strategies).

However, we can't just pass two sequences to the model and get a prediction of whether the two sentences are paraphrases or not. We need to handle the two sequences as a pair, and apply the appropriate preprocessing. Fortunately, the tokenizer can also take a pair of sequences and prepare it the way our BERT model expects: 

```py
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

```python out
{
    "input_ids": [
        101,
        2023,
        2003,
        1996,
        2034,
        6251,
        1012,
        102,
        2023,
        2003,
        1996,
        2117,
        2028,
        1012,
        102,
    ],
    "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
```

We discussed the `input_ids` and `attention_mask` keys in [Chapter 2](/course/chapter2), but we put off talking about `token_type_ids`. In this example, this is what tells the model which part of the input is the first sentence and which is the second sentence.

> [!TIP]
> ✏️ **Try it out!** Take element 15 of the training set and tokenize the two sentences separately and as a pair. What's the difference between the two results?

If we decode the IDs inside `input_ids` back to words:

```py
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```

we will get:

```python out
[
    "[CLS]",
    "this",
    "is",
    "the",
    "first",
    "sentence",
    ".",
    "[SEP]",
    "this",
    "is",
    "the",
    "second",
    "one",
    ".",
    "[SEP]",
]
```

So we see the model expects the inputs to be of the form `[CLS] sentence1 [SEP] sentence2 [SEP]` when there are two sentences. Aligning this with the `token_type_ids` gives us:

```python out
[
    "[CLS]",
    "this",
    "is",
    "the",
    "first",
    "sentence",
    ".",
    "[SEP]",
    "this",
    "is",
    "the",
    "second",
    "one",
    ".",
    "[SEP]",
]
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

As you can see, the parts of the input corresponding to `[CLS] sentence1 [SEP]` all have a token type ID of `0`, while the other parts, corresponding to `sentence2 [SEP]`, all have a token type ID of `1`.

Note that if you select a different checkpoint, you won't necessarily have the `token_type_ids` in your tokenized inputs (for instance, they're not returned if you use a DistilBERT model). They are only returned when the model will know what to do with them, because it has seen them during its pretraining. 

Here, BERT is pretrained with token type IDs, and on top of the masked language modeling objective we talked about in [Chapter 1](/course/chapter1), it has an additional objective called _next sentence prediction_. The goal with this task is to model the relationship between pairs of sentences.

With next sentence prediction, the model is provided pairs of sentences (with randomly masked tokens) and asked to predict whether the second sentence follows the first. To make the task non-trivial, half of the time the sentences follow each other in the original document they were extracted from, and the other half of the time the two sentences come from two different documents. 

In general, you don't need to worry about whether or not there are `token_type_ids` in your tokenized inputs: as long as you use the same checkpoint for the tokenizer and the model, everything will be fine as the tokenizer knows what to provide to its model.

Now that we have seen how our tokenizer can deal with one pair of sentences, we can use it to tokenize our whole dataset: like in the [previous chapter](/course/chapter2), we can feed the tokenizer a list of pairs of sentences by giving it the list of first sentences, then the list of second sentences. This is also compatible with the padding and truncation options we saw in [Chapter 2](/course/chapter2). So, one way to preprocess the training dataset is:

```py
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

This works well, but it has the disadvantage of returning a dictionary (with our keys, `input_ids`, `attention_mask`, and `token_type_ids`, and values that are lists of lists). It will also only work if you have enough RAM to store your whole dataset during the tokenization (whereas the datasets from the 🤗 Datasets library are [Apache Arrow](https://arrow.apache.org/) files stored on the disk, so you only keep the samples you ask for loaded in memory).

To keep the data as a dataset, we will use the [`Dataset.map()`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map) method. This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. The `map()` method works by applying a function on each element of the dataset, so let's define a function that tokenizes our inputs:

```py
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

This function takes a dictionary (like the items of our dataset) and returns a new dictionary with the keys `input_ids`, `attention_mask`, and `token_type_ids`. Note that it also works if the `example` dictionary contains several samples (each key as a list of sentences) since the `tokenizer` works on lists of pairs of sentences, as seen before. This will allow us to use the option `batched=True` in our call to `map()`, which will greatly speed up the tokenization. The `tokenizer` is backed by a tokenizer written in Rust from the [🤗 Tokenizers](https://github.com/huggingface/tokenizers) library. This tokenizer can be very fast, but only if we give it lots of inputs at once.

Note that we've left the `padding` argument out in our tokenization function for now. This is because padding all the samples to the maximum length is not efficient: it's better to pad the samples when we're building a batch, as then we only need to pad to the maximum length in that batch, and not the maximum length in the entire dataset. This can save a lot of time and processing power when the inputs have very variable lengths!

> [!TIP]
> 📚 **Performance Tips**: Learn more about efficient data processing techniques in the [🤗 Datasets performance guide](https://huggingface.co/docs/datasets/about_arrow).

Here is how we apply the tokenization function on all our datasets at once. We're using `batched=True` in our call to `map` so the function is applied to multiple elements of our dataset at once, and not on each element separately. This allows for faster preprocessing.

```py
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

The way the 🤗 Datasets library applies this processing is by adding new fields to the datasets, one for each key in the dictionary returned by the preprocessing function:

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```

You can even use multiprocessing when applying your preprocessing function with `map()` by passing along a `num_proc` argument. We didn't do this here because the 🤗 Tokenizers library already uses multiple threads to tokenize our samples faster, but if you are not using a fast tokenizer backed by this library, this could speed up your preprocessing.

Our `tokenize_function` returns a dictionary with the keys `input_ids`, `attention_mask`, and `token_type_ids`, so those three fields are added to all splits of our dataset. Note that we could also have changed existing fields if our preprocessing function returned a new value for an existing key in the dataset to which we applied `map()`.

The last thing we will need to do is pad all the examples to the length of the longest element when we batch elements together — a technique we refer to as *dynamic padding*.

##### Dynamic padding[[dynamic-padding]]

<Youtube id="7q5NyFT8REg"/>

The function that is responsible for putting together samples inside a batch is called a *collate function*. It's an argument you can pass when you build a `DataLoader`, the default being a function that will just convert your samples to PyTorch tensors and concatenate them (recursively if your elements are lists, tuples, or dictionaries). This won't be possible in our case since the inputs we have won't all be of the same size. We have deliberately postponed the padding, to only apply it as necessary on each batch and avoid having over-long inputs with a lot of padding. This will speed up training by quite a bit, but note that if you're training on a TPU it can cause problems — TPUs prefer fixed shapes, even when that requires extra padding.

> [!TIP]
> 🚀 **Optimization Guide**: For more details on optimizing training performance, including padding strategies and TPU considerations, see the [🤗 Transformers performance documentation](https://huggingface.co/docs/transformers/main/en/performance).

To do this in practice, we have to define a collate function that will apply the correct amount of padding to the items of the dataset we want to batch together. Fortunately, the 🤗 Transformers library provides us with such a function via `DataCollatorWithPadding`. It takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs) and will do everything you need:

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

To test this new toy, let's grab a few samples from our training set that we would like to batch together. Here, we remove the columns `idx`, `sentence1`, and `sentence2` as they won't be needed and contain strings (and we can't create tensors with strings) and have a look at the lengths of each entry in the batch:

```py
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
```

```python out
[50, 59, 47, 67, 59, 50, 62, 32]
```

No surprise, we get samples of varying length, from 32 to 67. Dynamic padding means the samples in this batch should all be padded to a length of 67, the maximum length inside the batch. Without dynamic padding, all of the samples would have to be padded to the maximum length in the whole dataset, or the maximum length the model can accept. Let's double-check that our `data_collator` is dynamically padding the batch properly:

```py
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

```python out
{
    "attention_mask": torch.Size([8, 67]),
    "input_ids": torch.Size([8, 67]),
    "token_type_ids": torch.Size([8, 67]),
    "labels": torch.Size([8]),
}
```

Looking good! Now that we've gone from raw text to batches our model can deal with, we're ready to fine-tune it!

> [!TIP]
> ✏️ **Try it out!** Replicate the preprocessing on the GLUE SST-2 dataset. It's a little bit different since it's composed of single sentences instead of pairs, but the rest of what we did should look the same. For a harder challenge, try to write a preprocessing function that works on any of the GLUE tasks.
>
> 📖 **Additional Practice**: Check out these hands-on examples from the [🤗 Transformers examples](https://huggingface.co/docs/transformers/main/en/notebooks).

Perfect! Now that we have preprocessed our data with the latest best practices from the 🤗 Datasets library, we're ready to move on to training our model using the modern Trainer API. The next section will show you how to fine-tune your model effectively using the latest features and optimizations available in the Hugging Face ecosystem.

## Section Quiz[[section-quiz]]

Test your understanding of data processing concepts:

### 1. What is the main advantage of using `Dataset.map()` with `batched=True`?

<Question
	choices={[
		{
			text: "It uses less memory.",
			explain: "While it can be more memory efficient, this is not the main advantage."
		},
		{
			text: "It processes multiple examples at once, making tokenization much faster.",
			explain: "Correct! Processing in batches allows the fast tokenizer to work on multiple examples simultaneously, significantly improving speed.",
            correct: true
		},
		{
			text: "It automatically handles padding for you.",
			explain: "Batching doesn't automatically handle padding - that's done by the data collator."
		},
        {
			text: "It converts the data to PyTorch tensors.",
			explain: "The tensor conversion happens when you set the format, not during batched mapping."
		}
	]}
/>

### 2. Why do we use dynamic padding instead of padding all sequences to the maximum length in the dataset?

<Question
	choices={[
		{
			text: "Dynamic padding is required by the model architecture.",
			explain: "No, models can handle both fixed and dynamic padding."
		},
		{
			text: "It reduces computational overhead by only padding to the maximum length in each batch.",
			explain: "Correct! Dynamic padding avoids unnecessary computation on padding tokens by only padding to the batch maximum, not the dataset maximum.",
            correct: true
		},
		{
			text: "It improves model accuracy.",
			explain: "Padding strategy doesn't directly affect model accuracy."
		},
        {
			text: "It's required when using the DataCollatorWithPadding.",
			explain: "DataCollatorWithPadding enables dynamic padding, but you could still use fixed padding if desired."
		}
	]}
/>

### 3. What does the `token_type_ids` field represent in BERT tokenization?

<Question
	choices={[
		{
			text: "The position of each token in the sequence.",
			explain: "That would be position embeddings, not token_type_ids."
		},
		{
			text: "Which sentence each token belongs to when processing sentence pairs.",
			explain: "Correct! token_type_ids distinguish between the first sentence (0) and second sentence (1) in sentence pair tasks.",
            correct: true
		},
		{
			text: "The attention mask for each token.",
			explain: "The attention mask is a separate field that indicates which tokens to attend to."
		},
        {
			text: "The vocabulary ID of each token.",
			explain: "That's the input_ids field, not token_type_ids."
		}
	]}
/>

### 4. When loading a dataset with `load_dataset('glue', 'mrpc')`, what does the second argument specify?

<Question
	choices={[
		{
			text: "The version of the dataset to load.",
			explain: "Version specification uses different parameters."
		},
		{
			text: "The specific task or subset within the GLUE benchmark.",
			explain: "Correct! MRPC is one of the specific tasks within the larger GLUE benchmark collection.",
            correct: true
		},
		{
			text: "The split of the dataset (train/validation/test).",
			explain: "Splits are accessed after loading, not specified in the load_dataset call."
		},
        {
			text: "The format to return the data in.",
			explain: "Format is set using the set_format() method after loading."
		}
	]}
/>

### 5. What is the purpose of removing columns like 'sentence1' and 'sentence2' before training?

<Question
	choices={[
		{
			text: "To save memory during training.",
			explain: "While it does save some memory, this is not the main reason."
		},
		{
			text: "The model doesn't expect these raw text columns and would throw an error.",
			explain: "Correct! Models expect numerical tensors, not raw text strings. Keeping text columns would cause errors.",
            correct: true
		},
		{
			text: "These columns are not needed for evaluation.",
			explain: "While true, the main reason is that the model can't process raw text."
		},
        {
			text: "It improves training speed significantly.",
			explain: "The speed improvement is minimal compared to avoiding errors from incompatible data types."
		}
	]}
/>

> [!TIP]
> 💡 **Key Takeaways:**
> - Use `batched=True` with `Dataset.map()` for significantly faster preprocessing
> - Dynamic padding with `DataCollatorWithPadding` is more efficient than fixed-length padding
> - Always preprocess your data to match what your model expects (numerical tensors, correct column names)
> - The 🤗 Datasets library provides powerful tools for efficient data processing at scale


---

<!-- Section 3.3 -->

<FrameworkSwitchCourse {fw} />

# Fine-tuning a model with the Trainer API[[fine-tuning-a-model-with-the-trainer-api]]

<CourseFloatingBanner chapter={3}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter3/section3.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter3/section3.ipynb"},
]} />

<Youtube id="nvBXf7s7vTI"/>

🤗 Transformers provides a `Trainer` class to help you fine-tune any of the pretrained models it provides on your dataset with modern best practices. Once you've done all the data preprocessing work in the last section, you have just a few steps left to define the `Trainer`. The hardest part is likely to be preparing the environment to run `Trainer.train()`, as it will run very slowly on a CPU. If you don't have a GPU set up, you can get access to free GPUs or TPUs on [Google Colab](https://colab.research.google.com/).

> [!TIP]
> 📚 **Training Resources**: Before diving into training, familiarize yourself with the comprehensive [🤗 Transformers training guide](https://huggingface.co/docs/transformers/main/en/training) and explore practical examples in the [fine-tuning cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu).

The code examples below assume you have already executed the examples in the previous section. Here is a short summary recapping what you need:

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Training[[training]]

The first step before we can define our `Trainer` is to define a `TrainingArguments` class that will contain all the hyperparameters the `Trainer` will use for training and evaluation. The only argument you have to provide is a directory where the trained model will be saved, as well as the checkpoints along the way. For all the rest, you can leave the defaults, which should work pretty well for a basic fine-tuning.

```py
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

If you want to automatically upload your model to the Hub during training, pass along `push_to_hub=True` in the `TrainingArguments`. We will learn more about this in [Chapter 4](/course/chapter4/3)

> [!TIP]
> 🚀 **Advanced Configuration**: For detailed information on all available training arguments and optimization strategies, check out the [TrainingArguments documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) and the [training configuration cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu).

The second step is to define our model. As in the [previous chapter](/course/chapter2), we will use the `AutoModelForSequenceClassification` class, with two labels:

```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

You will notice that unlike in [Chapter 2](/course/chapter2), you get a warning after instantiating this pretrained model. This is because BERT has not been pretrained on classifying pairs of sentences, so the head of the pretrained model has been discarded and a new head suitable for sequence classification has been added instead. The warnings indicate that some weights were not used (the ones corresponding to the dropped pretraining head) and that some others were randomly initialized (the ones for the new head). It concludes by encouraging you to train the model, which is exactly what we are going to do now.

Once we have our model, we can define a `Trainer` by passing it all the objects constructed up to now — the `model`, the `training_args`, the training and validation datasets, our `data_collator`, and our `processing_class`. The `processing_class` parameter is a newer addition that tells the Trainer which tokenizer to use for processing:

```py
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)
```

When you pass a tokenizer as the `processing_class`, the default `data_collator` used by the `Trainer` will be a `DataCollatorWithPadding`. You can skip the `data_collator=data_collator` line in this case, but we included it here to show you this important part of the processing pipeline.

> [!TIP]
> 📖 **Learn More**: For comprehensive details on the Trainer class and its parameters, visit the [Trainer API documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) and explore advanced usage patterns in the [training cookbook recipes](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu).

To fine-tune the model on our dataset, we just have to call the `train()` method of our `Trainer`:

```py
trainer.train()
```

This will start the fine-tuning (which should take a couple of minutes on a GPU) and report the training loss every 500 steps. It won't, however, tell you how well (or badly) your model is performing. This is because:

1. We didn't tell the `Trainer` to evaluate during training by setting `eval_strategy` in `TrainingArguments` to either `"steps"` (evaluate every `eval_steps`) or `"epoch"` (evaluate at the end of each epoch).
2. We didn't provide the `Trainer` with a `compute_metrics()` function to calculate a metric during said evaluation (otherwise the evaluation would just have printed the loss, which is not a very intuitive number).


### Evaluation[[evaluation]]

Let's see how we can build a useful `compute_metrics()` function and use it the next time we train. The function must take an `EvalPrediction` object (which is a named tuple with a `predictions` field and a `label_ids` field) and will return a dictionary mapping strings to floats (the strings being the names of the metrics returned, and the floats their values). To get some predictions from our model, we can use the `Trainer.predict()` command:

```py
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```python out
(408, 2)(
    408,
)
```

The output of the `predict()` method is another named tuple with three fields: `predictions`, `label_ids`, and `metrics`. The `metrics` field will just contain the loss on the dataset passed, as well as some time metrics (how long it took to predict, in total and on average). Once we complete our `compute_metrics()` function and pass it to the `Trainer`, that field will also contain the metrics returned by `compute_metrics()`.

As you can see, `predictions` is a two-dimensional array with shape 408 x 2 (408 being the number of elements in the dataset we used). Those are the logits for each element of the dataset we passed to `predict()` (as you saw in the [previous chapter](/course/chapter2), all Transformer models return logits). To transform them into predictions that we can compare to our labels, we need to take the index with the maximum value on the second axis:

```py
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```

We can now compare those `preds` to the labels. To build our `compute_metric()` function, we will rely on the metrics from the 🤗 [Evaluate](https://github.com/huggingface/evaluate/) library. We can load the metrics associated with the MRPC dataset as easily as we loaded the dataset, this time with the `evaluate.load()` function. The object returned has a `compute()` method we can use to do the metric calculation:

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

```python out
{"accuracy": 0.8578431372549019, "f1": 0.8996539792387542}
```

> [!TIP]
> Learn about different evaluation metrics and strategies in the [🤗 Evaluate documentation](https://huggingface.co/docs/evaluate/).

The exact results you get may vary, as the random initialization of the model head might change the metrics it achieved. Here, we can see our model has an accuracy of 85.78% on the validation set and an F1 score of 89.97. Those are the two metrics used to evaluate results on the MRPC dataset for the GLUE benchmark. The table in the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf) reported an F1 score of 88.9 for the base model. That was the `uncased` model while we are currently using the `cased` model, which explains the better result.

Wrapping everything together, we get our `compute_metrics()` function:

```py
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

And to see it used in action to report metrics at the end of each epoch, here is how we define a new `Trainer` with this `compute_metrics()` function:

```py
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
```

Note that we create a new `TrainingArguments` with its `eval_strategy` set to `"epoch"` and a new model — otherwise, we would just be continuing the training of the model we have already trained. To launch a new training run, we execute:

```py
trainer.train()
```

This time, it will report the validation loss and metrics at the end of each epoch on top of the training loss. Again, the exact accuracy/F1 score you reach might be a bit different from what we found, because of the random head initialization of the model, but it should be in the same ballpark.

### Advanced Training Features[[advanced-training-features]]

The `Trainer` comes with many built-in features that make modern deep learning best practices accessible:

**Mixed Precision Training**: Use `fp16=True` in your training arguments for faster training and reduced memory usage:

```py
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    fp16=True,  # Enable mixed precision
)
```

**Gradient Accumulation**: For effective larger batch sizes when GPU memory is limited:

```py
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
)
```

**Learning Rate Scheduling**: The Trainer uses linear decay by default, but you can customize this:

```py
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # Try different schedulers
)
```

> [!TIP]
> 🎯 **Performance Optimization**: For more advanced training techniques including distributed training, memory optimization, and hardware-specific optimizations, explore the [🤗 Transformers performance guide](https://huggingface.co/docs/transformers/main/en/performance).

The `Trainer` will work out of the box on multiple GPUs or TPUs and provides lots of options for distributed training. We will go over everything it supports in Chapter 10.

This concludes the introduction to fine-tuning using the `Trainer` API. An example of doing this for most common NLP tasks will be given in [Chapter 7](/course/chapter7), but for now let's look at how to do the same thing with a pure PyTorch training loop.

> [!TIP]
> 📝 **More Examples**: Check out the comprehensive collection of [🤗 Transformers notebooks](https://huggingface.co/docs/transformers/main/en/notebooks).

## Section Quiz[[section-quiz]]

Test your understanding of the Trainer API and fine-tuning concepts:

### 1. What is the purpose of the <code>processing_class</code> parameter in the Trainer?

<Question
	choices={[
		{
			text: "It specifies which model architecture to use.",
			explain: "Model architecture is specified when loading the model, not in the Trainer."
		},
		{
			text: "It tells the Trainer which tokenizer to use for processing data.",
			explain: "The processing_class parameter is a modern addition that helps the Trainer know which tokenizer to use.",
            correct: true
		},
		{
			text: "It determines the batch size for training.",
			explain: "Batch size is set in TrainingArguments, not through processing_class."
		},
        {
			text: "It controls the evaluation frequency.",
			explain: "Evaluation frequency is controlled by eval_strategy in TrainingArguments."
		}
	]}
/>

### 2. Which TrainingArguments parameter controls how often evaluation occurs during training?

<Question
	choices={[
		{
			text: "eval_frequency",
			explain: "There's no eval_frequency parameter in TrainingArguments."
		},
		{
			text: "eval_strategy",
			explain: "eval_strategy can be set to 'epoch', 'steps', or 'no' to control evaluation timing.",
            correct: true
		},
		{
			text: "evaluation_steps",
			explain: "eval_steps sets the number of steps between evaluations, but eval_strategy determines if/when evaluation happens."
		},
        {
			text: "do_eval",
			explain: "There's no do_eval parameter in modern TrainingArguments."
		}
	]}
/>

### 3. What does <code>fp16=True</code> in TrainingArguments enable?

<Question
	choices={[
		{
			text: "16-bit integer precision for faster training.",
			explain: "fp16 refers to floating-point precision, not integer precision."
		},
		{
			text: "Mixed precision training with 16-bit floating-point numbers for faster training and reduced memory usage.",
			explain: "Mixed precision training uses 16-bit floats for forward pass and 32-bit for gradients, improving speed and reducing memory usage.",
            correct: true
		},
		{
			text: "Training for exactly 16 epochs.",
			explain: "fp16 has nothing to do with the number of epochs."
		},
        {
			text: "Using 16 GPUs for distributed training.",
			explain: "The number of GPUs is not controlled by the fp16 parameter."
		}
	]}
/>

### 4. What is the role of the <code>compute_metrics</code> function in the Trainer?

<Question
	choices={[
		{
			text: "It calculates the loss during training.",
			explain: "Loss calculation is handled automatically by the model, not by compute_metrics."
		},
		{
			text: "It converts logits to predictions and calculates evaluation metrics like accuracy and F1.",
			explain: "compute_metrics takes predictions and labels, then returns metrics for evaluation.",
            correct: true
		},
		{
			text: "It determines which optimizer to use.",
			explain: "Optimizer selection is not handled by compute_metrics."
		},
        {
			text: "It preprocesses the training data.",
			explain: "Data preprocessing is done before training, not by compute_metrics during evaluation."
		}
	]}
/>

### 5. What happens when you don't provide an <code>eval_dataset</code> to the Trainer?

<Question
	choices={[
		{
			text: "Training will fail with an error.",
			explain: "Training can proceed without an eval_dataset, though you won't get evaluation metrics."
		},
		{
			text: "The Trainer will automatically split the training data for evaluation.",
			explain: "The Trainer doesn't automatically create validation splits."
		},
		{
			text: "You won't get evaluation metrics during training, but training will still work.",
			explain: "Evaluation is optional - you can train without it, but you won't see validation metrics.",
            correct: true
		},
        {
			text: "The model will use the training data for evaluation.",
			explain: "The Trainer won't automatically use training data for evaluation - it simply won't evaluate."
		}
	]}
/>

### 6. What is gradient accumulation and how do you enable it?

<Question
	choices={[
		{
			text: "It saves gradients to disk, enabled with save_gradients=True.",
			explain: "Gradient accumulation doesn't involve saving gradients to disk."
		},
		{
			text: "It accumulates gradients over multiple batches before updating, enabled with gradient_accumulation_steps.",
			explain: "This allows you to simulate larger batch sizes by accumulating gradients over multiple forward passes.",
            correct: true
		},
		{
			text: "It speeds up gradient computation, enabled automatically with fp16.",
			explain: "While fp16 can speed up training, gradient accumulation is a separate technique."
		},
        {
			text: "It prevents gradient overflow, enabled with gradient_clipping=True.",
			explain: "That describes gradient clipping, not gradient accumulation."
		}
	]}
/>

> [!TIP]
> 💡 **Key Takeaways:**
> - The `Trainer` API provides a high-level interface that handles most training complexity
> - Use `processing_class` to specify your tokenizer for proper data handling
> - `TrainingArguments` controls all aspects of training: learning rate, batch size, evaluation strategy, and optimizations
> - `compute_metrics` enables custom evaluation metrics beyond just training loss
> - Modern features like mixed precision (`fp16=True`) and gradient accumulation can significantly improve training efficiency



---

<!-- Section 3.4 -->

# A full training loop[[a-full-training]]

<CourseFloatingBanner chapter={3}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter3/section4.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter3/section4.ipynb"},
]} />

<Youtube id="Dh9CL8fyG80"/>

Now we'll see how to achieve the same results as we did in the last section without using the `Trainer` class, implementing a training loop from scratch with modern PyTorch best practices. Again, we assume you have done the data processing in section 2. Here is a short summary covering everything you will need:

> [!TIP]
> 🏗️ **Training from Scratch**: This section builds on the previous content. For comprehensive guidance on PyTorch training loops and best practices, check out the [🤗 Transformers training documentation](https://huggingface.co/docs/transformers/main/en/training#train-in-native-pytorch) and the [custom training cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu#model).

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Prepare for training[[prepare-for-training]]

Before actually writing our training loop, we will need to define a few objects. The first ones are the dataloaders we will use to iterate over batches. But before we can define those dataloaders, we need to apply a bit of postprocessing to our `tokenized_datasets`, to take care of some things that the `Trainer` did for us automatically. Specifically, we need to:

- Remove the columns corresponding to values the model does not expect (like the `sentence1` and `sentence2` columns).
- Rename the column `label` to `labels` (because the model expects the argument to be named `labels`).
- Set the format of the datasets so they return PyTorch tensors instead of lists.

Our `tokenized_datasets` has one method for each of those steps:

```py
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```

We can then check that the result only has columns that our model will accept:

```python
["attention_mask", "input_ids", "labels", "token_type_ids"]
```

Now that this is done, we can easily define our dataloaders:

```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

To quickly check there is no mistake in the data processing, we can inspect a batch like this:

```py
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

```python out
{
    "attention_mask": torch.Size([8, 65]),
    "input_ids": torch.Size([8, 65]),
    "labels": torch.Size([8]),
    "token_type_ids": torch.Size([8, 65]),
}
```

Note that the actual shapes will probably be slightly different for you since we set `shuffle=True` for the training dataloader and we are padding to the maximum length inside the batch.

Now that we're completely finished with data preprocessing (a satisfying yet elusive goal for any ML practitioner), let's turn to the model. We instantiate it exactly as we did in the previous section:

```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

To make sure that everything will go smoothly during training, we pass our batch to this model:

```py
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

```python out
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

All 🤗 Transformers models will return the loss when `labels` are provided, and we also get the logits (two for each input in our batch, so a tensor of size 8 x 2).

We're almost ready to write our training loop! We're just missing two things: an optimizer and a learning rate scheduler. Since we are trying to replicate what the `Trainer` was doing by hand, we will use the same defaults. The optimizer used by the `Trainer` is `AdamW`, which is the same as Adam, but with a twist for weight decay regularization (see ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) by Ilya Loshchilov and Frank Hutter):

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

> [!TIP]
> 💡 **Modern Optimization Tips**: For even better performance, you can try:
> - **AdamW with weight decay**: `AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)`
> - **8-bit Adam**: Use `bitsandbytes` for memory-efficient optimization
> - **Different learning rates**: Lower learning rates (1e-5 to 3e-5) often work better for large models
>
> 🚀 **Optimization Resources**: Learn more about optimizers and training strategies in the [🤗 Transformers optimization guide](https://huggingface.co/docs/transformers/main/en/performance#optimizer).

Finally, the learning rate scheduler used by default is just a linear decay from the maximum value (5e-5) to 0. To properly define it, we need to know the number of training steps we will take, which is the number of epochs we want to run multiplied by the number of training batches (which is the length of our training dataloader). The `Trainer` uses three epochs by default, so we will follow that:

```py
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```

```python out
1377
```

### The training loop[[the-training-loop]]

One last thing: we will want to use the GPU if we have access to one (on a CPU, training might take several hours instead of a couple of minutes). To do this, we define a `device` we will put our model and our batches on:

```py
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```

```python out
device(type="cuda")
```

We are now ready to train! To get some sense of when training will be finished, we add a progress bar over our number of training steps, using the `tqdm` library:

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

> [!TIP]
> 💡 **Modern Training Optimizations**: To make your training loop even more efficient, consider:
>
> - **Gradient Clipping**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
> - **Mixed Precision**: Use `torch.cuda.amp.autocast()` and `GradScaler` for faster training
> - **Gradient Accumulation**: Accumulate gradients over multiple batches to simulate larger batch sizes
> - **Checkpointing**: Save model checkpoints periodically to resume training if interrupted
>
> 🔧 **Implementation Guide**: For detailed examples of these optimizations, see the [🤗 Transformers efficient training guide](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one) and the [range of optimizers](https://huggingface.co/docs/transformers/main/en/optimizers).

You can see that the core of the training loop looks a lot like the one in the introduction. We didn't ask for any reporting, so this training loop will not tell us anything about how the model fares. We need to add an evaluation loop for that.


### The evaluation loop[[the-evaluation-loop]]

As we did earlier, we will use a metric provided by the 🤗 Evaluate library. We've already seen the `metric.compute()` method, but metrics can actually accumulate batches for us as we go over the prediction loop with the method `add_batch()`. Once we have accumulated all the batches, we can get the final result with `metric.compute()`. Here's how to implement all of this in an evaluation loop:

> [!TIP]
> 📊 **Evaluation Best Practices**: For more sophisticated evaluation strategies and metrics, explore the [🤗 Evaluate documentation](https://huggingface.co/docs/evaluate/) and the [comprehensive evaluation cookbook](https://github.com/huggingface/evaluation-guidebook).

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

```python out
{"accuracy": 0.8431372549019608, "f1": 0.8907849829351535}
```

Again, your results will be slightly different because of the randomness in the model head initialization and the data shuffling, but they should be in the same ballpark.

> [!TIP]
> ✏️ **Try it out!** Modify the previous training loop to fine-tune your model on the SST-2 dataset.

### Supercharge your training loop with 🤗 Accelerate[[supercharge-your-training-loop-with-accelerate]]

<Youtube id="s7dy8QRgjJ0" />

The training loop we defined earlier works fine on a single CPU or GPU. But using the [🤗 Accelerate](https://github.com/huggingface/accelerate) library, with just a few adjustments we can enable distributed training on multiple GPUs or TPUs. 🤗 Accelerate handles the complexity of distributed training, mixed precision, and device placement automatically. Starting from the creation of the training and validation dataloaders, here is what our manual training loop looks like:

> [!TIP]
> ⚡ **Accelerate Deep Dive**: Learn everything about distributed training, mixed precision, and hardware optimization in the [🤗 Accelerate documentation](https://huggingface.co/docs/accelerate/) and explore practical examples in the [transformers documentation](https://huggingface.co/docs/transformers/main/en/accelerate).

```py
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

The first line to add is the import line. The second line instantiates an `Accelerator` object that will look at the environment and initialize the proper distributed setup. 🤗 Accelerate handles the device placement for you, so you can remove the lines that put the model on the device (or, if you prefer, change them to use `accelerator.device` instead of `device`).

Then the main bulk of the work is done in the line that sends the dataloaders, the model, and the optimizer to `accelerator.prepare()`. This will wrap those objects in the proper container to make sure your distributed training works as intended. The remaining changes to make are removing the line that puts the batch on the `device` (again, if you want to keep this you can just change it to use `accelerator.device`) and replacing `loss.backward()` with `accelerator.backward(loss)`.

> [!TIP]
> ⚠️ In order to benefit from the speed-up offered by Cloud TPUs, we recommend padding your samples to a fixed length with the `padding="max_length"` and `max_length` arguments of the tokenizer.

If you'd like to copy and paste it to play around, here's what the complete training loop looks like with 🤗 Accelerate:

```py
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Putting this in a `train.py` script will make that script runnable on any kind of distributed setup. To try it out in your distributed setup, run the command:

```bash
accelerate config
```

which will prompt you to answer a few questions and dump your answers in a configuration file used by this command:

```
accelerate launch train.py
```

which will launch the distributed training.

If you want to try this in a Notebook (for instance, to test it with TPUs on Colab), just paste the code in a `training_function()` and run a last cell with:

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

You can find more examples in the [🤗 Accelerate repo](https://github.com/huggingface/accelerate/tree/main/examples).

> [!TIP]
> 🌐 **Distributed Training**: For comprehensive coverage of multi-GPU and multi-node training, check out the [🤗 Transformers distributed training guide](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many) and the [scaling training cookbook](https://huggingface.co/docs/transformers/main/en/accelerate).

### Next Steps and Best Practices[[next-steps-and-best-practices]]

Now that you've learned how to implement training from scratch, here are some additional considerations for production use:

**Model Evaluation**: Always evaluate your model on multiple metrics, not just accuracy. Use the 🤗 Evaluate library for comprehensive evaluation.

**Hyperparameter Tuning**: Consider using libraries like Optuna or Ray Tune for systematic hyperparameter optimization.

**Model Monitoring**: Track training metrics, learning curves, and validation performance throughout training.

**Model Sharing**: Once trained, share your model on the Hugging Face Hub to make it available to the community.

**Efficiency**: For large models, consider techniques like gradient checkpointing, parameter-efficient fine-tuning (LoRA, AdaLoRA), or quantization methods.

This concludes our deep dive into fine-tuning with custom training loops. The skills you've learned here will serve you well when you need full control over the training process or want to implement custom training logic that goes beyond what the `Trainer` API offers.

## Section Quiz[[section-quiz]]

Test your understanding of custom training loops and advanced training techniques:

### 1. What is the main difference between Adam and AdamW optimizers?

<Question
	choices={[
		{
			text: "AdamW uses a different learning rate schedule.",
			explain: "Learning rate scheduling is separate from the optimizer choice."
		},
		{
			text: "AdamW includes decoupled weight decay regularization.",
			explain: "Correct! AdamW separates weight decay from the gradient-based parameter updates, leading to better regularization.",
            correct: true
		},
		{
			text: "AdamW only works with transformer models.",
			explain: "AdamW can be used with any model architecture, not just transformers."
		},
        {
			text: "AdamW requires less memory than Adam.",
			explain: "Both optimizers have similar memory requirements."
		}
	]}
/>

### 2. In a training loop, what is the correct order of operations?

<Question
	choices={[
		{
			text: "Forward pass → Backward pass → Optimizer step → Zero gradients",
			explain: "Close, but you should zero gradients before the next forward pass to avoid accumulating old gradients."
		},
		{
			text: "Forward pass → Backward pass → Optimizer step → Scheduler step → Zero gradients",
			explain: "Correct! This is the proper order: compute loss, compute gradients, update parameters, update learning rate, then clear gradients.",
            correct: true
		},
		{
			text: "Zero gradients → Forward pass → Optimizer step → Backward pass",
			explain: "The backward pass must come after the forward pass to compute gradients from the loss."
		},
        {
			text: "Forward pass → Zero gradients → Backward pass → Optimizer step",
			explain: "Zeroing gradients before backward pass would eliminate the gradients you just computed."
		}
	]}
/>

### 3. What does the 🤗 Accelerate library primarily help with?

<Question
	choices={[
		{
			text: "Making your models train faster by optimizing the forward pass.",
			explain: "Accelerate doesn't optimize the model architecture itself."
		},
		{
			text: "Automatically selecting the best hyperparameters.",
			explain: "Accelerate doesn't do hyperparameter optimization."
		},
		{
			text: "Enabling distributed training across multiple GPUs/TPUs with minimal code changes.",
			explain: "Correct! Accelerate handles distributed training complexity, allowing your code to run on single or multiple devices seamlessly.",
            correct: true
		},
        {
			text: "Converting models to different frameworks like TensorFlow.",
			explain: "Accelerate works within PyTorch and doesn't convert between frameworks."
		}
	]}
/>

### 4. Why do we move batches to the device in a training loop?

<Question
	choices={[
		{
			text: "To make the training faster.",
			explain: "While it can affect speed, the main reason is compatibility."
		},
		{
			text: "Because the model and data must be on the same device (CPU/GPU) for computation.",
			explain: "Correct! PyTorch requires tensors to be on the same device for operations to work.",
            correct: true
		},
		{
			text: "To save memory.",
			explain: "Moving to device doesn't inherently save memory."
		},
        {
			text: "It's required by the DataLoader.",
			explain: "DataLoader doesn't require specific device placement."
		}
	]}
/>

### 5. What does `model.eval()` do before evaluation?

<Question
	choices={[
		{
			text: "It freezes the model parameters so they can't be updated.",
			explain: "model.eval() doesn't freeze parameters - that would be done by setting requires_grad=False."
		},
		{
			text: "It changes the behavior of layers like dropout and batch normalization for inference.",
			explain: "Correct! eval() mode disables dropout and uses running statistics for batch norm instead of computing them from the current batch.",
            correct: true
		},
		{
			text: "It enables gradient computation for evaluation metrics.",
			explain: "Actually, we typically use torch.no_grad() during evaluation to disable gradient computation."
		},
        {
			text: "It automatically calculates evaluation metrics.",
			explain: "model.eval() only changes layer behavior - you still need to implement metric calculation separately."
		}
	]}
/>

### 6. What is the purpose of `torch.no_grad()` during evaluation?

<Question
	choices={[
		{
			text: "To prevent the model from making predictions.",
			explain: "torch.no_grad() doesn't prevent predictions, just gradient computation."
		},
		{
			text: "To save memory and speed up computation by disabling gradient tracking.",
			explain: "Correct! Since we don't need gradients for evaluation, disabling them saves memory and computation.",
            correct: true
		},
		{
			text: "To enable evaluation mode for the model.",
			explain: "Evaluation mode is enabled with model.eval(), not torch.no_grad()."
		},
        {
			text: "To ensure consistent results across runs.",
			explain: "Reproducibility is handled by setting random seeds, not torch.no_grad()."
		}
	]}
/>

### 7. What changes when you use 🤗 Accelerate in your training loop?

<Question
	choices={[
		{
			text: "You must rewrite your entire training loop from scratch.",
			explain: "Accelerate requires minimal changes to existing PyTorch code."
		},
		{
			text: "You wrap key objects with accelerator.prepare() and use accelerator.backward() instead of loss.backward().",
			explain: "Correct! These are the main changes - prepare your objects and use accelerator.backward() for proper distributed training.",
            correct: true
		},
		{
			text: "You need to specify the number of GPUs in your code.",
			explain: "Accelerate automatically detects available hardware."
		},
        {
			text: "You must use a different optimizer and scheduler.",
			explain: "You can use the same optimizers and schedulers with Accelerate."
		}
	]}
/>

> [!TIP]
> 💡 **Key Takeaways:**
> - Manual training loops give you complete control but require understanding of the proper sequence: forward → backward → optimizer step → scheduler step → zero gradients
> - AdamW with weight decay is the recommended optimizer for transformer models
> - Always use `model.eval()` and `torch.no_grad()` during evaluation for correct behavior and efficiency
> - 🤗 Accelerate makes distributed training accessible with minimal code changes
> - Device management (moving tensors to GPU/CPU) is crucial for PyTorch operations
> - Modern techniques like mixed precision, gradient accumulation, and gradient clipping can significantly improve training efficiency


---

<!-- Section 3.5 -->

# Understanding Learning Curves[[understanding-learning-curves]]

<CourseFloatingBanner chapter={3}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter3/section7.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter3/section7.ipynb"},
]} />

Now that you've learned how to implement fine-tuning using both the `Trainer` API and custom training loops, it's crucial to understand how to interpret the results. Learning curves are invaluable tools that help you evaluate your model's performance during training and identify potential issues before they reduce performance.

In this section, we'll explore how to read and interpret accuracy and loss curves, understand what different curve shapes tell us about our model's behavior, and learn how to address common training issues.

## What are Learning Curves?[[what-are-learning-curves]]

Learning curves are visual representations of your model's performance metrics over time during training. The two most important curves to monitor are:

- **Loss curves**: Show how the model's error (loss) changes over training steps or epochs
- **Accuracy curves**: Show the percentage of correct predictions over training steps or epochs

These curves help us understand whether our model is learning effectively and can guide us in making adjustments to improve performance. In Transformers, these metrics are individually computed for each batch and then logged to the disk. We can then use libraries like [Weights & Biases](https://wandb.ai/) to visualize these curves and track our model's performance over time.

### Loss Curves[[loss-curves]]

The loss curve shows how the model's error decreases over time. In a typical successful training run, you'll see a curve similar to the one below:

![Loss Curve](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/1.png)

- **High initial loss**: The model starts without optimization, so predictions are initially poor
- **Decreasing loss**: As training progresses, the loss should generally decrease
- **Convergence**: Eventually, the loss stabilizes at a low value, indicating that the model has learned the patterns in the data

As in previous chapters, we can use the `Trainer` API to track these metrics and visualize them in a dashboard. Below is an example of how to do this with Weights & Biases.

```python
# Example of tracking loss during training with the Trainer
from transformers import Trainer, TrainingArguments
import wandb

# Initialize Weights & Biases for experiment tracking
wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # Log metrics every 10 steps
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="wandb",  # Send logs to Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and automatically log metrics
trainer.train()
```

### Accuracy Curves[[accuracy-curves]]

The accuracy curve shows the percentage of correct predictions over time. Unlike loss curves, accuracy curves should generally increase as the model learns and can typically include more steps than the loss curve.

![Accuracy Curve](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/2.png)

- **Start low**: Initial accuracy should be low, as the model has not yet learned the patterns in the data
- **Increase with training**: Accuracy should generally improve as the model learns if it is able to learn the patterns in the data
- **May show plateaus**: Accuracy often increases in discrete jumps rather than smoothly, as the model makes predictions that are close to the true labels

> [!TIP]
> 💡 **Why Accuracy Curves Are "Steppy"**: Unlike loss, which is continuous, accuracy is calculated by comparing discrete predictions to true labels. Small improvements in model confidence might not change the final prediction, causing accuracy to remain flat until a threshold is crossed.

### Convergence[[convergence]]

Convergence occurs when the model's performance stabilizes and the loss and accuracy curves level off. This is a sign that the model has learned the patterns in the data and is ready to be used. In simple terms, we are aiming for the model to converge to a stable performance every time we train it.

![Convergence](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/4.png)

Once models have converged, we can use them to make predictions on new data and refer to evaluation metrics to understand how well the model is performing.

## Interpreting Learning Curve Patterns[[interpreting-learning-curve-patterns]]

Different curve shapes reveal different aspects of your model's training. Let's examine the most common patterns and what they mean.

### Healthy Learning Curves[[healthy-learning-curves]]

A well-behaved training run typically shows curve shapes similar to the one below:

![Healthy Loss Curve](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/5.png)

Let's look at the illustration above. It displays both the loss curve (on the left) and the corresponding accuracy curve (on the right). These curves have distinct characteristics.

The loss curve shows the value of the model's loss over time. Initially, the loss is high and then it gradually decreases, indicating that the model is improving. A decrease in the loss value suggests that the model is making better predictions, as the loss represents the error between the predicted output and the true output.

Now let's shift our focus to the accuracy curve. It represents the model's accuracy over time. The accuracy curve begins at a low value and increases as training progresses. Accuracy measures the proportion of correctly classified instances. So, as the accuracy curve rises, it signifies that the model is making more correct predictions.

One notable difference between the curves is the smoothness and the presence of "plateaus" on the accuracy curve. While the loss decreases smoothly, the plateaus on the accuracy curve indicate discrete jumps in accuracy instead of a continuous increase. This behavior is attributed to how accuracy is measured. The loss can improve if the model's output gets closer to the target, even if the final prediction is still incorrect. Accuracy, however, only improves when the prediction crosses the threshold to be correct.

For example, in a binary classifier distinguishing cats (0) from dogs (1), if the model predicts 0.3 for an image of a dog (true value 1), this is rounded to 0 and is an incorrect classification. If in the next step it predicts 0.4, it's still incorrect. The loss will have decreased because 0.4 is closer to 1 than 0.3, but the accuracy remains unchanged, creating a plateau. The accuracy will only jump up when the model predicts a value greater than 0.5 that gets rounded to 1.

> [!TIP]
> **Characteristics of healthy curves:**
> - **Smooth decline in loss**: Both training and validation loss decrease steadily
> - **Close training/validation performance**: Small gap between training and validation metrics
> - **Convergence**: Curves level off, indicating the model has learned the patterns

### Practical Examples[[practical-examples]]

Let's work through some practical examples of learning curves. First, we will highlight some approaches to monitor the learning curves during training. Below, we will break down the different patterns that can be observed in the learning curves.

#### During Training[[during-training]]

During the training process (after you've hit `trainer.train()`), you can monitor these key indicators:

1. **Loss convergence**: Is the loss still decreasing or has it plateaued?
2. **Overfitting signs**: Is validation loss starting to increase while training loss decreases?
3. **Learning rate**: Are the curves too erratic (LR too high) or too flat (LR too low)?
4. **Stability**: Are there sudden spikes or drops that indicate problems?

#### After Training[[after-training]]

After the training process is complete, you can analyze the complete curves to understand the model's performance.

1. **Final performance**: Did the model reach acceptable performance levels?
2. **Efficiency**: Could the same performance be achieved with fewer epochs?
3. **Generalization**: How close are training and validation performance?
4. **Trends**: Would additional training likely improve performance?

> [!TIP]
> 🔍 **W&B Dashboard Features**: Weights & Biases automatically creates beautiful, interactive plots of your learning curves. You can:
> - Compare multiple runs side by side
> - Add custom metrics and visualizations  
> - Set up alerts for anomalous behavior
> - Share results with your team
>
> Learn more in the [Weights & Biases documentation](https://docs.wandb.ai/).

#### Overfitting[[overfitting]]

Overfitting occurs when the model learns too much from the training data and is unable to generalize to different data (represented by the validation set).

![Overfitting](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/10.png)

**Symptoms:**

- Training loss continues to decrease while validation loss increases or plateaus
- Large gap between training and validation accuracy
- Training accuracy much higher than validation accuracy

**Solutions for overfitting:**
- **Regularization**: Add dropout, weight decay, or other regularization techniques
- **Early stopping**: Stop training when validation performance stops improving
- **Data augmentation**: Increase training data diversity
- **Reduce model complexity**: Use a smaller model or fewer parameters

In the sample below, we use early stopping to prevent overfitting. We set the `early_stopping_patience` to 3, which means that if the validation loss does not improve for 3 consecutive epochs, the training will be stopped.

```python
# Example of detecting overfitting with early stopping
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=10,  # Set high, but we'll stop early
)

# Add early stopping to prevent overfitting
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

#### 2. Underfitting[[underfitting]]

Underfitting occurs when the model is too simple to capture the underlying patterns in the data. This can happen for several reasons:

- The model is too small or lacks capacity to learn the patterns
- The learning rate is too low, causing slow learning
- The dataset is too small or not representative of the problem
- The model is not properly regularized

![Underfitting](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/7.png)

**Symptoms:**
- Both training and validation loss remain high
- Model performance plateaus early in training
- Training accuracy is lower than expected

**Solutions for underfitting:**
- **Increase model capacity**: Use a larger model or more parameters
- **Train longer**: Increase the number of epochs
- **Adjust learning rate**: Try different learning rates
- **Check data quality**: Ensure your data is properly preprocessed

In the sample below, we train for more epochs to see if the model can learn the patterns in the data.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    -num_train_epochs=5,
    +num_train_epochs=10,
)
```

#### 3. Erratic Learning Curves[[erratic-learning-curves]]

Erratic learning curves occur when the model is not learning effectively. This can happen for several reasons:

- The learning rate is too high, causing the model to overshoot the optimal parameters
- The batch size is too small, causing the model to learn slowly
- The model is not properly regularized, causing it to overfit to the training data
- The dataset is not properly preprocessed, causing the model to learn from noise

![Erratic Learning Curves](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/3.png)

**Symptoms:**
- Frequent fluctuations in loss or accuracy
- Curves show high variance or instability
- Performance oscillates without clear trend

Both training and validation curves show erratic behavior.

![Erratic Learning Curves](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter3/9.png)

**Solutions for erratic curves:**
- **Lower learning rate**: Reduce step size for more stable training
- **Increase batch size**: Larger batches provide more stable gradients
- **Gradient clipping**: Prevent exploding gradients
- **Better data preprocessing**: Ensure consistent data quality

In the sample below, we lower the learning rate and increase the batch size.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    -learning_rate=1e-5,
    +learning_rate=1e-4,
    -per_device_train_batch_size=16,
    +per_device_train_batch_size=32,
)
```

## Key Takeaways[[key-takeaways]]

Understanding learning curves is crucial for becoming an effective machine learning practitioner. These visual tools provide immediate feedback about your model's training progress and help you make informed decisions about when to stop training, adjust hyperparameters, or try different approaches. With practice, you'll develop an intuitive understanding of what healthy learning curves look like and how to address issues when they arise. 

> [!TIP]
> 💡 **Key Takeaways:**
> - Learning curves are essential tools for understanding model training progress
> - Monitor both loss and accuracy curves, but remember they have different characteristics
> - Overfitting shows as diverging training/validation performance
> - Underfitting shows as poor performance on both training and validation data
> - Tools like Weights & Biases make it easy to track and analyze learning curves
> - Early stopping and proper regularization can address most common training issues
>
> 🔬 **Next Steps**: Practice analyzing learning curves on your own fine-tuning experiments. Try different hyperparameters and observe how they affect the curve shapes. This hands-on experience is the best way to develop intuition for reading training progress.

## Section Quiz[[section-quiz]]

Test your understanding of learning curves and training analysis:

### 1. What does it typically mean when training loss decreases but validation loss starts increasing?

<Question
	choices={[
		{
			text: "The model is learning successfully and will continue to improve.",
			explain: "If validation loss is increasing while training loss decreases, this indicates a problem, not success."
		},
		{
			text: "The model is overfitting to the training data.",
			explain: "Correct! This is a classic sign of overfitting - the model performs well on training data but poorly on unseen validation data.",
            correct: true
		},
		{
			text: "The learning rate is too low.",
			explain: "A low learning rate would cause slow learning, not the divergence between training and validation performance."
		},
        {
			text: "The dataset is too small.",
			explain: "While small datasets can contribute to overfitting, this specific pattern is the definition of overfitting regardless of dataset size."
		}
	]}
/>

### 2. Why do accuracy curves often show a "steppy" or plateau-like pattern rather than smooth increases?

<Question
	choices={[
		{
			text: "There's an error in the accuracy calculation.",
			explain: "The steppy pattern is normal and expected, not an error."
		},
		{
			text: "Accuracy is a discrete metric that only changes when predictions cross decision boundaries.",
			explain: "Correct! Unlike loss, accuracy depends on discrete prediction decisions, so small improvements in confidence may not change the final accuracy until a threshold is crossed.",
            correct: true
		},
		{
			text: "The model is not learning effectively.",
			explain: "Steppy accuracy curves are normal even when the model is learning well."
		},
        {
			text: "The batch size is too small.",
			explain: "Batch size affects training stability but doesn't explain the inherently discrete nature of accuracy metrics."
		}
	]}
/>

### 3. What is the best approach when you observe erratic, highly fluctuating learning curves?

<Question
	choices={[
		{
			text: "Increase the learning rate to speed up convergence.",
			explain: "Increasing the learning rate would likely make the fluctuations worse."
		},
		{
			text: "Reduce the learning rate and possibly increase the batch size.",
			explain: "Correct! Lower learning rates and larger batch sizes typically lead to more stable training.",
            correct: true
		},
		{
			text: "Stop training immediately as the model won't improve.",
			explain: "Erratic curves can often be fixed with hyperparameter adjustments."
		},
        {
			text: "Switch to a completely different model architecture.",
			explain: "This is premature - erratic curves are usually fixable with hyperparameter tuning."
		}
	]}
/>

### 4. When should you consider using early stopping?

<Question
	choices={[
		{
			text: "Always, as it prevents any form of overfitting.",
			explain: "Early stopping is useful but not always necessary, especially if other regularization methods are working."
		},
		{
			text: "When validation performance stops improving or starts degrading.",
			explain: "Correct! Early stopping helps prevent overfitting by stopping training when the model no longer generalizes better.",
            correct: true
		},
		{
			text: "Only when training loss is still decreasing rapidly.",
			explain: "If training loss is decreasing rapidly and validation performance is good, you might want to continue training."
		},
        {
			text: "Never, as it prevents the model from reaching its full potential.",
			explain: "Early stopping is a valuable technique that often improves final model performance by preventing overfitting."
		}
	]}
/>

### 5. What indicates that your model might be underfitting?

<Question
	choices={[
		{
			text: "Training accuracy is much higher than validation accuracy.",
			explain: "This describes overfitting, not underfitting."
		},
		{
			text: "Both training and validation performance are poor and plateau early.",
			explain: "Correct! Underfitting occurs when the model lacks capacity to learn the patterns, resulting in poor performance on both training and validation data.",
            correct: true
		},
		{
			text: "The learning curves are very smooth with no fluctuations.",
			explain: "Smooth curves are generally good and don't indicate underfitting."
		},
        {
			text: "Validation loss is decreasing faster than training loss.",
			explain: "This would actually be a positive sign, not a problem."
		}
	]}
/>



---

<!-- Section 3.6 -->

<FrameworkSwitchCourse {fw} />

# Fine-tuning, Check![[fine-tuning-check]]

<CourseFloatingBanner
    chapter={3}
    classNames="absolute z-10 right-0 top-0"
/>

That was comprehensive! In the first two chapters you learned about models and tokenizers, and now you know how to fine-tune them for your own data using modern best practices. To recap, in this chapter you:

* Learned about datasets on the [Hub](https://huggingface.co/datasets) and modern data processing techniques
* Learned how to load and preprocess datasets efficiently, including using dynamic padding and data collators
* Implemented fine-tuning and evaluation using the high-level `Trainer` API with the latest features
* Implemented a complete custom training loop from scratch with PyTorch
* Used 🤗 Accelerate to make your training code work seamlessly on multiple GPUs or TPUs
* Applied modern optimization techniques like mixed precision training and gradient accumulation

> [!TIP]
> 🎉 **Congratulations!** You've mastered the fundamentals of fine-tuning transformer models. You're now ready to tackle real-world ML projects!
>
> 📖 **Continue Learning**: Explore these resources to deepen your knowledge:
> - [🤗 Transformers task guides](https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification) for specific NLP tasks
> - [🤗 Transformers examples](https://huggingface.co/docs/transformers/main/en/notebooks) for comprehensive notebooks
>
> 🚀 **Next Steps**: 
> - Try fine-tuning on your own dataset using the techniques you've learned
> - Experiment with different model architectures available on the [Hugging Face Hub](https://huggingface.co/models)
> - Join the [Hugging Face community](https://discuss.huggingface.co/) to share your projects and get help

This is just the beginning of your journey with 🤗 Transformers. In the next chapter, we'll explore how to share your models and tokenizers with the community and contribute to the ever-growing ecosystem of pretrained models.

The skills you've developed here - data preprocessing, training configuration, evaluation, and optimization - are fundamental to any machine learning project. Whether you're working on text classification, named entity recognition, question answering, or any other NLP task, these techniques will serve you well.

> [!TIP]
> 💡 **Pro Tips for Success**:
> - Always start with a strong baseline using the `Trainer` API before implementing custom training loops
> - Use the 🤗 Hub to find pretrained models that are close to your task for better starting points
> - Monitor your training with proper evaluation metrics and don't forget to save checkpoints
> - Leverage the community - share your models and datasets to help others and get feedback on your work


---

<!-- Section 3.7 -->

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter Certificate

<CourseFloatingBanner chapter={3}
  classNames="absolute z-10 right-0 top-0"
/>

Congratulations on completing the course! You've learned how to fine-tune pretrained models, understand learning curves, and share your models with the community. Now it's time to take the quiz to test your knowledge and get your certificate.

To take the quiz, you will need to follow these steps:

1. Sign in to your Hugging Face account.
2. Answer the questions in the quiz.
3. Submit your answers.


## Multiple Choice Quiz

In this quiz, you will be asked to select the correct answer from a list of options. We'll test you on the fundamentals of supervised finetuning.

<iframe
	src="https://huggingface-course-unit-3-quiz.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

