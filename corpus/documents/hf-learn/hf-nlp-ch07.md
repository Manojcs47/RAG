# Hugging Face NLP Course — Chapter 7

Source: https://huggingface.co/learn/nlp-course/chapter7


---

<!-- Section 7.1 -->

<FrameworkSwitchCourse {fw} />

# Introduction[[introduction]]

<CourseFloatingBanner
    chapter={7}
    classNames="absolute z-10 right-0 top-0"
/>

In [Chapter 3](/course/chapter3), you saw how to fine-tune a model for text classification. In this chapter, we will tackle the following common language tasks that are essential for working with both traditional NLP models and modern LLMs:

- Token classification
- Masked language modeling (like BERT)
- Summarization
- Translation
- Causal language modeling pretraining (like GPT-2)
- Question answering

These fundamental tasks form the foundation of how Large Language Models (LLMs) work and understanding them is crucial for effectively working with today's most advanced language models.

{#if fw === 'pt'}

To do this, you'll need to leverage everything you learned about the `Trainer` API and the 🤗 Accelerate library in [Chapter 3](/course/chapter3), the 🤗 Datasets library in [Chapter 5](/course/chapter5), and the 🤗 Tokenizers library in [Chapter 6](/course/chapter6). We'll also upload our results to the Model Hub, like we did in [Chapter 4](/course/chapter4), so this is really the chapter where everything comes together!

Each section can be read independently and will show you how to train a model with the `Trainer` API or with your own training loop, using 🤗 Accelerate. Feel free to skip either part and focus on the one that interests you the most: the `Trainer` API is great for fine-tuning or training your model without worrying about what's going on behind the scenes, while the training loop with `Accelerate` will let you customize any part you want more easily.

{:else}

To do this, you'll need to leverage everything you learned about training models with the Keras API in [Chapter 3](/course/chapter3), the 🤗 Datasets library in [Chapter 5](/course/chapter5), and the 🤗 Tokenizers library in [Chapter 6](/course/chapter6). We'll also upload our results to the Model Hub, like we did in [Chapter 4](/course/chapter4), so this is really the chapter where everything comes together!

Each section can be read independently.

{/if}


> [!TIP]
> If you read the sections in sequence, you will notice that they have quite a bit of code and prose in common. The repetition is intentional, to allow you to dip in (or come back later) to any task that interests you and find a complete working example.


---

<!-- Section 7.2 -->

<FrameworkSwitchCourse {fw} />

# Token classification[[token-classification]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section2_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section2_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section2_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section2_tf.ipynb"},
]} />

{/if}

The first application we'll explore is token classification. This generic task encompasses any problem that can be formulated as "attributing a label to each token in a sentence," such as:

- **Named entity recognition (NER)**: Find the entities (such as persons, locations, or organizations) in a sentence. This can be formulated as attributing a label to each token by having one class per entity and one class for "no entity."
- **Part-of-speech tagging (POS)**: Mark each word in a sentence as corresponding to a particular part of speech (such as noun, verb, adjective, etc.).
- **Chunking**: Find the tokens that belong to the same entity. This task (which can be combined with POS or NER) can be formulated as attributing one label (usually `B-`) to any tokens that are at the beginning of a chunk, another label (usually `I-`) to tokens that are inside a chunk, and a third label (usually `O`) to tokens that don't belong to any chunk.

<Youtube id="wVHdVlPScxA"/>

Of course, there are many other types of token classification problem; those are just a few representative examples. In this section, we will fine-tune a model (BERT) on a NER task, which will then be able to compute predictions like this one:

<iframe src="https://course-demos-bert-finetuned-ner.hf.space" frameBorder="0" height="350" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

<a class="flex justify-center" href="/huggingface-course/bert-finetuned-ner">
<img class="block dark:hidden lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/model-eval-bert-finetuned-ner.png" alt="One-hot encoded labels for question answering."/>
<img class="hidden dark:block lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/model-eval-bert-finetuned-ner-dark.png" alt="One-hot encoded labels for question answering."/>
</a>

You can find the model we'll train and upload to the Hub and double-check its predictions [here](https://huggingface.co/huggingface-course/bert-finetuned-ner?text=My+name+is+Sylvain+and+I+work+at+Hugging+Face+in+Brooklyn).

## Preparing the data[[preparing-the-data]]

First things first, we need a dataset suitable for token classification. In this section we will use the [CoNLL-2003 dataset](https://huggingface.co/datasets/conll2003), which contains news stories from Reuters. 

> [!TIP]
> 💡 As long as your dataset consists of texts split into words with their corresponding labels, you will be able to adapt the data processing procedures described here to your own dataset. Refer back to [Chapter 5](/course/chapter5) if you need a refresher on how to load your own custom data in a `Dataset`.

### The CoNLL-2003 dataset[[the-conll-2003-dataset]]

To load the CoNLL-2003 dataset, we use the `load_dataset()` method from the 🤗 Datasets library:

```py
from datasets import load_dataset

raw_datasets = load_dataset("conll2003")
```

This will download and cache the dataset, like we saw in [Chapter 3](/course/chapter3) for the GLUE MRPC dataset. Inspecting this object shows us the columns present and the split between the training, validation, and test sets:

```py
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3453
    })
})
```

In particular, we can see the dataset contains labels for the three tasks we mentioned earlier: NER, POS, and chunking. A big difference from other datasets is that the input texts are not presented as sentences or documents, but lists of words (the last column is called `tokens`, but it contains words in the sense that these are pre-tokenized inputs that still need to go through the tokenizer for subword tokenization).

Let's have a look at the first element of the training set:

```py
raw_datasets["train"][0]["tokens"]
```

```python out
["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
```

Since we want to perform named entity recognition, we will look at the NER tags:

```py
raw_datasets["train"][0]["ner_tags"]
```

```python out
[3, 0, 7, 0, 0, 0, 7, 0, 0]
```

Those are the labels as integers ready for training, but they're not necessarily useful when we want to inspect the data. Like for text classification, we can access the correspondence between those integers and the label names by looking at the `features` attribute of our dataset:

```py
ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature
```

```python out
Sequence(
    feature=ClassLabel(
        num_classes=9,
        names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
        names_file=None,
        id=None,
    ),
    length=-1,
    id=None,
)
```

So this column contains elements that are sequences of `ClassLabel`s. The type of the elements of the sequence is in the `feature` attribute of this `ner_feature`, and we can access the list of names by looking at the `names` attribute of that `feature`:

```py
label_names = ner_feature.feature.names
label_names
```

```python out
["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
```

We already saw these labels when digging into the `token-classification` pipeline in [Chapter 6](/course/chapter6/3), but for a quick refresher: 

- `O` means the word doesn't correspond to any entity.
- `B-PER`/`I-PER` means the word corresponds to the beginning of/is inside a *person* entity.
- `B-ORG`/`I-ORG` means the word corresponds to the beginning of/is inside an *organization* entity.
- `B-LOC`/`I-LOC` means the word corresponds to the beginning of/is inside a *location* entity.
- `B-MISC`/`I-MISC` means the word corresponds to the beginning of/is inside a *miscellaneous* entity.

Now decoding the labels we saw earlier gives us this:

```python
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

```python out
"EU    rejects German call to boycott British lamb ."

"B-ORG O       B-MISC O    O  O       B-MISC  O    O"
```

And for an example mixing `B-` and `I-` labels, here's what the same code gives us on the element of the training set at index 4:

```python out
"Germany 's representative to the European Union 's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer ."

"B-LOC   O  O              O  O   B-ORG    I-ORG O  O          O         B-PER  I-PER     O    O  O         O         O      O   O         O    O         O     O    B-LOC   O     O   O          O      O   O       O"
```

As we can see, entities spanning two words, like "European Union" and "Werner Zwingmann," are attributed a `B-` label for the first word and an `I-` label for the second.

> [!TIP]
> ✏️ **Your turn!** Print the same two sentences with their POS or chunking labels.

### Processing the data[[processing-the-data]]

<Youtube id="iY2AZYdZAr0"/>

As usual, our texts need to be converted to token IDs before the model can make sense of them. As we saw in [Chapter 6](/course/chapter6/), a big difference in the case of token classification tasks is that we have pre-tokenized inputs. Fortunately, the tokenizer API can deal with that pretty easily; we just need to warn the `tokenizer` with a special flag.

To begin, let's create our `tokenizer` object. As we said before, we will be using a BERT pretrained model, so we'll start by downloading and caching the associated tokenizer:

```python
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

You can replace the `model_checkpoint` with any other model you prefer from the [Hub](https://huggingface.co/models), or with a local folder in which you've saved a pretrained model and a tokenizer. The only constraint is that the tokenizer needs to be backed by the 🤗 Tokenizers library, so there's a "fast" version available. You can see all the architectures that come with a fast version in [this big table](https://huggingface.co/transformers/#supported-frameworks), and to check that the `tokenizer` object you're using is indeed backed by 🤗 Tokenizers you can look at its `is_fast` attribute:

```py
tokenizer.is_fast
```

```python out
True
```

To tokenize a pre-tokenized input, we can use our `tokenizer` as usual and just add `is_split_into_words=True`:

```py
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()
```

```python out
["[CLS]", "EU", "rejects", "German", "call", "to", "boycott", "British", "la", "##mb", ".", "[SEP]"]
```

As we can see, the tokenizer added the special tokens used by the model (`[CLS]` at the beginning and `[SEP]` at the end) and left most of the words untouched. The word `lamb`, however, was tokenized into two subwords, `la` and `##mb`. This introduces a mismatch between our inputs and the labels: the list of labels has only 9 elements, whereas our input now has 12 tokens. Accounting for the special tokens is easy (we know they are at the beginning and the end), but we also need to make sure we align all the labels with the proper words.

Fortunately, because we're using a fast tokenizer we have access to the 🤗 Tokenizers superpowers, which means we can easily map each token to its corresponding word (as seen in [Chapter 6](/course/chapter6/3)):

```py
inputs.word_ids()
```

```python out
[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
```

With a tiny bit of work, we can then expand our label list to match the tokens. The first rule we'll apply is that special tokens get a label of `-100`. This is because by default `-100` is an index that is ignored in the loss function we will use (cross entropy). Then, each token gets the same label as the token that started the word it's inside, since they are part of the same entity. For tokens inside a word but not at the beginning, we replace the `B-` with `I-` (since the token does not begin the entity):

```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
```

Let's try it out on our first sentence:

```py
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))
```

```python out
[3, 0, 7, 0, 0, 0, 7, 0, 0]
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
```

As we can see, our function added the `-100` for the two special tokens at the beginning and the end, and a new `0` for our word that was split into two tokens.

> [!TIP]
> ✏️ **Your turn!** Some researchers prefer to attribute only one label per word, and assign `-100` to the other subtokens in a given word. This is to avoid long words that split into lots of subtokens contributing heavily to the loss. Change the previous function to align labels with input IDs by following this rule.

To preprocess our whole dataset, we need to tokenize all the inputs and apply `align_labels_with_tokens()` on all the labels. To take advantage of the speed of our fast tokenizer, it's best to tokenize lots of texts at the same time, so we'll write a function that processes a list of examples and use the `Dataset.map()` method with the option `batched=True`. The only thing that is different from our previous example is that the `word_ids()` function needs to get the index of the example we want the word IDs of when the inputs to the tokenizer are lists of texts (or in our case, list of lists of words), so we add that too:

```py
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
```

Note that we haven't padded our inputs yet; we'll do that later, when creating the batches with a data collator. 

We can now apply all that preprocessing in one go on the other splits of our dataset:

```py
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

We've done the hardest part! Now that the data has been preprocessed, the actual training will look a lot like what we did in [Chapter 3](/course/chapter3).

{#if fw === 'pt'}

## Fine-tuning the model with the `Trainer` API[[fine-tuning-the-model-with-the-trainer-api]]

The actual code using the `Trainer` will be the same as before; the only changes are the way the data is collated into a batch and the metric computation function.

{:else}

## Fine-tuning the model with Keras[[fine-tuning-the-model-with-keras]]

The actual code using Keras will be very similar to before; the only changes are the way the data is collated into a batch and the metric computation function.

{/if}


### Data collation[[data-collation]]

We can't just use a `DataCollatorWithPadding` like in [Chapter 3](/course/chapter3) because that only pads the inputs (input IDs, attention mask, and token type IDs). Here our labels should be padded the exact same way as the inputs so that they stay the same size, using `-100` as a value so that the corresponding predictions are ignored in the loss computation.

This is all done by a [`DataCollatorForTokenClassification`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorfortokenclassification). Like the `DataCollatorWithPadding`, it takes the `tokenizer` used to preprocess the inputs:

{#if fw === 'pt'}

```py
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

{:else}

```py
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
```

{/if}

To test this on a few samples, we can just call it on a list of examples from our tokenized training set:

```py
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]
```

```python out
tensor(
    [
        [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100],
        [-100, 1, 2, -100, -100, -100, -100, -100, -100, -100, -100, -100],
    ]
)
```

Let's compare this to the labels for the first and second elements in our dataset:

```py
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])
```

```python out
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
[-100, 1, 2, -100]
```

{#if fw === 'pt'}

As we can see, the second set of labels has been padded to the length of the first one using `-100`s.

{:else}

Our data collator is ready to go! Now let's use it to make a `tf.data.Dataset` with the `to_tf_dataset()` method. You can also use `model.prepare_tf_dataset()` to do this with a bit less boilerplate code - you'll see this in some of the other sections of this chapter.

```py
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
```


 Next stop: the model itself.

{/if}

{#if fw === 'tf'}

### Defining the model[[defining-the-model]]

Since we are working on a token classification problem, we will use the `TFAutoModelForTokenClassification` class. The main thing to remember when defining this model is to pass along some information on the number of labels we have. The easiest way to do this is to pass that number with the `num_labels` argument, but if we want a nice inference widget working like the one we saw at the beginning of this section, it's better to set the correct label correspondences instead.

They should be set by two dictionaries, `id2label` and `label2id`, which contain the mapping from ID to label and vice versa:

```py
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
```

Now we can just pass them to the `TFAutoModelForTokenClassification.from_pretrained()` method, and they will be set in the model's configuration, then properly saved and uploaded to the Hub:

```py
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

Like when we defined our `TFAutoModelForSequenceClassification` in [Chapter 3](/course/chapter3), creating the model issues a warning that some weights were not used (the ones from the pretraining head) and some other weights are randomly initialized (the ones from the new token classification head), and that this model should be trained. We will do that in a minute, but first let's double-check that our model has the right number of labels:

```python
model.config.num_labels
```

```python out
9
```

> [!WARNING]
> ⚠️ If you have a model with the wrong number of labels, you will get an obscure error when calling `model.fit()` later. This can be annoying to debug, so make sure you do this check to confirm you have the expected number of labels.

### Fine-tuning the model[[fine-tuning-the-model]]

We are now ready to train our model! We have just a little more housekeeping to do first, though: we should log in to Hugging Face and define our training hyperparameters. If you're working in a notebook, there's a convenience function to help you with this:

```python
from huggingface_hub import notebook_login

notebook_login()
```

This will display a widget where you can enter your Hugging Face login credentials.

If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

After logging in, we can prepare everything we need to compile our model. 🤗 Transformers provides a convenient `create_optimizer()` function that will give you an `AdamW` optimizer with appropriate settings for the weight decay and learning rate decay, both of which will improve your model's performance compared to the built-in `Adam` optimizer: 

```python
from transformers import create_optimizer
import tensorflow as tf

# Train in mixed-precision float16
# Comment this line out if you're using a GPU that will not benefit from this
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)
```

Note also that we don't supply a `loss` argument to `compile()`. This is because the models can actually compute loss internally -- if you compile without a loss and supply your labels in the input dictionary (as we do in our datasets), then the model will train using that internal loss, which will be appropriate for the task and model type you have chosen.

Next, we define a `PushToHubCallback` to upload our model to the Hub during training, and fit the model with that callback:

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="bert-finetuned-ner", tokenizer=tokenizer)

model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)
```

You can specify the full name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, when we pushed the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/bert-finetuned-ner"`. By default, the repository used will be in your namespace and named after the output directory you set, for example `"cool_huggingface_user/bert-finetuned-ner"`.

> [!TIP]
> 💡 If the output directory you are using already exists, it needs to be a local clone of the repository you want to push to. If it isn't, you'll get an error when calling `model.fit()` and will need to set a new name.

Note that while the training happens, each time the model is saved (here, every epoch) it is uploaded to the Hub in the background. This way, you will be able to to resume your training on another machine if necessary.

At this stage, you can use the inference widget on the Model Hub to test your model and share it with your friends. You have successfully fine-tuned a model on a token classification task -- congratulations! But how good is our model, really? We should evaluate some metrics to find out.

{/if}


### Metrics[[metrics]]

{#if fw === 'pt'}

To have the `Trainer` compute a metric every epoch, we will need to define a `compute_metrics()` function that takes the arrays of predictions and labels, and returns a dictionary with the metric names and values. 

The traditional framework used to evaluate token classification prediction is [*seqeval*](https://github.com/chakki-works/seqeval). To use this metric, we first need to install the *seqeval* library:

```py
!pip install seqeval
```

We can then load it via the `evaluate.load()` function like we did in [Chapter 3](/course/chapter3):

{:else}

The traditional framework used to evaluate token classification prediction is [*seqeval*](https://github.com/chakki-works/seqeval). To use this metric, we first need to install the *seqeval* library:

```py
!pip install seqeval
```

We can then load it via the `evaluate.load()` function like we did in [Chapter 3](/course/chapter3):

{/if}

```py
import evaluate

metric = evaluate.load("seqeval")
```

This metric does not behave like the standard accuracy: it will actually take the lists of labels as strings, not integers, so we will need to fully decode the predictions and labels before passing them to the metric. Let's see how it works. First, we'll get the labels for our first training example:

```py
labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
labels
```

```python out
["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"]
```

We can then create fake predictions for those by just changing the value at index 2:

```py
predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])
```

Note that the metric takes a list of predictions (not just one) and a list of labels. Here's the output:

```python out
{
    "MISC": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "number": 2},
    "ORG": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "number": 1},
    "overall_precision": 1.0,
    "overall_recall": 0.67,
    "overall_f1": 0.8,
    "overall_accuracy": 0.89,
}
```

{#if fw === 'pt'}

This is sending back a lot of information! We get the precision, recall, and F1 score for each separate entity, as well as overall. For our metric computation we will only keep the overall score, but feel free to tweak the `compute_metrics()` function to return all the metrics you would like reported.

This `compute_metrics()` function first takes the argmax of the logits to convert them to predictions (as usual, the logits and the probabilities are in the same order, so we don't need to apply the softmax). Then we have to convert both labels and predictions from integers to strings. We remove all the values where the label is `-100`, then pass the results to the `metric.compute()` method:

```py
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

Now that this is done, we are almost ready to define our `Trainer`. We just need a `model` to fine-tune!

{:else}

This is sending back a lot of information! We get the precision, recall, and F1 score for each separate entity, as well as overall. Now let's see what happens if we try using our actual model predictions to compute some real scores.

TensorFlow doesn't like concatenating our predictions together, because they have variable sequence lengths. This means we can't just use `model.predict()` -- but that's not going to stop us. We'll get some predictions a batch at a time and concatenate them into one big long list as we go, dropping the `-100` tokens that indicate masking/padding, then compute metrics on the list at the end:

```py
import numpy as np

all_predictions = []
all_labels = []
for batch in tf_eval_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(label_names[predicted_idx])
            all_labels.append(label_names[label_idx])
metric.compute(predictions=[all_predictions], references=[all_labels])
```


```python out
{
    "LOC": {"precision": 0.91, "recall": 0.92, "f1": 0.91, "number": 1668},
    "MISC": {"precision": 0.70, "recall": 0.79, "f1": 0.74, "number": 702},
    "ORG": {"precision": 0.85, "recall": 0.90, "f1": 0.88, "number": 1661},
    "PER": {"precision": 0.95, "recall": 0.95, "f1": 0.95, "number": 1617},
    "overall_precision": 0.87,
    "overall_recall": 0.91,
    "overall_f1": 0.89,
    "overall_accuracy": 0.97,
}
```

How did your model do, compared to ours? If you got similar numbers, your training was a success!

{/if}

{#if fw === 'pt'}

### Defining the model[[defining-the-model]]

Since we are working on a token classification problem, we will use the `AutoModelForTokenClassification` class. The main thing to remember when defining this model is to pass along some information on the number of labels we have. The easiest way to do this is to pass that number with the `num_labels` argument, but if we want a nice inference widget working like the one we saw at the beginning of this section, it's better to set the correct label correspondences instead.

They should be set by two dictionaries, `id2label` and `label2id`, which contain the mappings from ID to label and vice versa:

```py
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
```

Now we can just pass them to the `AutoModelForTokenClassification.from_pretrained()` method, and they will be set in the model's configuration and then properly saved and uploaded to the Hub:

```py
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

Like when we defined our `AutoModelForSequenceClassification` in [Chapter 3](/course/chapter3), creating the model issues a warning that some weights were not used (the ones from the pretraining head) and some other weights are randomly initialized (the ones from the new token classification head), and that this model should be trained. We will do that in a minute, but first let's double-check that our model has the right number of labels:

```python
model.config.num_labels
```

```python out
9
```

> [!WARNING]
> ⚠️ If you have a model with the wrong number of labels, you will get an obscure error when calling the `Trainer.train()` method later on (something like "CUDA error: device-side assert triggered"). This is the number one cause of bugs reported by users for such errors, so make sure you do this check to confirm that you have the expected number of labels.

### Fine-tuning the model[[fine-tuning-the-model]]

We are now ready to train our model! We just need to do two last things before we define our `Trainer`: log in to Hugging Face and define our training arguments. If you're working in a notebook, there's a convenience function to help you with this:

```python
from huggingface_hub import notebook_login

notebook_login()
```

This will display a widget where you can enter your Hugging Face login credentials.

If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

Once this is done, we can define our `TrainingArguments`:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)
```

You've seen most of those before: we set some hyperparameters (like the learning rate, the number of epochs to train for, and the weight decay), and we specify `push_to_hub=True` to indicate that we want to save the model and evaluate it at the end of every epoch, and that we want to upload our results to the Model Hub. Note that you can specify the name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, when we pushed the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/bert-finetuned-ner"` to `TrainingArguments`. By default, the repository used will be in your namespace and named after the output directory you set, so in our case it will be `"sgugger/bert-finetuned-ner"`.

> [!TIP]
> 💡 If the output directory you are using already exists, it needs to be a local clone of the repository you want to push to. If it isn't, you'll get an error when defining your `Trainer` and will need to set a new name.

Finally, we just pass everything to the `Trainer` and launch the training:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train()
```

Note that while the training happens, each time the model is saved (here, every epoch) it is uploaded to the Hub in the background. This way, you will be able to to resume your training on another machine if necessary.

Once the training is complete, we use the `push_to_hub()` method to make sure we upload the most recent version of the model:

```py
trainer.push_to_hub(commit_message="Training complete")
```

This command returns the URL of the commit it just did, if you want to inspect it:

```python out
"https://huggingface.co/sgugger/bert-finetuned-ner/commit/26ab21e5b1568f9afeccdaed2d8715f571d786ed"
```

The `Trainer` also drafts a model card with all the evaluation results and uploads it. At this stage, you can use the inference widget on the Model Hub to test your model and share it with your friends. You have successfully fine-tuned a model on a token classification task -- congratulations!

If you want to dive a bit more deeply into the training loop, we will now show you how to do the same thing using 🤗 Accelerate.

## A custom training loop[[a-custom-training-loop]]

Let's now take a look at the full training loop, so you can easily customize the parts you need. It will look a lot like what we did in [Chapter 3](/course/chapter3/4), with a few changes for the evaluation.

### Preparing everything for training[[preparing-everything-for-training]]

First we need to build the `DataLoader`s from our datasets. We'll reuse our `data_collator` as a `collate_fn` and shuffle the training set, but not the validation set:

```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
```

Next we reinstantiate our model, to make sure we're not continuing the fine-tuning from before but starting from the BERT pretrained model again:

```py
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

Then we will need an optimizer. We'll use the classic `AdamW`, which is like `Adam`, but with a fix in the way weight decay is applied:

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Once we have all those objects, we can send them to the `accelerator.prepare()` method:

```py
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

> [!TIP]
> 🚨 If you're training on a TPU, you'll need to move all the code starting from the cell above into a dedicated training function. See [Chapter 3](/course/chapter3) for more details.

Now that we have sent our `train_dataloader` to `accelerator.prepare()`, we can use its length to compute the number of training steps. Remember that we should always do this after preparing the dataloader, as that method will change its length. We use a classic linear schedule from the learning rate to 0:

```py
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Lastly, to push our model to the Hub, we will need to create a `Repository` object in a working folder. First log in to Hugging Face, if you're not logged in already. We'll determine the repository name from the model ID we want to give our model (feel free to replace the `repo_name` with your own choice; it just needs to contain your username, which is what the function `get_full_repo_name()` does):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "bert-finetuned-ner-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
"sgugger/bert-finetuned-ner-accelerate"
```

Then we can clone that repository in a local folder. If it already exists, this local folder should be an existing clone of the repository we are working with:

```py
output_dir = "bert-finetuned-ner-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

We can now upload anything we save in `output_dir` by calling the `repo.push_to_hub()` method. This will help us upload the intermediate models at the end of each epoch.

### Training loop[[training-loop]]

We are now ready to write the full training loop. To simplify its evaluation part, we define this `postprocess()` function that takes predictions and labels and converts them to lists of strings, like our `metric` object expects:

```py
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
```

Then we can write the training loop. After defining a progress bar to follow how training goes, the loop has three parts:

- The training in itself, which is the classic iteration over the `train_dataloader`, forward pass through the model, then backward pass and optimizer step.
- The evaluation, in which there is a novelty after getting the outputs of our model on a batch: since two processes may have padded the inputs and labels to different shapes, we need to use `accelerator.pad_across_processes()` to make the predictions and labels the same shape before calling the `gather()` method. If we don't do this, the evaluation will either error out or hang forever. Then we send the results to `metric.add_batch()` and call `metric.compute()` once the evaluation loop is over.
- Saving and uploading, where we first save the model and the tokenizer, then call `repo.push_to_hub()`. Notice that we use the argument `blocking=False` to tell the 🤗 Hub library to push in an asynchronous process. This way, training continues normally and this (long) instruction is executed in the background.

Here's the complete code for the training loop:

```py
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {key: results[f"overall_{key}"] for key in ["precision", "recall", "f1", "accuracy"]},
    )

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
```

In case this is the first time you're seeing a model saved with 🤗 Accelerate, let's take a moment to inspect the three lines of code that go with it:

```py
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
```

The first line is self-explanatory: it tells all the processes to wait until everyone is at that stage before continuing. This is to make sure we have the same model in every process before saving. Then we grab the `unwrapped_model`, which is the base model we defined. The `accelerator.prepare()` method changes the model to work in distributed training, so it won't have the `save_pretrained()` method anymore; the `accelerator.unwrap_model()` method undoes that step. Lastly, we call `save_pretrained()` but tell that method to use `accelerator.save()` instead of `torch.save()`. 

Once this is done, you should have a model that produces results pretty similar to the one trained with the `Trainer`. You can check the model we trained using this code at [*huggingface-course/bert-finetuned-ner-accelerate*](https://huggingface.co/huggingface-course/bert-finetuned-ner-accelerate). And if you want to test out any tweaks to the training loop, you can directly implement them by editing the code shown above!

{/if}

## Using the fine-tuned model[[using-the-fine-tuned-model]]

We've already shown you how you can use the model we fine-tuned on the Model Hub with the inference widget. To use it locally in a `pipeline`, you just have to specify the proper model identifier:

```py
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[
    {"entity_group": "PER", "score": 0.9988506, "word": "Sylvain", "start": 11, "end": 18},
    {"entity_group": "ORG", "score": 0.9647625, "word": "Hugging Face", "start": 33, "end": 45},
    {"entity_group": "LOC", "score": 0.9986118, "word": "Brooklyn", "start": 49, "end": 57},
]
```

Great! Our model is working as well as the default one for this pipeline!


---

<!-- Section 7.3 -->

<FrameworkSwitchCourse {fw} />

# Fine-tuning a masked language model[[fine-tuning-a-masked-language-model]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_tf.ipynb"},
]} />

{/if}

For many NLP applications involving Transformer models, you can simply take a pretrained model from the Hugging Face Hub and fine-tune it directly on your data for the task at hand. Provided that the corpus used for pretraining is not too different from the corpus used for fine-tuning, transfer learning will usually produce good results. 

However, there are a few cases where you'll want to first fine-tune the language models on your data, before training a task-specific head. For example, if your dataset contains legal contracts or scientific articles, a vanilla Transformer model like BERT will typically treat the domain-specific words in your corpus as rare tokens, and the resulting performance may be less than satisfactory. By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once!

This process of fine-tuning a pretrained language model on in-domain data is usually called _domain adaptation_. It was popularized in 2018 by [ULMFiT](https://arxiv.org/abs/1801.06146), which was one of the first neural architectures (based on LSTMs) to make transfer learning really work for NLP. An example of domain adaptation with ULMFiT is shown in the image below; in this section we'll do something similar, but with a Transformer instead of an LSTM!

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/ulmfit.svg" alt="ULMFiT."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/ulmfit-dark.svg" alt="ULMFiT."/>
</div>

By the end of this section you'll have a [masked language model](https://huggingface.co/huggingface-course/distilbert-base-uncased-finetuned-imdb?text=This+is+a+great+%5BMASK%5D.) on the Hub that can autocomplete sentences as shown below:

<iframe src="https://course-demos-distilbert-base-uncased-finetuned-imdb.hf.space" frameBorder="0" height="300" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

Let's dive in!

<Youtube id="mqElG5QJWUg"/>

> [!TIP]
> 🙋 If the terms "masked language modeling" and "pretrained model" sound unfamiliar to you, go check out [Chapter 1](/course/chapter1), where we explain all these core concepts, complete with videos!

## Picking a pretrained model for masked language modeling[[picking-a-pretrained-model-for-masked-language-modeling]]

To get started, let's pick a suitable pretrained model for masked language modeling. As shown in the following screenshot, you can find a list of candidates by applying the "Fill-Mask" filter on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads):

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/mlm-models.png" alt="Hub models." width="80%"/>
</div>

Although the BERT and RoBERTa family of models are the most downloaded, we'll use a model called [DistilBERT](https://huggingface.co/distilbert-base-uncased) 
that can be trained much faster with little to no loss in downstream performance. This model was trained using a special technique called [_knowledge distillation_](https://en.wikipedia.org/wiki/Knowledge_distillation), where a large "teacher model" like BERT is used to guide the training of a "student model" that has far fewer parameters. An explanation of the details of knowledge distillation would take us too far afield in this section, but if you're interested you can read all about it in [_Natural Language Processing with Transformers_](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) (colloquially known as the Transformers textbook).

{#if fw === 'pt'}

Let's go ahead and download DistilBERT using the `AutoModelForMaskedLM` class:

```python
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

We can see how many parameters this model has by calling the `num_parameters()` method:

```python
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")
```

```python out
">>> DistilBERT number of parameters: 67M"

">>> BERT number of parameters: 110M"
```

{:else}

Let's go ahead and download DistilBERT using the `AutoModelForMaskedLM` class:

```python
from transformers import TFAutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

We can see how many parameters this model has by calling the `summary()` method:

```python
model.summary()
```

```python out
Model: "tf_distil_bert_for_masked_lm"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
distilbert (TFDistilBertMain multiple                  66362880  
_________________________________________________________________
vocab_transform (Dense)      multiple                  590592    
_________________________________________________________________
vocab_layer_norm (LayerNorma multiple                  1536      
_________________________________________________________________
vocab_projector (TFDistilBer multiple                  23866170  
=================================================================
Total params: 66,985,530
Trainable params: 66,985,530
Non-trainable params: 0
_________________________________________________________________
```

{/if}

With around 67 million parameters, DistilBERT is approximately two times smaller than the BERT base model, which roughly translates into a two-fold speedup in training -- nice! Let's now see what kinds of tokens this model predicts are the most likely completions of a small sample of text:

```python
text = "This is a great [MASK]."
```

As humans, we can imagine many possibilities for the `[MASK]` token, such as "day", "ride", or "painting". For pretrained models, the predictions depend on the corpus the model was trained on, since it learns to pick up the statistical patterns present in the data. Like BERT, DistilBERT was pretrained on the [English Wikipedia](https://huggingface.co/datasets/wikipedia) and [BookCorpus](https://huggingface.co/datasets/bookcorpus) datasets, so we expect the predictions for `[MASK]` to reflect these domains. To predict the mask we need DistilBERT's tokenizer to produce the inputs for the model, so let's download that from the Hub as well:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

With a tokenizer and a model, we can now pass our text example to the model, extract the logits, and print out the top 5 candidates:

{#if fw === 'pt'}

```python
import torch

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
```

{:else}

```python
import numpy as np
import tensorflow as tf

inputs = tokenizer(text, return_tensors="np")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
# We negate the array before argsort to get the largest, not the smallest, logits
top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()

for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")
```

{/if}

```python out
">>> This is a great deal."

">>> This is a great success."
">>> This is a great adventure."
">>> This is a great idea."
">>> This is a great feat."
```

We can see from the outputs that the model's predictions refer to everyday terms, which is perhaps not surprising given the foundation of English Wikipedia. Let's see how we can change this domain to something a bit more niche -- highly polarized movie reviews!


## The dataset[[the-dataset]]

To showcase domain adaptation, we'll use the famous [Large Movie Review Dataset](https://huggingface.co/datasets/imdb) (or IMDb for short), which is a corpus of movie reviews that is often used to benchmark sentiment analysis models. By fine-tuning DistilBERT on this corpus, we expect the language model will adapt its vocabulary from the factual data of Wikipedia that it was pretrained on to the more subjective elements of movie reviews. We can get the data from the Hugging Face Hub with the `load_dataset()` function from 🤗 Datasets:

```python
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
imdb_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
```

We can see that the `train` and `test` splits each consist of 25,000 reviews, while there is an unlabeled split called `unsupervised` that contains 50,000 reviews. Let's take a look at a few samples to get an idea of what kind of text we're dealing with. As we've done in previous chapters of the course, we'll chain the `Dataset.shuffle()` and `Dataset.select()` functions to create a random sample:

```python
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")
```

```python out
'>>> Review: This is your typical Priyadarshan movie--a bunch of loony characters out on some silly mission. His signature climax has the entire cast of the film coming together and fighting each other in some crazy moshpit over hidden money. Whether it is a winning lottery ticket in Malamaal Weekly, black money in Hera Pheri, "kodokoo" in Phir Hera Pheri, etc., etc., the director is becoming ridiculously predictable. Don\'t get me wrong; as clichéd and preposterous his movies may be, I usually end up enjoying the comedy. However, in most his previous movies there has actually been some good humor, (Hungama and Hera Pheri being noteworthy ones). Now, the hilarity of his films is fading as he is using the same formula over and over again.<br /><br />Songs are good. Tanushree Datta looks awesome. Rajpal Yadav is irritating, and Tusshar is not a whole lot better. Kunal Khemu is OK, and Sharman Joshi is the best.'

">>> Label: 0"

'>>> Review: Okay, the story makes no sense, the characters lack any dimensionally, the best dialogue is ad-libs about the low quality of movie, the cinematography is dismal, and only editing saves a bit of the muddle, but Sam" Peckinpah directed the film. Somehow, his direction is not enough. For those who appreciate Peckinpah and his great work, this movie is a disappointment. Even a great cast cannot redeem the time the viewer wastes with this minimal effort.<br /><br />The proper response to the movie is the contempt that the director San Peckinpah, James Caan, Robert Duvall, Burt Young, Bo Hopkins, Arthur Hill, and even Gig Young bring to their work. Watch the great Peckinpah films. Skip this mess.'
">>> Label: 0"

">>> Review: I saw this movie at the theaters when I was about 6 or 7 years old. I loved it then, and have recently come to own a VHS version. <br /><br />My 4 and 6 year old children love this movie and have been asking again and again to watch it. <br /><br />I have enjoyed watching it again too. Though I have to admit it is not as good on a little TV.<br /><br />I do not have older children so I do not know what they would think of it. <br /><br />The songs are very cute. My daughter keeps singing them over and over.<br /><br />Hope this helps."
">>> Label: 1"
```

Yep, these are certainly movie reviews, and if you're old enough you may even understand the comment in the last review about owning a VHS version 😜! Although we won't need the labels for language modeling, we can already see that a `0` denotes a negative review, while a `1` corresponds to a positive one.

> [!TIP]
> ✏️ **Try it out!** Create a random sample of the `unsupervised` split and verify that the labels are neither `0` nor `1`. While you're at it, you could also check that the labels in the `train` and `test` splits are indeed `0` or `1` -- this is a useful sanity check that every NLP practitioner should perform at the start of a new project!

Now that we've had a quick look at the data, let's dive into preparing it for masked language modeling. As we'll see, there are some additional steps that one needs to take compared to the sequence classification tasks we saw in [Chapter 3](/course/chapter3). Let's go!

## Preprocessing the data[[preprocessing-the-data]]

<Youtube id="8PmhEIXhBvI"/>

For both auto-regressive and masked language modeling, a common preprocessing step is to concatenate all the examples and then split the whole corpus into chunks of equal size. This is quite different from our usual approach, where we simply tokenize individual examples. Why concatenate everything together? The reason is that individual examples might get truncated if they're too long, and that would result in losing information that might be useful for the language modeling task!

So to get started, we'll first tokenize our corpus as usual, but _without_ setting the `truncation=True` option in our tokenizer. We'll also grab the word IDs if they are available ((which they will be if we're using a fast tokenizer, as described in [Chapter 6](/course/chapter6/3)), as we will need them later on to do whole word masking. We'll wrap this in a simple function, and while we're at it we'll remove the `text` and `label` columns since we don't need them any longer:

```python
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 50000
    })
})
```

Since DistilBERT is a BERT-like model, we can see that the encoded texts consist of the `input_ids` and `attention_mask` that we've seen in other chapters, as well as the `word_ids` we added. 

Now that we've tokenized our movie reviews, the next step is to group them all together and split the result into chunks. But how big should these chunks be? This will ultimately be determined by the amount of GPU memory that you have available, but a good starting point is to see what the model's maximum context size is. This can be inferred by inspecting the `model_max_length` attribute of the tokenizer:

```python
tokenizer.model_max_length
```

```python out
512
```

This value is derived from the *tokenizer_config.json* file associated with a checkpoint; in this case we can see that the context size is 512 tokens, just like with BERT.

> [!TIP]
> ✏️ **Try it out!** Some Transformer models, like [BigBird](https://huggingface.co/google/bigbird-roberta-base) and [Longformer](hf.co/allenai/longformer-base-4096), have a much longer context length than BERT and other early Transformer models. Instantiate the tokenizer for one of these checkpoints and verify that the `model_max_length` agrees with what's quoted on its model card.

So, in order to run our experiments on GPUs like those found on Google Colab, we'll pick something a bit smaller that can fit in memory:

```python
chunk_size = 128
```

> [!WARNING]
> Note that using a small chunk size can be detrimental in real-world scenarios, so you should use a size that corresponds to the use case you will apply your model to.

Now comes the fun part. To show how the concatenation works, let's take a few reviews from our tokenized training set and print out the number of tokens per review:

```python
# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
```

```python out
">>> Review 0 length: 200"

">>> Review 1 length: 559"
">>> Review 2 length: 192"
```

We can then concatenate all these examples with a simple dictionary comprehension, as follows:

```python
concatenated_examples = {k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
```

```python out
">>> Concatenated reviews length: 951"
```

Great, the total length checks out -- so now let's split the concatenated reviews into chunks of the size given by `chunk_size`. To do so, we iterate over the features in `concatenated_examples` and use a list comprehension to create slices of each feature. The result is a dictionary of chunks for each feature:

```python
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")
```

```python out
">>> Chunk length: 128"

">>> Chunk length: 128"
">>> Chunk length: 128"
">>> Chunk length: 128"
">>> Chunk length: 128"
">>> Chunk length: 128"
">>> Chunk length: 128"
">>> Chunk length: 55"
```

As you can see in this example, the last chunk will generally be smaller than the maximum chunk size. There are two main strategies for dealing with this:

* Drop the last chunk if it's smaller than `chunk_size`.
* Pad the last chunk until its length equals `chunk_size`.

We'll take the first approach here, so let's wrap all of the above logic in a single function that we can apply to our tokenized datasets:

```python
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
```

Note that in the last step of `group_texts()` we create a new `labels` column which is a copy of the `input_ids` one. As we'll see shortly, that's because in masked language modeling the objective is to predict randomly masked tokens in the input batch, and by creating a `labels` column we provide the ground truth for our language model to learn from. 

Let's now apply `group_texts()` to our tokenized datasets using our trusty `Dataset.map()` function:

```python
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 61289
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 59905
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 122963
    })
})
```

You can see that grouping and then chunking the texts has produced many more examples than our original 25,000 for the `train` and `test` splits. That's because we now have examples involving _contiguous tokens_ that span across multiple examples from the original corpus. You can see this explicitly by looking for the special `[SEP]` and `[CLS]` tokens in one of the chunks:

```python
tokenizer.decode(lm_datasets["train"][1]["input_ids"])
```

```python out
".... at.......... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```

In this example you can see two overlapping movie reviews, one about a high school movie and the other about homelessness. Let's also check out what the labels look like for masked language modeling:

```python out
tokenizer.decode(lm_datasets["train"][1]["labels"])
```

```python out
".... at.......... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```

As expected from our `group_texts()` function above, this looks identical to the decoded `input_ids` -- but then how can our model possibly learn anything? We're missing a key step: inserting `[MASK]` tokens at random positions in the inputs! Let's see how we can do this on the fly during fine-tuning using a special data collator.

## Fine-tuning DistilBERT with the `Trainer` API[[fine-tuning-distilbert-with-the-trainer-api]]

Fine-tuning a masked language model is almost identical to fine-tuning a sequence classification model, like we did in [Chapter 3](/course/chapter3). The only difference is that we need a special data collator that can randomly mask some of the tokens in each batch of texts. Fortunately, 🤗 Transformers comes prepared with a dedicated `DataCollatorForLanguageModeling` for just this task. We just have to pass it the tokenizer and an `mlm_probability` argument that specifies what fraction of the tokens to mask. We'll pick 15%, which is the amount used for BERT and a common choice in the literature:

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

To see how the random masking works, let's feed a few examples to the data collator. Since it expects a list of `dict`s, where each `dict` represents a single chunk of contiguous text, we first iterate over the dataset before feeding the batch to the collator. We remove the `"word_ids"` key for this data collator as it does not expect it:

```python
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

```python output
'>>> [CLS] bromwell [MASK] is a cartoon comedy. it ran at the same [MASK] as some other [MASK] about school life, [MASK] as " teachers ". [MASK] [MASK] [MASK] in the teaching [MASK] lead [MASK] to believe that bromwell high\'[MASK] satire is much closer to reality than is " teachers ". the scramble [MASK] [MASK] financially, the [MASK]ful students whogn [MASK] right through [MASK] pathetic teachers\'pomp, the pettiness of the whole situation, distinction remind me of the schools i knew and their students. when i saw [MASK] episode in [MASK] a student repeatedly tried to burn down the school, [MASK] immediately recalled. [MASK]...'

">>> .... at.. [MASK]... [MASK]... high. a classic line plucked inspector : i'[MASK] here to [MASK] one of your [MASK]. student : welcome to bromwell [MASK]. i expect that many adults of my age think that [MASK]mwell [MASK] is [MASK] fetched. what a pity that it isn't! [SEP] [CLS] [MASK]ness ( or [MASK]lessness as george 宇in stated )公 been an issue for years but never [MASK] plan to help those on the street that were once considered human [MASK] did everything from going to school, [MASK], [MASK] vote for the matter. most people think [MASK] the homeless"
```

Nice, it worked! We can see that the `[MASK]` token has been randomly inserted at various locations in our text. These will be the tokens which our model will have to predict during training -- and the beauty of the data collator is that it will randomize the `[MASK]` insertion with every batch! 

> [!TIP]
> ✏️ **Try it out!** Run the code snippet above several times to see the random masking happen in front of your very eyes! Also replace the `tokenizer.decode()` method with `tokenizer.convert_ids_to_tokens()` to see that sometimes a single token from a given word is masked, and not the others.

{#if fw === 'pt'}

One side effect of random masking is that our evaluation metrics will not be deterministic when using the `Trainer`, since we use the same data collator for the training and test sets. We'll see later, when we look at fine-tuning with 🤗 Accelerate, how we can use the flexibility of a custom evaluation loop to freeze the randomness.

{/if}

When training models for masked language modeling, one technique that can be used is to mask whole words together, not just individual tokens. This approach is called _whole word masking_. If we want to use whole word masking, we will need to build a data collator ourselves. A data collator is just a function that takes a list of samples and converts them into a batch, so let's do this now! We'll use the word IDs computed earlier to make a map between word indices and the corresponding tokens, then randomly decide which words to mask and apply that mask on the inputs. Note that the labels are all `-100` except for the ones corresponding to mask words.

{#if fw === 'pt'}

```py
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)
```

{:else}

```py
import collections
import numpy as np

from transformers.data.data_collator import tf_default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return tf_default_data_collator(features)
```

{/if}

Next, we can try it on the same samples as before:

```py
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

```python out
'>>> [CLS] bromwell high is a cartoon comedy [MASK] it ran at the same time as some other programs about school life, such as " teachers ". my 35 years in the teaching profession lead me to believe that bromwell high\'s satire is much closer to reality than is " teachers ". the scramble to survive financially, the insightful students who can see right through their pathetic teachers\'pomp, the pettiness of the whole situation, all remind me of the schools i knew and their students. when i saw the episode in which a student repeatedly tried to burn down the school, i immediately recalled.....'

">>> .... [MASK] [MASK] [MASK] [MASK]....... high. a classic line : inspector : i'm here to sack one of your teachers. student : welcome to bromwell high. i expect that many adults of my age think that bromwell high is far fetched. what a pity that it isn't! [SEP] [CLS] homelessness ( or houselessness as george carlin stated ) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. most people think of the homeless"
```

> [!TIP]
> ✏️ **Try it out!** Run the code snippet above several times to see the random masking happen in front of your very eyes! Also replace the `tokenizer.decode()` method with `tokenizer.convert_ids_to_tokens()` to see that the tokens from a given word are always masked together.

Now that we have two data collators, the rest of the fine-tuning steps are standard. Training can take a while on Google Colab if you're not lucky enough to score a mythical P100 GPU 😭, so we'll first downsample the size of the training set to a few thousand examples. Don't worry, we'll still get a pretty decent language model! A quick way to downsample a dataset in 🤗 Datasets is via the `Dataset.train_test_split()` function that we saw in [Chapter 5](/course/chapter5):

```python
train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
        num_rows: 1000
    })
})
```

This has automatically created new `train` and `test` splits, with the training set size set to 10,000 examples and the validation set to 10% of that -- feel free to increase this if you have a beefy GPU! The next thing we need to do is log in to the Hugging Face Hub. If you're running this code in a notebook, you can do so with the following utility function:

```python
from huggingface_hub import notebook_login

notebook_login()
```

which will display a widget where you can enter your credentials. Alternatively, you can run: 

```
huggingface-cli login
```

in your favorite terminal and log in there. 

{#if fw === 'tf'}

Once we're logged in, we can create our `tf.data` datasets. To do so, we'll use the `prepare_tf_dataset()` method, which uses our model to automatically infer which columns should go into the dataset. If you want to control exactly which columns to use, you can use the `Dataset.to_tf_dataset()` method instead. To keep things simple, we'll just use the standard data collator here, but you can also try the whole word masking collator and compare the results as an exercise:

```python
tf_train_dataset = model.prepare_tf_dataset(
    downsampled_dataset["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)

tf_eval_dataset = model.prepare_tf_dataset(
    downsampled_dataset["test"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)
```

Next, we set up our training hyperparameters and compile our model. We use the `create_optimizer()` function from the 🤗 Transformers library, which gives us an `AdamW` optimizer with linear learning rate decay. We also use the model's built-in loss, which is the default when no loss is specified as an argument to `compile()`, and we set the training precision to `"mixed_float16"`. Note that if you're using a Colab GPU or other GPU that does not have accelerated float16 support, you should probably comment out that line.

In addition, we set up a `PushToHubCallback` that will save the model to the Hub after each epoch. You can specify the name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, to push the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/distilbert-finetuned-imdb"`. By default, the repository used will be in your namespace and named after the output directory you set, so in our case it will be `"lewtun/distilbert-finetuned-imdb"`.

```python
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model_name = model_checkpoint.split("/")[-1]
callback = PushToHubCallback(output_dir=f"{model_name}-finetuned-imdb", tokenizer=tokenizer)
```

We're now ready to run `model.fit()` -- but before doing so let's briefly look at _perplexity_, which is a common metric to evaluate the performance of language models.

{:else}

Once we're logged in, we can specify the arguments for the `Trainer`:

```python
from transformers import TrainingArguments

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)
```

Here we tweaked a few of the default options, including `logging_steps` to ensure we track the training loss with each epoch. We've also used `fp16=True` to enable mixed-precision training, which gives us another boost in speed. By default, the `Trainer` will remove any columns that are not part of the model's `forward()` method. This means that if you're using the whole word masking collator, you'll also need to set `remove_unused_columns=False` to ensure we don't lose the `word_ids` column during training.

Note that you can specify the name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, when we pushed the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/distilbert-finetuned-imdb"` to `TrainingArguments`. By default, the repository used will be in your namespace and named after the output directory you set, so in our case it will be `"lewtun/distilbert-finetuned-imdb"`.

We now have all the ingredients to instantiate the `Trainer`. Here we just use the standard `data_collator`, but you can try the whole word masking collator and compare the results as an exercise: 

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

We're now ready to run `trainer.train()` -- but before doing so let's briefly look at _perplexity_, which is a common metric to evaluate the performance of language models.

{/if}

### Perplexity for language models[[perplexity-for-language-models]]

<Youtube id="NURcDHhYe98"/>

Unlike other tasks like text classification or question answering where we're given a labeled corpus to train on, with language modeling we don't have any explicit labels. So how do we determine what makes a good language model? Like with the autocorrect feature in your phone, a good language model is one that assigns high probabilities to sentences that are grammatically correct, and low probabilities to nonsense sentences. To give you a better idea of what this looks like, you can find whole sets of "autocorrect fails" online, where the model in a person's phone has produced some rather funny (and often inappropriate) completions! 

{#if fw === 'pt'}

Assuming our test set consists mostly of sentences that are grammatically correct, then one way to measure the quality of our language model is to calculate the probabilities it assigns to the next word in all the sentences of the test set. High probabilities indicates that the model is not "surprised" or "perplexed" by the unseen examples, and suggests it has learned the basic patterns of grammar in the language. There are various mathematical definitions of perplexity, but the one we'll use defines it as the exponential of the cross-entropy loss. Thus, we can calculate the perplexity of our pretrained model by using the `Trainer.evaluate()` function to compute the cross-entropy loss on the test set and then taking the exponential of the result:

```python
import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

{:else}

Assuming our test set consists mostly of sentences that are grammatically correct, then one way to measure the quality of our language model is to calculate the probabilities it assigns to the next word in all the sentences of the test set. High probabilities indicates that the model indicates that the model is not "surprised" or "perplexed" by the unseen examples, and suggests it has learned the basic patterns of grammar in the language. There are various mathematical definitions of perplexity, but the one we'll use defines it as the exponential of the cross-entropy loss. Thus, we can calculate the perplexity of our pretrained model by using the `model.evaluate()` method to compute the cross-entropy loss on the test set and then taking the exponential of the result:

```python
import math

eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")
```

{/if}

```python out
>>> Perplexity: 21.75
```

A lower perplexity score means a better language model, and we can see here that our starting model has a somewhat large value. Let's see if we can lower it by fine-tuning! To do that, we first run the training loop:

{#if fw === 'pt'}

```python
trainer.train()
```

{:else}

```python
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])
```

{/if}

and then compute the resulting perplexity on the test set as before:

{#if fw === 'pt'}

```python
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

{:else}

```python
eval_loss = model.evaluate(tf_eval_dataset)
print(f"Perplexity: {math.exp(eval_loss):.2f}")
```

{/if}

```python out
>>> Perplexity: 11.32
```

Nice -- this is quite a reduction in perplexity, which tells us the model has learned something about the domain of movie reviews!

{#if fw === 'pt'}

Once training is finished, we can push the model card with the training information to the Hub (the checkpoints are saved during training itself):

```python
trainer.push_to_hub()
```

{/if}

> [!TIP]
> ✏️ **Your turn!** Run the training above after changing the data collator to the whole word masking collator. Do you get better results?

{#if fw === 'pt'} 

In our use case we didn't need to do anything special with the training loop, but in some cases you might need to implement some custom logic. For these applications, you can use 🤗 Accelerate -- let's take a look!

## Fine-tuning DistilBERT with 🤗 Accelerate[[fine-tuning-distilbert-with-accelerate]]

As we saw with the `Trainer`, fine-tuning a masked language model is very similar to the text classification example from [Chapter 3](/course/chapter3). In fact, the only subtlety is the use of a special data collator, and we've already covered that earlier in this section! 

However, we saw that `DataCollatorForLanguageModeling` also applies random masking with each evaluation, so we'll see some fluctuations in our perplexity scores with each training run. One way to eliminate this source of randomness is to apply the masking _once_ on the whole test set, and then use the default data collator in 🤗 Transformers to collect the batches during evaluation. To see how this works, let's implement a simple function that applies the masking on a batch, similar to our first encounter with `DataCollatorForLanguageModeling`:

```python
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
```

Next, we'll apply this function to our test set and drop the unmasked columns so we can replace them with the masked ones. You can use whole word masking by replacing the `data_collator` above with the appropriate one, in which case you should remove the first line here:

```py
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
```

We can then set up the dataloaders as usual, but we'll use the `default_data_collator` from 🤗 Transformers for the evaluation set:

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)
```

Form here, we follow the standard steps with 🤗 Accelerate. The first order of business is to load a fresh version of the pretrained model:

```
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

Then we need to specify the optimizer; we'll use the standard `AdamW`:

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

With these objects, we can now prepare everything for training with the `Accelerator` object:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

Now that our model, optimizer, and dataloaders are configured, we can specify the learning rate scheduler as follows:

```python
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

There is just one last thing to do before training: create a model repository on the Hugging Face Hub! We can use the 🤗 Hub library to first generate the full name of our repo:

```python
from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
"lewtun/distilbert-base-uncased-finetuned-imdb-accelerate"
```

then create and clone the repository using the `Repository` class from 🤗 Hub:

```python
from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)
```

With that done, it's just a simple matter of writing out the full training and evaluation loop:

```python
from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
```

```python out
>>> Epoch 0: Perplexity: 11.397545307900472
>>> Epoch 1: Perplexity: 10.904909330983092
>>> Epoch 2: Perplexity: 10.729503505340409
```

Cool, we've been able to evaluate perplexity with each epoch and ensure that multiple training runs are reproducible!

{/if}

## Using our fine-tuned model[[using-our-fine-tuned-model]]

You can interact with your fine-tuned model either by using its widget on the Hub or locally with the `pipeline` from 🤗 Transformers. Let's use the latter to download our model using the `fill-mask` pipeline:

```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
```

We can then feed the pipeline our sample text of "This is a great [MASK]" and see what the top 5 predictions are:

```python
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
```

```python out
">>> this is a great movie."

">>> this is a great film."
">>> this is a great story."
">>> this is a great movies."
">>> this is a great character."
```

Neat -- our model has clearly adapted its weights to predict words that are more strongly associated with movies!

<Youtube id="0Oxphw4Q9fo"/>

This wraps up our first experiment with training a language model. In [section 6](/course/en/chapter7/6) you'll learn how to train an auto-regressive model like GPT-2 from scratch; head over there if you'd like to see how you can pretrain your very own Transformer model!

> [!TIP]
> ✏️ **Try it out!** To quantify the benefits of domain adaptation, fine-tune a classifier on the IMDb labels for both the pretrained and fine-tuned DistilBERT checkpoints. If you need a refresher on text classification, check out [Chapter 3](/course/chapter3).


---

<!-- Section 7.4 -->

<FrameworkSwitchCourse {fw} />

# Translation[[translation]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section4_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section4_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section4_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section4_tf.ipynb"},
]} />

{/if}

Let's now dive into translation. This is another [sequence-to-sequence task](/course/chapter1/7), which means it's a problem that can be formulated as going from one sequence to another. In that sense the problem is pretty close to [summarization](/course/chapter7/6), and you could adapt what we will see here to other sequence-to-sequence problems such as:

- **Style transfer**: Creating a model that *translates* texts written in a certain style to another (e.g., formal to casual or Shakespearean English to modern English)
- **Generative question answering**: Creating a model that generates answers to questions, given a context

<Youtube id="1JvfrvZgi6c"/>

If you have a big enough corpus of texts in two (or more) languages, you can train a new translation model from scratch like we will in the section on [causal language modeling](/course/chapter7/6). It will be faster, however, to fine-tune an existing translation model, be it a multilingual one like mT5 or mBART that you want to fine-tune to a specific language pair, or even a model specialized for translation from one language to another that you want to fine-tune to your specific corpus.

In this section, we will fine-tune a Marian model pretrained to translate from English to French (since a lot of Hugging Face employees speak both those languages) on the [KDE4 dataset](https://huggingface.co/datasets/kde4), which is a dataset of localized files for the [KDE apps](https://apps.kde.org/). The model we will use has been pretrained on a large corpus of French and English texts taken from the [Opus dataset](https://opus.nlpl.eu/), which actually contains the KDE4 dataset. But even if the pretrained model we use has seen that data during its pretraining, we will see that we can get a better version of it after fine-tuning.

Once we're finished, we will have a model able to make predictions like this one:

<iframe src="https://course-demos-marian-finetuned-kde4-en-to-fr.hf.space" frameBorder="0" height="350" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

<a class="flex justify-center" href="/huggingface-course/marian-finetuned-kde4-en-to-fr">
<img class="block dark:hidden lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/modeleval-marian-finetuned-kde4-en-to-fr.png" alt="One-hot encoded labels for question answering."/>
<img class="hidden dark:block lg:w-3/5" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/modeleval-marian-finetuned-kde4-en-to-fr-dark.png" alt="One-hot encoded labels for question answering."/>
</a>

As in the previous sections, you can find the actual model that we'll train and upload to the Hub using the code below and double-check its predictions [here](https://huggingface.co/huggingface-course/marian-finetuned-kde4-en-to-fr?text=This+plugin+allows+you+to+automatically+translate+web+pages+between+several+languages.).

## Preparing the data[[preparing-the-data]]

To fine-tune or train a translation model from scratch, we will need a dataset suitable for the task. As mentioned previously, we'll use the [KDE4 dataset](https://huggingface.co/datasets/kde4) in this section, but you can adapt the code to use your own data quite easily, as long as you have pairs of sentences in the two languages you want to translate from and into. Refer back to [Chapter 5](/course/chapter5) if you need a reminder of how to load your custom data in a `Dataset`.

### The KDE4 dataset[[the-kde4-dataset]]

As usual, we download our dataset using the `load_dataset()` function:

```py
from datasets import load_dataset

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
```

If you want to work with a different pair of languages, you can specify them by their codes. A total of 92 languages are available for this dataset; you can see them all by expanding the language tags on its [dataset card](https://huggingface.co/datasets/kde4).

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/language_tags.png" alt="Language available for the KDE4 dataset." width="100%">

Let's have a look at the dataset:

```py
raw_datasets
```

```python out
DatasetDict({train: Dataset({features: ["id", "translation"], num_rows: 210173})})
```

We have 210,173 pairs of sentences, but in one single split, so we will need to create our own validation set. As we saw in [Chapter 5](/course/chapter5), a `Dataset` has a `train_test_split()` method that can help us. We'll provide a seed for reproducibility:

```py
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 189155
    })
    test: Dataset({
        features: ['id', 'translation'],
        num_rows: 21018
    })
})
```

We can rename the `"test"` key to `"validation"` like this:

```py
split_datasets["validation"] = split_datasets.pop("test")
```

Now let's take a look at one element of the dataset:

```py
split_datasets["train"][1]["translation"]
```

```python out
{"en": "Default to expanded threads", "fr": "Par défaut, développer les fils de discussion"}
```

We get a dictionary with two sentences in the pair of languages we requested. One particularity of this dataset full of technical computer science terms is that they are all fully translated in French. However, French engineers leave most computer science-specific words in English when they talk. Here, for instance, the word "threads" might well appear in a French sentence, especially in a technical conversation; but in this dataset it has been translated into the more correct "fils de discussion." The pretrained model we use, which has been pretrained on a larger corpus of French and English sentences, takes the easier option of leaving the word as is:

```py
from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
```

```python out
[{"translation_text": "Par défaut pour les threads élargis"}]
```

Another example of this behavior can be seen with the word "plugin," which isn't officially a French word but which most native speakers will understand and not bother to translate.
In the KDE4 dataset this word has been translated in French into the more official "module d'extension":

```py
split_datasets["train"][172]["translation"]
```

```python out
{
    "en": "Unable to import %1 using the OFX importer plugin. This file is not the correct format.",
    "fr": "Impossible d'importer %1 en utilisant le module d'extension d'importation OFX. Ce fichier n'a pas un format correct.",
}
```

Our pretrained model, however, sticks with the compact and familiar English word:

```py
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)
```

```python out
[
    {
        "translation_text": "Impossible d'importer %1 en utilisant le plugin d'importateur OFX. Ce fichier n'est pas le bon format."
    }
]
```

It will be interesting to see if our fine-tuned model picks up on those particularities of the dataset (spoiler alert: it will).

<Youtube id="0Oxphw4Q9fo"/>

> [!TIP]
> ✏️ **Your turn!** Another English word that is often used in French is "email." Find the first sample in the training dataset that uses this word. How is it translated? How does the pretrained model translate the same English sentence?

### Processing the data[[processing-the-data]]

<Youtube id="XAR8jnZZuUs"/>

You should know the drill by now: the texts all need to be converted into sets of token IDs so the model can make sense of them. For this task, we'll need to tokenize both the inputs and the targets. Our first task is to create our `tokenizer` object. As noted earlier, we'll be using a Marian English to French pretrained model. If you are trying this code with another pair of languages, make sure to adapt the model checkpoint. The [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) organization provides more than a thousand models in multiple languages.

```python
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
```

You can also replace the `model_checkpoint` with any other model you prefer from the [Hub](https://huggingface.co/models), or a local folder where you've saved a pretrained model and a tokenizer.

> [!TIP]
> 💡 If you are using a multilingual tokenizer such as mBART, mBART-50, or M2M100, you will need to set the language codes of your inputs and targets in the tokenizer by setting `tokenizer.src_lang` and `tokenizer.tgt_lang` to the right values.

The preparation of our data is pretty straightforward. There's just one thing to remember; you need to ensure that the tokenizer processes the targets in the output language (here, French). You can do this by passing the targets to the `text_targets` argument of the tokenizer's `__call__` method.

To see how this works, let's process one sample of each language in the training set:

```python
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
inputs
```

```python out
{
    "input_ids": [47591, 12, 9842, 19634, 9, 0],
    "attention_mask": [1, 1, 1, 1, 1, 1],
    "labels": [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0],
}
```

As we can see, the output contains the input IDs associated with the English sentence, while the IDs associated with the French one are stored in the `labels` field. If you forget to indicate that you are tokenizing labels, they will be tokenized by the input tokenizer, which in the case of a Marian model is not going to go well at all:

```python
wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))
```

```python out
[
    "▁Par",
    "▁dé",
    "f",
    "aut",
    ",",
    "▁dé",
    "ve",
    "lop",
    "per",
    "▁les",
    "▁fil",
    "s",
    "▁de",
    "▁discussion",
    "</s>",
]
["▁Par", "▁défaut", ",", "▁développer", "▁les", "▁fils", "▁de", "▁discussion", "</s>"]
```

As we can see, using the English tokenizer to preprocess a French sentence results in a lot more tokens, since the tokenizer doesn't know any French words (except those that also appear in the English language, like "discussion").

Since `inputs` is a dictionary with our usual keys (input IDs, attention mask, etc.), the last step is to define the preprocessing function we will apply on the datasets:

```python
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs
```

Note that we set the same maximum length for our inputs and outputs. Since the texts we're dealing with seem pretty short, we use 128.

> [!TIP]
> 💡 If you are using a T5 model (more specifically, one of the `t5-xxx` checkpoints), the model will expect the text inputs to have a prefix indicating the task at hand, such as `translate: English to French:`.

> [!WARNING]
> ⚠️ We don't pay attention to the attention mask of the targets, as the model won't expect it. Instead, the labels corresponding to a padding token should be set to `-100` so they are ignored in the loss computation. This will be done by our data collator later on since we are applying dynamic padding, but if you use padding here, you should adapt the preprocessing function to set all labels that correspond to the padding token to `-100`.

We can now apply that preprocessing in one go on all the splits of our dataset:

```py
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
```

Now that the data has been preprocessed, we are ready to fine-tune our pretrained model!

{#if fw === 'pt'}

## Fine-tuning the model with the `Trainer` API[[fine-tuning-the-model-with-the-trainer-api]]

The actual code using the `Trainer` will be the same as before, with just one little change: we use a [`Seq2SeqTrainer`](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainer) here, which is a subclass of `Trainer` that will allow us to properly deal with the evaluation, using the `generate()` method to predict outputs from the inputs. We'll dive into that in more detail when we talk about the metric computation.

First things first, we need an actual model to fine-tune. We'll use the usual `AutoModel` API:

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

{:else}

## Fine-tuning the model with Keras[[fine-tuning-the-model-with-keras]]

First things first, we need an actual model to fine-tune. We'll use the usual `AutoModel` API:

```py
from transformers import TFAutoModelForSeq2SeqLM

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt=True)
```

<Tip warning={false}>

💡 The `Helsinki-NLP/opus-mt-en-fr` checkpoint only has PyTorch weights, so
you'll get an error if you try to load the model without using the
`from_pt=True` argument in the `from_pretrained()` method. When you specify
`from_pt=True`, the library will automatically download and convert the
PyTorch weights for you. As you can see, it is very simple to switch between
frameworks in 🤗 Transformers!

</Tip>

{/if}

Note that this time we are using a model that was trained on a translation task and can actually be used already, so there is no warning about missing weights or newly initialized ones.

### Data collation[[data-collation]]

We'll need a data collator to deal with the padding for dynamic batching. We can't just use a `DataCollatorWithPadding` like in [Chapter 3](/course/chapter3) in this case, because that only pads the inputs (input IDs, attention mask, and token type IDs). Our labels should also be padded to the maximum length encountered in the labels. And, as mentioned previously, the padding value used to pad the labels should be `-100` and not the padding token of the tokenizer, to make sure those padded values are ignored in the loss computation.

This is all done by a [`DataCollatorForSeq2Seq`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorforseq2seq). Like the `DataCollatorWithPadding`, it takes the `tokenizer` used to preprocess the inputs, but it also takes the `model`. This is because this data collator will also be responsible for preparing the decoder input IDs, which are shifted versions of the labels with a special token at the beginning. Since this shift is done slightly differently for different architectures, the `DataCollatorForSeq2Seq` needs to know the `model` object:

{#if fw === 'pt'}

```py
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

{:else}

```py
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
```

{/if}

To test this on a few samples, we just call it on a list of examples from our tokenized training set:

```py
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
batch.keys()
```

```python out
dict_keys(["attention_mask", "input_ids", "labels", "decoder_input_ids"])
```

We can check our labels have been padded to the maximum length of the batch, using `-100`:

```py
batch["labels"]
```

```python out
tensor(
    [
        [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0, -100, -100, -100, -100, -100, -100, -100],
        [1211, 3, 49, 9409, 1211, 3, 29140, 817, 3124, 817, 550, 7032, 5821, 7907, 12649, 0],
    ]
)
```

And we can also have a look at the decoder input IDs, to see that they are shifted versions of the labels:

```py
batch["decoder_input_ids"]
```

```python out
tensor(
    [
        [59513, 577, 5891, 2, 3184, 16, 2542, 5, 1710, 0, 59513, 59513, 59513, 59513, 59513, 59513],
        [59513, 1211, 3, 49, 9409, 1211, 3, 29140, 817, 3124, 817, 550, 7032, 5821, 7907, 12649],
    ]
)
```

Here are the labels for the first and second elements in our dataset:

```py
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])
```

```python out
[577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]
[1211, 3, 49, 9409, 1211, 3, 29140, 817, 3124, 817, 550, 7032, 5821, 7907, 12649, 0]
```

{#if fw === 'pt'}

We will pass this `data_collator` along to the `Seq2SeqTrainer`. Next, let's have a look at the metric.

{:else}

We can now use this `data_collator` to convert each of our datasets to a `tf.data.Dataset`, ready for training:

```python
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
```

{/if}


### Metrics[[metrics]]

<Youtube id="M05L1DhFqcw"/>

{#if fw === 'pt'}

The feature that `Seq2SeqTrainer` adds to its superclass `Trainer` is the ability to use the `generate()` method during evaluation or prediction. During training, the model will use the `decoder_input_ids` with an attention mask ensuring it does not use the tokens after the token it's trying to predict, to speed up training. During inference we won't be able to use those since we won't have labels, so it's a good idea to evaluate our model with the same setup.

As we saw in [Chapter 1](/course/chapter1/6), the decoder performs inference by predicting tokens one by one -- something that's implemented behind the scenes in 🤗 Transformers by the `generate()` method. The `Seq2SeqTrainer` will let us use that method for evaluation if we set `predict_with_generate=True`.

{/if}

The traditional metric used for translation is the [BLEU score](https://en.wikipedia.org/wiki/BLEU), introduced in [a 2002 article](https://aclanthology.org/P02-1040.pdf) by Kishore Papineni et al. The BLEU score evaluates how close the translations are to their labels. It does not measure the intelligibility or grammatical correctness of the model's generated outputs, but uses statistical rules to ensure that all the words in the generated outputs also appear in the targets. In addition, there are rules that penalize repetitions of the same words if they are not also repeated in the targets (to avoid the model outputting sentences like `"the the the the the"`) and output sentences that are shorter than those in the targets (to avoid the model outputting sentences like `"the"`).

One weakness with BLEU is that it expects the text to already be tokenized, which makes it difficult to compare scores between models that use different tokenizers. So instead, the most commonly used metric for benchmarking translation models today is [SacreBLEU](https://github.com/mjpost/sacrebleu), which addresses this weakness (and others) by standardizing the tokenization step. To use this metric, we first need to install the SacreBLEU library:

```py
!pip install sacrebleu
```

We can then load it via `evaluate.load()` like we did in [Chapter 3](/course/chapter3):

```py
import evaluate

metric = evaluate.load("sacrebleu")
```

This metric will take texts as inputs and targets. It is designed to accept several acceptable targets, as there are often multiple acceptable translations of the same sentence -- the dataset we're using only provides one, but it's not uncommon in NLP to find datasets that give several sentences as labels. So, the predictions should be a list of sentences, but the references should be a list of lists of sentences.

Let's try an example:

```py
predictions = ["This plugin lets you translate web pages between several languages automatically."]
references = [
    ["This plugin allows you to automatically translate web pages between several languages."]
]
metric.compute(predictions=predictions, references=references)
```

```python out
{
    "score": 46.750469682990165,
    "counts": [11, 6, 4, 3],
    "totals": [12, 11, 10, 9],
    "precisions": [91.67, 54.54, 40.0, 33.33],
    "bp": 0.9200444146293233,
    "sys_len": 12,
    "ref_len": 13,
}
```

This gets a BLEU score of 46.75, which is rather good -- for reference, the original Transformer model in the ["Attention Is All You Need" paper](https://arxiv.org/pdf/1706.03762.pdf) achieved a BLEU score of 41.8 on a similar translation task between English and French! (For more information about the individual metrics, like `counts` and `bp`, see the [SacreBLEU repository](https://github.com/mjpost/sacrebleu/blob/078c440168c6adc89ba75fe6d63f0d922d42bcfe/sacrebleu/metrics/bleu.py#L74).) On the other hand, if we try with the two bad types of predictions (lots of repetitions or too short) that often come out of translation models, we will get rather bad BLEU scores:

```py
predictions = ["This This This This"]
references = [
    ["This plugin allows you to automatically translate web pages between several languages."]
]
metric.compute(predictions=predictions, references=references)
```

```python out
{
    "score": 1.683602693167689,
    "counts": [1, 0, 0, 0],
    "totals": [4, 3, 2, 1],
    "precisions": [25.0, 16.67, 12.5, 12.5],
    "bp": 0.10539922456186433,
    "sys_len": 4,
    "ref_len": 13,
}
```

```py
predictions = ["This plugin"]
references = [
    ["This plugin allows you to automatically translate web pages between several languages."]
]
metric.compute(predictions=predictions, references=references)
```

```python out
{
    "score": 0.0,
    "counts": [2, 1, 0, 0],
    "totals": [2, 1, 0, 0],
    "precisions": [100.0, 100.0, 0.0, 0.0],
    "bp": 0.004086771438464067,
    "sys_len": 2,
    "ref_len": 13,
}
```

The score can go from 0 to 100, and higher is better.

{#if fw === 'tf'}

To get from the model outputs to texts the metric can use, we will use the `tokenizer.batch_decode()` method. We just have to clean up all the `-100`s in the labels; the tokenizer will automatically do the same for the padding token. Let's define a function that takes our model and a dataset and computes metrics on it. We're also going to use a trick that dramatically increases performance - compiling our generation code with [XLA](https://www.tensorflow.org/xla), TensorFlow's accelerated linear algebra compiler. XLA applies various optimizations to the model's computation graph, and results in significant improvements to speed and memory usage. As described in the Hugging Face [blog](https://huggingface.co/blog/tf-xla-generate), XLA works best when our input shapes don't vary too much. To handle this, we'll pad our inputs to multiples of 128, and make a new dataset with the padding collator, and then we'll apply the `@tf.function(jit_compile=True)` decorator to our generation function, which marks the whole function for compilation with XLA. 

```py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

generation_data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128
)

tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
)


@tf.function(jit_compile=True)
def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=128,
    )


def compute_metrics():
    all_preds = []
    all_labels = []

    for batch, labels in tqdm(tf_generate_dataset):
        predictions = generate_with_xla(batch)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = labels.numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"bleu": result["score"]}
```

{:else}

To get from the model outputs to texts the metric can use, we will use the `tokenizer.batch_decode()` method. We just have to clean up all the `-100`s in the labels (the tokenizer will automatically do the same for the padding token):

```py
import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
```

{/if}

Now that this is done, we are ready to fine-tune our model!


### Fine-tuning the model[[fine-tuning-the-model]]

The first step is to log in to Hugging Face, so you're able to upload your results to the Model Hub. There's a convenience function to help you with this in a notebook:

```python
from huggingface_hub import notebook_login

notebook_login()
```

This will display a widget where you can enter your Hugging Face login credentials.

If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

{#if fw === 'tf'}

Before we start, let's see what kind of results we get from our model without any training:

```py
print(compute_metrics())
```

```
{'bleu': 33.26983701454733}
```

Once this is done, we can prepare everything we need to compile and train our model. Note the use of `tf.keras.mixed_precision.set_global_policy("mixed_float16")` -- this will tell Keras to train using float16, which can give a significant speedup on GPUs that support it (Nvidia 20xx/V100 or newer).

```python
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Next, we define a `PushToHubCallback` to upload our model to the Hub during training, as we saw in [section 2]((/course/chapter7/2)), and then we simply fit the model with that callback:

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="marian-finetuned-kde4-en-to-fr", tokenizer=tokenizer)

model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)
```

Note that you can specify the name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, when we pushed the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/marian-finetuned-kde4-en-to-fr"` to `Seq2SeqTrainingArguments`. By default, the repository used will be in your namespace and named after the output directory you set, so here it will be `"sgugger/marian-finetuned-kde4-en-to-fr"` (which is the model we linked to at the beginning of this section).

> [!TIP]
> 💡 If the output directory you are using already exists, it needs to be a local clone of the repository you want to push to. If it isn't, you'll get an error when calling `model.fit()` and will need to set a new name.

Finally, let's see what our metrics look like now that training has finished:

```py
print(compute_metrics())
```

```
{'bleu': 57.334066271545865}
```

At this stage, you can use the inference widget on the Model Hub to test your model and share it with your friends. You have successfully fine-tuned a model on a translation task -- congratulations!

{:else}

Once this is done, we can define our `Seq2SeqTrainingArguments`. Like for the `Trainer`, we use a subclass of `TrainingArguments` that contains a few more fields:

```python
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
```

Apart from the usual hyperparameters (like learning rate, number of epochs, batch size, and some weight decay), here are a few changes compared to what we saw in the previous sections:

- We don't set any regular evaluation, as evaluation takes a while; we will just evaluate our model once before training and after.
- We set `fp16=True`, which speeds up training on modern GPUs.
- We set `predict_with_generate=True`, as discussed above.
- We use `push_to_hub=True` to upload the model to the Hub at the end of each epoch.

Note that you can specify the full name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, when we pushed the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/marian-finetuned-kde4-en-to-fr"` to `Seq2SeqTrainingArguments`. By default, the repository used will be in your namespace and named after the output directory you set, so in our case it will be `"sgugger/marian-finetuned-kde4-en-to-fr"` (which is the model we linked to at the beginning of this section).

> [!TIP]
> 💡 If the output directory you are using already exists, it needs to be a local clone of the repository you want to push to. If it isn't, you'll get an error when defining your `Seq2SeqTrainer` and will need to set a new name.


Finally, we just pass everything to the `Seq2SeqTrainer`:

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

Before training, we'll first look at the score our model gets, to double-check that we're not making things worse with our fine-tuning. This command will take a bit of time, so you can grab a coffee while it executes:

```python
trainer.evaluate(max_length=max_length)
```

```python out
{
    "eval_loss": 1.6964408159255981,
    "eval_bleu": 39.26865061007616,
    "eval_runtime": 965.8884,
    "eval_samples_per_second": 21.76,
    "eval_steps_per_second": 0.341,
}
```

A BLEU score of 39 is not too bad, which reflects the fact that our model is already good at translating English sentences to French ones.

Next is the training, which will also take a bit of time:

```python
trainer.train()
```

Note that while the training happens, each time the model is saved (here, every epoch) it is uploaded to the Hub in the background. This way, you will be able to to resume your training on another machine if necessary.

Once training is done, we evaluate our model again -- hopefully we will see some amelioration in the BLEU score!

```py
trainer.evaluate(max_length=max_length)
```

```python out
{
    "eval_loss": 0.8558505773544312,
    "eval_bleu": 52.94161337775576,
    "eval_runtime": 714.2576,
    "eval_samples_per_second": 29.426,
    "eval_steps_per_second": 0.461,
    "epoch": 3.0,
}
```

That's a nearly 14-point improvement, which is great.

Finally, we use the `push_to_hub()` method to make sure we upload the latest version of the model. The `Trainer` also drafts a model card with all the evaluation results and uploads it. This model card contains metadata that helps the Model Hub pick the widget for the inference demo. Usually, there is no need to say anything as it can infer the right widget from the model class, but in this case, the same model class can be used for all kinds of sequence-to-sequence problems, so we specify it's a translation model:

```py
trainer.push_to_hub(tags="translation", commit_message="Training complete")
```

This command returns the URL of the commit it just did, if you want to inspect it:

```python out
"https://huggingface.co/sgugger/marian-finetuned-kde4-en-to-fr/commit/3601d621e3baae2bc63d3311452535f8f58f6ef3"
```

At this stage, you can use the inference widget on the Model Hub to test your model and share it with your friends. You have successfully fine-tuned a model on a translation task -- congratulations!

If you want to dive a bit more deeply into the training loop, we will now show you how to do the same thing using 🤗 Accelerate.

{/if}

{#if fw === 'pt'}

## A custom training loop[[a-custom-training-loop]]

Let's now take a look at the full training loop, so you can easily customize the parts you need. It will look a lot like what we did in [section 2](/course/chapter7/2) and [Chapter 3](/course/chapter3/4).

### Preparing everything for training[[preparing-everything-for-training]]

You've seen all of this a few times now, so we'll go through the code quite quickly. First we'll build the `DataLoader`s from our datasets, after setting the datasets to the `"torch"` format so we get PyTorch tensors:

```py
from torch.utils.data import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
```

Next we reinstantiate our model, to make sure we're not continuing the fine-tuning from before but starting from the pretrained model again:

```py
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

Then we will need an optimizer:

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Once we have all those objects, we can send them to the `accelerator.prepare()` method. Remember that if you want to train on TPUs in a Colab notebook, you will need to move all of this code into a training function, and that shouldn't execute any cell that instantiates an `Accelerator`.

```py
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

Now that we have sent our `train_dataloader` to `accelerator.prepare()`, we can use its length to compute the number of training steps. Remember we should always do this after preparing the dataloader, as that method will change the length of the `DataLoader`. We use a classic linear schedule from the learning rate to 0:

```py
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

Lastly, to push our model to the Hub, we will need to create a `Repository` object in a working folder. First log in to the Hugging Face Hub, if you're not logged in already. We'll determine the repository name from the model ID we want to give our model (feel free to replace the `repo_name` with your own choice; it just needs to contain your username, which is what the function `get_full_repo_name()` does):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
"sgugger/marian-finetuned-kde4-en-to-fr-accelerate"
```

Then we can clone that repository in a local folder. If it already exists, this local folder should be a clone of the repository we are working with:

```py
output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

We can now upload anything we save in `output_dir` by calling the `repo.push_to_hub()` method. This will help us upload the intermediate models at the end of each epoch.

### Training loop[[training-loop]]

We are now ready to write the full training loop. To simplify its evaluation part, we define this `postprocess()` function that takes predictions and labels and converts them to the lists of strings our `metric` object will expect:

```py
def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels
```

The training loop looks a lot like the ones in [section 2](/course/chapter7/2) and [Chapter 3](/course/chapter3), with a few differences in the evaluation part -- so let's focus on that!

The first thing to note is that we use the `generate()` method to compute predictions, but this is a method on our base model, not the wrapped model 🤗 Accelerate created in the `prepare()` method. That's why we unwrap the model first, then call this method.

The second thing is that, like with [token classification](/course/chapter7/2), two processes may have padded the inputs and labels to different shapes, so we use `accelerator.pad_across_processes()` to make the predictions and labels the same shape before calling the `gather()` method. If we don't do this, the evaluation will either error out or hang forever.

```py
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
```

```python out
epoch 0, BLEU score: 53.47
epoch 1, BLEU score: 54.24
epoch 2, BLEU score: 54.44
```

Once this is done, you should have a model that has results pretty similar to the one trained with the `Seq2SeqTrainer`. You can check the one we trained using this code at [*huggingface-course/marian-finetuned-kde4-en-to-fr-accelerate*](https://huggingface.co/huggingface-course/marian-finetuned-kde4-en-to-fr-accelerate). And if you want to test out any tweaks to the training loop, you can directly implement them by editing the code shown above!

{/if}

## Using the fine-tuned model[[using-the-fine-tuned-model]]

We've already shown you how you can use the model we fine-tuned on the Model Hub with the inference widget. To use it locally in a `pipeline`, we just have to specify the proper model identifier:

```py
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
```

```python out
[{"translation_text": "Par défaut, développer les fils de discussion"}]
```

As expected, our pretrained model adapted its knowledge to the corpus we fine-tuned it on, and instead of leaving the English word "threads" alone, it now translates it to the French official version. It's the same for "plugin":

```py
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)
```

```python out
[
    {
        "translation_text": "Impossible d'importer %1 en utilisant le module externe d'importation OFX. Ce fichier n'est pas le bon format."
    }
]
```

Another great example of domain adaptation!

> [!TIP]
> ✏️ **Your turn!** What does the model return on the sample with the word "email" you identified earlier?


---

<!-- Section 7.5 -->

<FrameworkSwitchCourse {fw} />

# Summarization[[summarization]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section5_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section5_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section5_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section5_tf.ipynb"},
]} />

{/if}


In this section we'll take a look at how Transformer models can be used to condense long documents into summaries, a task known as _text summarization_. This is one of the most challenging NLP tasks as it requires a range of abilities, such as understanding long passages and generating coherent text that captures the main topics in a document. However, when done well, text summarization is a powerful tool that can speed up various business processes by relieving the burden of domain experts to read long documents in detail.

<Youtube id="yHnr5Dk2zCI"/>

Although there already exist various fine-tuned models for summarization on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=summarization&sort=downloads), almost all of these are only suitable for English documents. So, to add a twist in this section, we'll train a bilingual model for English and Spanish. By the end of this section, you'll have a [model](https://huggingface.co/huggingface-course/mt5-small-finetuned-amazon-en-es) that can summarize customer reviews like the one shown here:

<iframe src="https://course-demos-mt5-small-finetuned-amazon-en-es.hf.space" frameBorder="0" height="400" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

As we'll see, these summaries are concise because they're learned from the titles that customers provide in their product reviews. Let's start by putting together a suitable bilingual corpus for this task.

## Preparing a multilingual corpus[[preparing-a-multilingual-corpus]]

We'll use the [Multilingual Amazon Reviews Corpus](https://huggingface.co/datasets/amazon_reviews_multi) to create our bilingual summarizer. This corpus consists of Amazon product reviews in six languages and is typically used to benchmark multilingual classifiers. However, since each review is accompanied by a short title, we can use the titles as the target summaries for our model to learn from! To get started, let's download the English and Spanish subsets from the Hugging Face Hub:

```python
from datasets import load_dataset

spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")
english_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 200000
    })
    validation: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'],
        num_rows: 5000
    })
})
```

As you can see, for each language there are 200,000 reviews for the `train` split, and 5,000 reviews for each of the `validation` and `test` splits. The review information we are interested in is contained in the `review_body` and `review_title` columns. Let's take a look at a few examples by creating a simple function that takes a random sample from the training set with the techniques we learned in [Chapter 5](/course/chapter5):

```python
def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}'")


show_samples(english_dataset)
```

```python out
">> Title: Worked in front position, not rear"

">> Review: 3 stars because these are not rear brakes as stated in the item description. At least the mount adapter only worked on the front fork of the bike that I got it for."

">> Title: meh"
">> Review: Does it’s job and it’s gorgeous but mine is falling apart, I had to basically put it together again with hot glue"

">> Title: Can't beat these for the money"
">> Review: Bought this for handling miscellaneous aircraft parts and hanger \"stuff\" that I needed to organize; it really fit the bill. The unit arrived quickly, was well packaged and arrived intact (always a good sign). There are five wall mounts-- three on the top and two on the bottom. I wanted to mount it on the wall, so all I had to do was to remove the top two layers of plastic drawers, as well as the bottom corner drawers, place it when I wanted and mark it; I then used some of the new plastic screw in wall anchors (the 50 pound variety) and it easily mounted to the wall. Some have remarked that they wanted dividers for the drawers, and that they made those. Good idea. My application was that I needed something that I can see the contents at about eye level, so I wanted the fuller-sized drawers. I also like that these are the new plastic that doesn't get brittle and split like my older plastic drawers did. I like the all-plastic construction. It's heavy duty enough to hold metal parts, but being made of plastic it's not as heavy as a metal frame, so you can easily mount it to the wall and still load it up with heavy stuff, or light stuff. No problem there. For the money, you can't beat it. Best one of these I've bought to date-- and I've been using some version of these for over forty years."
```

> [!TIP]
> ✏️ **Try it out!** Change the random seed in the `Dataset.shuffle()` command to explore other reviews in the corpus. If you're a Spanish speaker, take a look at some of the reviews in `spanish_dataset` to see if the titles also seem like reasonable summaries.

This sample shows the diversity of reviews one typically finds online, ranging from positive to negative (and everything in between!). Although the example with the "meh" title is not very informative, the other titles look like decent summaries of the reviews themselves. Training a summarization model on all 400,000 reviews would take far too long on a single GPU, so instead we'll focus on generating summaries for a single domain of products. To get a feel for what domains we can choose from, let's convert `english_dataset` to a `pandas.DataFrame` and compute the number of reviews per product category:

```python
english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
# Show counts for top 20 products
english_df["product_category"].value_counts()[:20]
```

```python out
home                      17679
apparel                   15951
wireless                  15717
other                     13418
beauty                    12091
drugstore                 11730
kitchen                   10382
toy                        8745
sports                     8277
automotive                 7506
lawn_and_garden            7327
home_improvement           7136
pet_products               7082
digital_ebook_purchase     6749
pc                         6401
electronics                6186
office_product             5521
shoes                      5197
grocery                    4730
book                       3756
Name: product_category, dtype: int64
```

The most popular products in the English dataset are about household items, clothing, and wireless electronics. To stick with the Amazon theme, though, let's focus on summarizing book reviews -- after all, this is what the company was founded on! We can see two product categories that fit the bill (`book` and `digital_ebook_purchase`), so let's filter the datasets in both languages for just these products. As we saw in [Chapter 5](/course/chapter5), the `Dataset.filter()` function allows us to slice a dataset very efficiently, so we can define a simple function to do this:

```python
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )
```

Now when we apply this function to `english_dataset` and `spanish_dataset`, the result will contain just those rows involving the book categories. Before applying the filter, let's switch the format of `english_dataset` from `"pandas"` back to `"arrow"`:

```python
english_dataset.reset_format()
```

We can then apply the filter function, and as a sanity check let's inspect a sample of reviews to see if they are indeed about books:

```python
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
show_samples(english_books)
```

```python out
">> Title: I'm dissapointed."

">> Review: I guess I had higher expectations for this book from the reviews. I really thought I'd at least like it. The plot idea was great. I loved Ash but, it just didnt go anywhere. Most of the book was about their radio show and talking to callers. I wanted the author to dig deeper so we could really get to know the characters. All we know about Grace is that she is attractive looking, Latino and is kind of a brat. I'm dissapointed."

">> Title: Good art, good price, poor design"
">> Review: I had gotten the DC Vintage calendar the past two years, but it was on backorder forever this year and I saw they had shrunk the dimensions for no good reason. This one has good art choices but the design has the fold going through the picture, so it's less aesthetically pleasing, especially if you want to keep a picture to hang. For the price, a good calendar"

">> Title: Helpful"
">> Review: Nearly all the tips useful and. I consider myself an intermediate to advanced user of OneNote. I would highly recommend."
```

Okay, we can see that the reviews are not strictly about books and might refer to things like calendars and electronic applications such as OneNote. Nevertheless, the domain seems about right to train a summarization model on. Before we look at various models that are suitable for this task, we have one last bit of data preparation to do: combining the English and Spanish reviews as a single `DatasetDict` object. 🤗 Datasets provides a handy `concatenate_datasets()` function that (as the name suggests) will stack two `Dataset` objects on top of each other. So, to create our bilingual dataset, we'll loop over each split, concatenate the datasets for that split, and shuffle the result to ensure our model doesn't overfit to a single language:

```python
from datasets import concatenate_datasets, DatasetDict

books_dataset = DatasetDict()

for split in english_books.keys():
    books_dataset[split] = concatenate_datasets([english_books[split], spanish_books[split]])
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

# Peek at a few examples
show_samples(books_dataset)
```

```python out
">> Title: Easy to follow!!!!"

">> Review: I loved The dash diet weight loss Solution. Never hungry. I would recommend this diet. Also the menus are well rounded. Try it. Has lots of the information need thanks."

">> Title: PARCIALMENTE DAÑADO"
">> Review: Me llegó el día que tocaba, junto a otros libros que pedí, pero la caja llegó en mal estado lo cual dañó las esquinas de los libros porque venían sin protección (forro)."

">> Title: no lo he podido descargar"
">> Review: igual que el anterior"
```

This certainly looks like a mix of English and Spanish reviews! Now that we have a training corpus, one final thing to check is the distribution of words in the reviews and their titles. This is especially important for summarization tasks, where short reference summaries in the data can bias the model to only output one or two words in the generated summaries. The plots below show the word distributions, and we can see that the titles are heavily skewed toward just 1-2 words:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/review-lengths.svg" alt="Word count distributions for the review titles and texts."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/review-lengths-dark.svg" alt="Word count distributions for the review titles and texts."/>
</div>

To deal with this, we'll filter out the examples with very short titles so that our model can produce more interesting summaries. Since we're dealing with English and Spanish texts, we can use a rough heuristic to split the titles on whitespace and then use our trusty `Dataset.filter()` method as follows:

```python
books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
```

Now that we've prepared our corpus, let's take a look at a few possible Transformer models that one might fine-tune on it!

## Models for text summarization[[models-for-text-summarization]]

If you think about it, text summarization is a similar sort of task to machine translation: we have a body of text like a review that we'd like to "translate" into a shorter version that captures the salient features of the input. Accordingly, most Transformer models for summarization adopt the encoder-decoder architecture that we first encountered in [Chapter 1](/course/chapter1), although there are some exceptions like the GPT family of models which can also be used for summarization in few-shot settings. The following table lists some popular pretrained models that can be fine-tuned for summarization.

| Transformer model | Description                                                                                                                                                                                                    | Multilingual? |
| :---------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: |
|    [GPT-2](https://huggingface.co/gpt2-xl)    | Although trained as an auto-regressive language model, you can make GPT-2 generate summaries by appending "TL;DR" at the end of the input text.                                                                          |      ❌       |
|   [PEGASUS](https://huggingface.co/google/pegasus-large)   | Uses a pretraining objective to predict masked sentences in multi-sentence texts. This pretraining objective is closer to summarization than vanilla language modeling and scores highly on popular benchmarks. |      ❌       |
|     [T5](https://huggingface.co/t5-base)      | A universal Transformer architecture that formulates all tasks in a text-to-text framework; e.g., the input format for the model to summarize a document is `summarize: ARTICLE`.                              |      ❌       |
|     [mT5](https://huggingface.co/google/mt5-base)     | A multilingual version of T5, pretrained on the multilingual Common Crawl corpus (mC4), covering 101 languages.                                                                                                |      ✅       |
|    [BART](https://huggingface.co/facebook/bart-base)     | A novel Transformer architecture with both an encoder and a decoder stack trained to reconstruct corrupted input that combines the pretraining schemes of BERT and GPT-2.                                    |      ❌       |
|  [mBART-50](https://huggingface.co/facebook/mbart-large-50)   | A multilingual version of BART, pretrained on 50 languages.                                                                                                                                                     |      ✅       |

As you can see from this table, the majority of Transformer models for summarization (and indeed most NLP tasks) are monolingual. This is great if your task is in a "high-resource" language like English or German, but less so for the thousands of other languages in use across the world. Fortunately, there is a class of multilingual Transformer models, like mT5 and mBART, that come to the rescue. These models are pretrained using language modeling, but with a twist: instead of training on a corpus of one language, they are trained jointly on texts in over 50 languages at once!

We'll focus on mT5, an interesting architecture based on T5 that was pretrained in a text-to-text framework. In T5, every NLP task is formulated in terms of a prompt prefix like `summarize:` which conditions the model to adapt the generated text to the prompt. As shown in the figure below, this makes T5 extremely versatile, as you can solve many tasks with a single model!

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/t5.svg" alt="Different tasks performed by the T5 architecture."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/t5-dark.svg" alt="Different tasks performed by the T5 architecture."/>
</div>

mT5 doesn't use prefixes, but shares much of the versatility of T5 and has the advantage of being multilingual. Now that we've picked a model, let's take a look at preparing our data for training.


> [!TIP]
> ✏️ **Try it out!** Once you've worked through this section, see how well mT5 compares to mBART by fine-tuning the latter with the same techniques. For bonus points, you can also try fine-tuning T5 on just the English reviews. Since T5 has a special prefix prompt, you'll need to prepend `summarize:` to the input examples in the preprocessing steps below.

## Preprocessing the data[[preprocessing-the-data]]

<Youtube id="1m7BerpSq8A"/>

Our next task is to tokenize and encode our reviews and their titles. As usual, we begin by loading the tokenizer associated with the pretrained model checkpoint. We'll use `mt5-small` as our checkpoint so we can fine-tune the model in a reasonable amount of time:

```python
from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

> [!TIP]
> 💡 In the early stages of your NLP projects, a good practice is to train a class of "small" models on a small sample of data. This allows you to debug and iterate faster toward an end-to-end workflow. Once you are confident in the results, you can always scale up the model by simply changing the model checkpoint!

Let's test out the mT5 tokenizer on a small example:

```python
inputs = tokenizer("I loved reading the Hunger Games!")
inputs
```

```python out
{
    "input_ids": [336, 259, 28387, 11807, 287, 62893, 295, 12507, 1],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],
}
```

Here we can see the familiar `input_ids` and `attention_mask` that we encountered in our first fine-tuning experiments back in [Chapter 3](/course/chapter3). Let's decode these input IDs with the tokenizer's `convert_ids_to_tokens()` function to see what kind of tokenizer we're dealing with:

```python
tokenizer.convert_ids_to_tokens(inputs.input_ids)
```

```python out
["▁I", "▁", "loved", "▁reading", "▁the", "▁Hung", "er", "▁Games", "</s>"]
```

The special Unicode character `▁` and end-of-sequence token `</s>` indicate that we're dealing with the SentencePiece tokenizer, which is based on the Unigram segmentation algorithm discussed in [Chapter 6](/course/chapter6). Unigram is especially useful for multilingual corpora since it allows SentencePiece to be agnostic about accents, punctuation, and the fact that many languages, like Japanese, do not have whitespace characters.

To tokenize our corpus, we have to deal with a subtlety associated with summarization: because our labels are also text, it is possible that they exceed the model's maximum context size. This means we need to apply truncation to both the reviews and their titles to ensure we don't pass excessively long inputs to our model. The tokenizers in 🤗 Transformers provide a nifty `text_target` argument that allows you to tokenize the labels in parallel to the inputs. Here is an example of how the inputs and targets are processed for mT5:

```python
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(examples["review_title"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Let's walk through this code to understand what's happening. The first thing we've done is define values for `max_input_length` and `max_target_length`, which set the upper limits for how long our reviews and titles can be. Since the review body is typically much larger than the title, we've scaled these values accordingly.

With `preprocess_function()`, it is then a simple matter to tokenize the whole corpus using the handy `Dataset.map()` function we've used extensively throughout this course:

```python
tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
```

Now that the corpus has been preprocessed, let's take a look at some metrics that are commonly used for summarization. As we'll see, there is no silver bullet when it comes to measuring the quality of machine-generated text.

> [!TIP]
> 💡 You may have noticed that we used `batched=True` in our `Dataset.map()` function above. This encodes the examples in batches of 1,000 (the default) and allows you to make use of the multithreading capabilities of the fast tokenizers in 🤗 Transformers. Where possible, try using `batched=True` to get the most out of your preprocessing!


## Metrics for text summarization[[metrics-for-text-summarization]]

<Youtube id="TMshhnrEXlg"/>

In comparison to most of the other tasks we've covered in this course, measuring the performance of text generation tasks like summarization or translation is not as straightforward. For example, given a review like "I loved reading the Hunger Games", there are multiple valid summaries, like "I loved the Hunger Games" or "Hunger Games is a great read". Clearly, applying some sort of exact match between the generated summary and the label is not a good solution -- even humans would fare poorly under such a metric, because we all have our own writing style.

For summarization, one of the most commonly used metrics is the [ROUGE score](https://en.wikipedia.org/wiki/ROUGE_(metric)) (short for Recall-Oriented Understudy for Gisting Evaluation). The basic idea behind this metric is to compare a generated summary against a set of reference summaries that are typically created by humans. To make this more precise, suppose we want to compare the following two summaries:

```python
generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
```

One way to compare them could be to count the number of overlapping words, which in this case would be 6. However, this is a bit crude, so instead ROUGE is based on computing the _precision_ and _recall_ scores for the overlap.

> [!TIP]
> 🙋 Don't worry if this is the first time you've heard of precision and recall -- we'll go through some explicit examples together to make it all clear. These metrics are usually encountered in classification tasks, so if you want to understand how precision and recall are defined in that context, we recommend checking out the `scikit-learn` [guides](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).

For ROUGE, recall measures how much of the reference summary is captured by the generated one. If we are just comparing words, recall can be calculated according to the following formula:

$$ \mathrm{Recall} = \frac{\mathrm{Number\,of\,overlapping\, words}}{\mathrm{Total\, number\, of\, words\, in\, reference\, summary}} $$

For our simple example above, this formula gives a perfect recall of 6/6 = 1; i.e., all the words in the reference summary have been produced by the model. This may sound great, but imagine if our generated summary had been "I really really loved reading the Hunger Games all night". This would also have perfect recall, but is arguably a worse summary since it is verbose. To deal with these scenarios we also compute the precision, which in the ROUGE context measures how much of the generated summary was relevant:

$$ \mathrm{Precision} = \frac{\mathrm{Number\,of\,overlapping\, words}}{\mathrm{Total\, number\, of\, words\, in\, generated\, summary}} $$

Applying this to our verbose summary gives a precision of 6/10  = 0.6, which is considerably worse than the precision of 6/7 = 0.86 obtained by our shorter one. In practice, both precision and recall are usually computed, and then the F1-score (the harmonic mean of precision and recall) is reported. We can do this easily in 🤗 Datasets by first installing the `rouge_score` package:

```py
!pip install rouge_score
```

and then loading the ROUGE metric as follows:

```python
import evaluate

rouge_score = evaluate.load("rouge")
```

Then we can use the `rouge_score.compute()` function to calculate all the metrics at once:

```python
scores = rouge_score.compute(predictions=[generated_summary], references=[reference_summary])
scores
```

```python out
{
    "rouge1": AggregateScore(
        low=Score(precision=0.86, recall=1.0, fmeasure=0.92),
        mid=Score(precision=0.86, recall=1.0, fmeasure=0.92),
        high=Score(precision=0.86, recall=1.0, fmeasure=0.92),
    ),
    "rouge2": AggregateScore(
        low=Score(precision=0.67, recall=0.8, fmeasure=0.73),
        mid=Score(precision=0.67, recall=0.8, fmeasure=0.73),
        high=Score(precision=0.67, recall=0.8, fmeasure=0.73),
    ),
    "rougeL": AggregateScore(
        low=Score(precision=0.86, recall=1.0, fmeasure=0.92),
        mid=Score(precision=0.86, recall=1.0, fmeasure=0.92),
        high=Score(precision=0.86, recall=1.0, fmeasure=0.92),
    ),
    "rougeLsum": AggregateScore(
        low=Score(precision=0.86, recall=1.0, fmeasure=0.92),
        mid=Score(precision=0.86, recall=1.0, fmeasure=0.92),
        high=Score(precision=0.86, recall=1.0, fmeasure=0.92),
    ),
}
```

Whoa, there's a lot of information in that output -- what does it all mean? First, 🤗 Datasets actually computes confidence intervals for precision, recall, and F1-score; these are the `low`, `mid`, and `high` attributes you can see here. Moreover, 🤗 Datasets computes a variety of ROUGE scores which are based on different types of text granularity when comparing the generated and reference summaries. The `rouge1` variant is the overlap of unigrams -- this is just a fancy way of saying the overlap of words and is exactly the metric we've discussed above. To verify this, let's pull out the `mid` value of our scores:

```python
scores["rouge1"].mid
```

```python out
Score(precision=0.86, recall=1.0, fmeasure=0.92)
```

Great, the precision and recall numbers match up! Now what about those other ROUGE scores? `rouge2` measures the overlap between bigrams (think the overlap of pairs of words), while `rougeL` and `rougeLsum` measure the longest matching sequences of words by looking for the longest common substrings in the generated and reference summaries. The "sum" in `rougeLsum` refers to the fact that this metric is computed over a whole summary, while `rougeL` is computed as the average over individual sentences.

> [!TIP]
> ✏️ **Try it out!** Create your own example of a generated and reference summary and see if the resulting ROUGE scores agree with a manual calculation based on the formulas for precision and recall. For bonus points, split the text into bigrams and compare the precision and recall for the `rouge2` metric.

We'll use these ROUGE scores to track the performance of our model, but before doing that let's do something every good NLP practitioner should do: create a strong, yet simple baseline!

### Creating a strong baseline[[creating-a-strong-baseline]]

A common baseline for text summarization is to simply take the first three sentences of an article, often called the _lead-3_ baseline. We could use full stops to track the sentence boundaries, but this will fail on acronyms like "U.S." or "U.N." -- so instead we'll use the `nltk` library, which includes a better algorithm to handle these cases. You can install the package using `pip` as follows:

```python
!pip install nltk
```

and then download the punctuation rules:

```python
import nltk

nltk.download("punkt")
```

Next, we import the sentence tokenizer from `nltk` and create a simple function to extract the first three sentences in a review. The convention in text summarization is to separate each summary with a newline, so let's also include this and test it on a training example:

```python
from nltk.tokenize import sent_tokenize


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


print(three_sentence_summary(books_dataset["train"][1]["review_body"]))
```

```python out
'I grew up reading Koontz, and years ago, I stopped,convinced i had "outgrown" him.'

"Still,when a friend was looking for something suspenseful too read, I suggested Koontz."
"She found Strangers."
```

This seems to work, so let's now implement a function that extracts these "summaries" from a dataset and computes the ROUGE scores for the baseline:

```python
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])
```

We can then use this function to compute the ROUGE scores over the validation set and prettify them a bit using Pandas:

```python
import pandas as pd

score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
rouge_dict
```

```python out
{"rouge1": 16.74, "rouge2": 8.83, "rougeL": 15.6, "rougeLsum": 15.96}
```

We can see that the `rouge2` score is significantly lower than the rest; this likely reflects the fact that review titles are typically concise and so the lead-3 baseline is too verbose. Now that we have a good baseline to work from, let's turn our attention toward fine-tuning mT5!

{#if fw === 'pt'}

## Fine-tuning mT5 with the `Trainer` API[[fine-tuning-mt5-with-the-trainer-api]]

Fine-tuning a model for summarization is very similar to the other tasks we've covered in this chapter. The first thing we need to do is load the pretrained model from the `mt5-small` checkpoint. Since summarization is a sequence-to-sequence task, we can load the model with the `AutoModelForSeq2SeqLM` class, which will automatically download and cache the weights:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

{:else}

## Fine-tuning mT5 with Keras[[fine-tuning-mt5-with-keras]]

Fine-tuning a model for summarization is very similar to the other tasks we've covered in this chapter. The first thing we need to do is load the pretrained model from the `mt5-small` checkpoint. Since summarization is a sequence-to-sequence task, we can load the model with the `TFAutoModelForSeq2SeqLM` class, which will automatically download and cache the weights:

```python
from transformers import TFAutoModelForSeq2SeqLM

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

{/if}

> [!TIP]
> 💡 If you're wondering why you don't see any warnings about fine-tuning the model on a downstream task, that's because for sequence-to-sequence tasks we keep all the weights of the network. Compare this to our text classification model in [Chapter 3](/course/chapter3), where the head of the pretrained model was replaced with a randomly initialized network.

The next thing we need to do is log in to the Hugging Face Hub. If you're running this code in a notebook, you can do so with the following utility function:

```python
from huggingface_hub import notebook_login

notebook_login()
```

which will display a widget where you can enter your credentials. Alternatively, you can run this command in your terminal and log in there:

```
huggingface-cli login
```

{#if fw === 'pt'}

We'll need to generate summaries in order to compute ROUGE scores during training. Fortunately, 🤗 Transformers provides dedicated `Seq2SeqTrainingArguments` and `Seq2SeqTrainer` classes that can do this for us automatically! To see how this works, let's first define the hyperparameters and other arguments for our experiments:

```python
from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)
```

Here, the `predict_with_generate` argument has been set to indicate that we should generate summaries during evaluation so that we can compute ROUGE scores for each epoch. As discussed in [Chapter 1](/course/chapter1), the decoder performs inference by predicting tokens one by one, and this is implemented by the model's `generate()` method. Setting `predict_with_generate=True` tells the `Seq2SeqTrainer` to use that method for evaluation. We've also adjusted some of the default hyperparameters, like the learning rate, number of epochs, and weight decay, and we've set the `save_total_limit` option to only save up to 3 checkpoints during training -- this is because even the "small" version of mT5 uses around a GB of hard drive space, and we can save a bit of room by limiting the number of copies we save.

The `push_to_hub=True` argument will allow us to push the model to the Hub after training; you'll find the repository under your user profile in the location defined by `output_dir`. Note that you can specify the name of the repository you want to push to with the `hub_model_id` argument (in particular, you will have to use this argument to push to an organization). For instance, when we pushed the model to the [`huggingface-course` organization](https://huggingface.co/huggingface-course), we added `hub_model_id="huggingface-course/mt5-finetuned-amazon-en-es"` to `Seq2SeqTrainingArguments`.

The next thing we need to do is provide the trainer with a `compute_metrics()` function so that we can evaluate our model during training. For summarization this is a bit more involved than simply calling `rouge_score.compute()` on the model's predictions, since we need to _decode_ the outputs and labels into text before we can compute the ROUGE scores. The following function does exactly that, and also makes use of the `sent_tokenize()` function from `nltk` to separate the summary sentences with newlines:

```python
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
```

{/if}

Next, we need to define a data collator for our sequence-to-sequence task. Since mT5 is an encoder-decoder Transformer model, one subtlety with preparing our batches is that during decoding we need to shift the labels to the right by one. This is required to ensure that the decoder only sees the previous ground truth labels and not the current or future ones, which would be easy for the model to memorize. This is similar to how masked self-attention is applied to the inputs in a task like [causal language modeling](/course/chapter7/6).

Luckily, 🤗 Transformers provides a `DataCollatorForSeq2Seq` collator that will dynamically pad the inputs and the labels for us. To instantiate this collator, we simply need to provide the `tokenizer` and `model`:

{#if fw === 'pt'}

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

{:else}

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
```

{/if}

Let's see what this collator produces when fed a small batch of examples. First, we need to remove the columns with strings because the collator won't know how to pad these elements:

```python
tokenized_datasets = tokenized_datasets.remove_columns(books_dataset["train"].column_names)
```

Since the collator expects a list of `dict`s, where each `dict` represents a single example in the dataset, we also need to wrangle the data into the expected format before passing it to the data collator:

```python
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)
```

```python out
{
    "attention_mask": tensor(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ]
    ),
    "input_ids": tensor(
        [
            [
                1494,
                259,
                8622,
                390,
                259,
                262,
                2316,
                3435,
                955,
                772,
                281,
                772,
                1617,
                263,
                305,
                14701,
                260,
                1385,
                3031,
                259,
                24146,
                332,
                1037,
                259,
                43906,
                305,
                336,
                260,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                259,
                27531,
                13483,
                259,
                7505,
                260,
                112240,
                15192,
                305,
                53198,
                276,
                259,
                74060,
                263,
                260,
                459,
                25640,
                776,
                2119,
                336,
                259,
                2220,
                259,
                18896,
                288,
                4906,
                288,
                1037,
                3931,
                260,
                7083,
                101476,
                1143,
                260,
                1,
            ],
        ]
    ),
    "labels": tensor([[7483, 259, 2364, 15695, 1, -100], [259, 27531, 13483, 259, 7505, 1]]),
    "decoder_input_ids": tensor(
        [[0, 7483, 259, 2364, 15695, 1], [0, 259, 27531, 13483, 259, 7505]]
    ),
}
```

The main thing to notice here is that the first example is longer than the second one, so the `input_ids` and `attention_mask` of the second example have been padded on the right with a `[PAD]` token (whose ID is `0`). Similarly, we can see that the `labels` have been padded with `-100`s, to make sure the padding tokens are ignored by the loss function. And finally, we can see a new `decoder_input_ids` which has shifted the labels to the right by inserting a `[PAD]` token in the first entry.

{#if fw === 'pt'}

We finally have all the ingredients we need to train with! We now simply need to instantiate the trainer with the standard arguments:

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

and launch our training run:

```python
trainer.train()
```

During training, you should see the training loss decrease and the ROUGE scores increase with each epoch. Once the training is complete, you can see the final ROUGE scores by running `Trainer.evaluate()`:

```python
trainer.evaluate()
```

```python out
{
    "eval_loss": 3.028524398803711,
    "eval_rouge1": 16.9728,
    "eval_rouge2": 8.2969,
    "eval_rougeL": 16.8366,
    "eval_rougeLsum": 16.851,
    "eval_gen_len": 10.1597,
    "eval_runtime": 6.1054,
    "eval_samples_per_second": 38.982,
    "eval_steps_per_second": 4.914,
}
```

From the scores we can see that our model has handily outperformed our lead-3 baseline -- nice! The final thing to do is push the model weights to the Hub, as follows:

```
trainer.push_to_hub(commit_message="Training complete", tags="summarization")
```

```python out
"https://huggingface.co/huggingface-course/mt5-finetuned-amazon-en-es/commit/aa0536b829b28e73e1e4b94b8a5aacec420d40e0"
```

This will save the checkpoint and configuration files to `output_dir`, before uploading all the files to the Hub. By specifying the `tags` argument, we also ensure that the widget on the Hub will be one for a summarization pipeline instead of the default text generation one associated with the mT5 architecture (for more information about model tags, see the [🤗 Hub documentation](https://huggingface.co/docs/hub/main#how-is-a-models-type-of-inference-api-and-widget-determined)). The output from `trainer.push_to_hub()` is a URL to the Git commit hash, so you can easily see the changes that were made to the model repository!

To wrap up this section, let's take a look at how we can also fine-tune mT5 using the low-level features provided by 🤗 Accelerate.

{:else}

We're almost ready to train! We just need to convert our datasets to `tf.data.Dataset`s using the data collator we defined above, and then `compile()` and `fit()` the model. First, the datasets:

```python
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=8,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=8,
)
```

Now, we define our training hyperparameters and compile:

```python
from transformers import create_optimizer
import tensorflow as tf

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_epochs = 8
num_train_steps = len(tf_train_dataset) * num_train_epochs
model_name = model_checkpoint.split("/")[-1]

optimizer, schedule = create_optimizer(
    init_lr=5.6e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

And finally, we fit the model. We use a `PushToHubCallback` to save the model to the Hub after each epoch, which will allow us to use it for inference later:

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir=f"{model_name}-finetuned-amazon-en-es", tokenizer=tokenizer)

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback], epochs=8)
```

We got some loss values during training, but really we'd like to see the ROUGE metrics we computed earlier. To get those metrics, we'll need to generate outputs from the model and convert them to strings. Let's build some lists of labels and predictions for the ROUGE metric to compare (note that if you get import errors for this section, you may need to`!pip install tqdm`). We're also going to use a trick that dramatically increases performance - compiling our generation code with [XLA](https://www.tensorflow.org/xla), TensorFlow's accelerated linear algebra compiler. XLA applies various optimizations to the model's computation graph, and results in significant improvements to speed and memory usage. As described in the Hugging Face [blog](https://huggingface.co/blog/tf-xla-generate), XLA works best when our input shapes don't vary too much. To handle this, we'll pad our inputs to multiples of 128, and make a new dataset with the padding collator, and then we'll apply the `@tf.function(jit_compile=True)` decorator to our generation function, which marks the whole function for compilation with XLA. 

```python
from tqdm import tqdm
import numpy as np

generation_data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=320
)

tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
    drop_remainder=True,
)


@tf.function(jit_compile=True)
def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=32,
    )


all_preds = []
all_labels = []
for batch, labels in tqdm(tf_generate_dataset):
    predictions = generate_with_xla(batch)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = labels.numpy()
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    all_preds.extend(decoded_preds)
    all_labels.extend(decoded_labels)
```

Once we have our lists of label and prediction strings, computing the ROUGE score is easy:

```python
result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
{k: round(v, 4) for k, v in result.items()}
```

```
{'rouge1': 31.4815, 'rouge2': 25.4386, 'rougeL': 31.4815, 'rougeLsum': 31.4815}
```


{/if}

{#if fw === 'pt'}

## Fine-tuning mT5 with 🤗 Accelerate[[fine-tuning-mt5-with-accelerate]]

Fine-tuning our model with 🤗 Accelerate is very similar to the text classification example we encountered in [Chapter 3](/course/chapter3). The main differences will be the need to explicitly generate our summaries during training and define how we compute the ROUGE scores (recall that the `Seq2SeqTrainer` took care of the generation for us). Let's take a look how we can implement these two requirements within 🤗 Accelerate!

### Preparing everything for training[[preparing-everything-for-training]]

The first thing we need to do is create a `DataLoader` for each of our splits. Since the PyTorch dataloaders expect batches of tensors, we need to set the format to `"torch"` in our datasets:

```python
tokenized_datasets.set_format("torch")
```

Now that we've got datasets consisting of just tensors, the next thing to do is instantiate the `DataCollatorForSeq2Seq` again. For this we need to provide a fresh version of the model, so let's load it again from our cache:

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

We can then instantiate the data collator and use this to define our dataloaders:

```python
from torch.utils.data import DataLoader

batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size
)
```

The next thing to do is define the optimizer we want to use. As in our other examples, we'll use `AdamW`, which works well for most problems:

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Finally, we feed our model, optimizer, and dataloaders to the `accelerator.prepare()` method:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

> [!TIP]
> 🚨 If you're training on a TPU, you'll need to move all the code above into a dedicated training function. See [Chapter 3](/course/chapter3) for more details.

Now that we've prepared our objects, there are three remaining things to do:

* Define the learning rate schedule.
* Implement a function to post-process the summaries for evaluation.
* Create a repository on the Hub that we can push our model to.

For the learning rate schedule, we'll use the standard linear one from previous sections:

```python
from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

For post-processing, we need a function that splits the generated summaries into sentences that are separated by newlines. This is the format the ROUGE metric expects, and we can achieve this with the following snippet of code:

```python
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
```

This should look familiar to you if you recall how we defined the `compute_metrics()` function of the `Seq2SeqTrainer`. 

Finally, we need to create a model repository on the Hugging Face Hub. For this, we can use the appropriately titled 🤗 Hub library. We just need to define a name for our repository, and the library has a utility function to combine the repository ID with the user profile:

```python
from huggingface_hub import get_full_repo_name

model_name = "test-bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
"lewtun/mt5-finetuned-amazon-en-es-accelerate"
```

Now we can use this repository name to clone a local version to our results directory that will store the training artifacts:

```python
from huggingface_hub import Repository

output_dir = "results-mt5-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

This will allow us to push the artifacts back to the Hub by calling the `repo.push_to_hub()` method during training! Let's now wrap up our analysis by writing out the training loop.

### Training loop[[training-loop]]

The training loop for summarization is quite similar to the other 🤗 Accelerate examples that we've encountered and is roughly split into four main steps:

1. Train the model by iterating over all the examples in `train_dataloader` for each epoch.
2. Generate model summaries at the end of each epoch, by first generating the tokens and then decoding them (and the reference summaries) into text.
3. Compute the ROUGE scores using the same techniques we saw earlier.
4. Save the checkpoints and push everything to the Hub. Here we rely on the nifty `blocking=False` argument of the `Repository` object so that we can push the checkpoints per epoch _asynchronously_. This allows us to continue training without having to wait for the somewhat slow upload associated with a GB-sized model!

These steps can be seen in the following block of code:

```python
from tqdm.auto import tqdm
import torch
import numpy as np

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = rouge_score.compute()
    # Extract the median ROUGE scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
```

```python out
Epoch 0: {'rouge1': 5.6351, 'rouge2': 1.1625, 'rougeL': 5.4866, 'rougeLsum': 5.5005}
Epoch 1: {'rouge1': 9.8646, 'rouge2': 3.4106, 'rougeL': 9.9439, 'rougeLsum': 9.9306}
Epoch 2: {'rouge1': 11.0872, 'rouge2': 3.3273, 'rougeL': 11.0508, 'rougeLsum': 10.9468}
Epoch 3: {'rouge1': 11.8587, 'rouge2': 4.8167, 'rougeL': 11.7986, 'rougeLsum': 11.7518}
Epoch 4: {'rouge1': 12.9842, 'rouge2': 5.5887, 'rougeL': 12.7546, 'rougeLsum': 12.7029}
Epoch 5: {'rouge1': 13.4628, 'rouge2': 6.4598, 'rougeL': 13.312, 'rougeLsum': 13.2913}
Epoch 6: {'rouge1': 12.9131, 'rouge2': 5.8914, 'rougeL': 12.6896, 'rougeLsum': 12.5701}
Epoch 7: {'rouge1': 13.3079, 'rouge2': 6.2994, 'rougeL': 13.1536, 'rougeLsum': 13.1194}
Epoch 8: {'rouge1': 13.96, 'rouge2': 6.5998, 'rougeL': 13.9123, 'rougeLsum': 13.7744}
Epoch 9: {'rouge1': 14.1192, 'rouge2': 7.0059, 'rougeL': 14.1172, 'rougeLsum': 13.9509}
```

And that's it! Once you run this, you'll have a model and results that are pretty similar to the ones we obtained with the `Trainer`.

{/if}

## Using your fine-tuned model[[using-your-fine-tuned-model]]

Once you've pushed the model to the Hub, you can play with it either via the inference widget or with a `pipeline` object, as follows:

```python
from transformers import pipeline

hub_model_id = "huggingface-course/mt5-small-finetuned-amazon-en-es"
summarizer = pipeline("summarization", model=hub_model_id)
```

We can feed some examples from the test set (which the model has not seen) to our pipeline to get a feel for the quality of the summaries. First let's implement a simple function to show the review, title, and generated summary together:

```python
def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")
```

Let's take a look at one of the English examples we get:

```python
print_summary(100)
```

```python out
">>> Review: Nothing special at all about this product... the book is too small and stiff and hard to write in. The huge sticker on the back doesn’t come off and looks super tacky. I would not purchase this again. I could have just bought a journal from the dollar store and it would be basically the same thing. It’s also really expensive for what it is."

">>> Title: Not impressed at all... buy something else"

">>> Summary: Nothing special at all about this product"
```

This is not too bad! We can see that our model has actually been able to perform _abstractive_ summarization by augmenting parts of the review with new words. And perhaps the coolest aspect of our model is that it is bilingual, so we can also generate summaries of Spanish reviews:

```python
print_summary(0)
```

```python out
">>> Review: Es una trilogia que se hace muy facil de leer. Me ha gustado, no me esperaba el final para nada"

">>> Title: Buena literatura para adolescentes"

">>> Summary: Muy facil de leer"
```

The summary translates into "Very easy to read" in English, which we can see in this case was extracted directly from the review. Nevertheless, this shows the versatility of the mT5 model and has given you a taste of what it's like to deal with a multilingual corpus!

Next, we'll turn our attention to a slightly more complex task: training a language model from scratch.


---

<!-- Section 7.6 -->

<FrameworkSwitchCourse {fw} />

# Training a causal language model from scratch[[training-a-causal-language-model-from-scratch]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_tf.ipynb"},
]} />

{/if}

Up until now, we've mostly been using pretrained models and fine-tuning them for new use cases by reusing the weights from pretraining. As we saw in [Chapter 1](/course/chapter1), this is commonly referred to as _transfer learning_, and it's a very successful strategy for applying Transformer models to most real-world use cases where labeled data is sparse. In this chapter, we'll take a different approach and train a completely new model from scratch. This is a good approach to take if you have a lot of data and it is very different from the pretraining data used for the available models. However, it also requires considerably more compute resources to pretrain a language model than just to fine-tune an existing one. Examples where it can make sense to train a new model include for datasets consisting of musical notes, molecular sequences such as DNA, or programming languages. The latter have recently gained traction thanks to tools such as TabNine and GitHub's Copilot, powered by OpenAI's Codex model, that can generate long sequences of code. This task of text generation is best addressed with auto-regressive or causal language models such as GPT-2.

In this section we will build a scaled-down version of a code generation model: we'll focus on one-line completions instead of full functions or classes, using a subset of Python code. When working with data in Python you are in frequent contact with the Python data science stack, consisting of the `matplotlib`, `seaborn`, `pandas`, and `scikit-learn` libraries. When using those frameworks it's common to need to look up specific commands, so it would be nice if we could use a model to complete these calls for us.

<Youtube id="Vpjb1lu0MDk"/>

In [Chapter 6](/course/chapter6) we created an efficient tokenizer to process Python source code, but what we still need is a large-scale dataset to pretrain a model on. Here, we'll apply our tokenizer to a corpus of Python code derived from GitHub repositories. We will then use the `Trainer` API and 🤗 Accelerate to train the model. Let's get to it!

<iframe src="https://course-demos-codeparrot-ds.hf.space" frameBorder="0" height="300" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

This is actually showcasing the model that was trained and uploaded to the Hub using the code shown in this section. You can find it [here](https://huggingface.co/huggingface-course/codeparrot-ds?text=plt.imshow%28). Note that since there is some randomization happening in the text generation, you will probably get a slightly different result.
 
## Gathering the data[[gathering-the-data]]

Python code is abundantly available from code repositories such as GitHub, which we can use to create a dataset by scraping for every Python repository. This was the approach taken in the [Transformers textbook](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/) to pretrain a large GPT-2 model. Using a GitHub dump of about 180 GB containing roughly 20 million Python files called `codeparrot`, the authors built a dataset that they then shared on the [Hugging Face Hub](https://huggingface.co/datasets/transformersbook/codeparrot).

However, training on the full corpus is time- and compute-consuming, and we only need the subset of the dataset concerned with the Python data science stack. So, let's start by filtering the `codeparrot` dataset for all files that include any of the libraries in this stack. Because of the dataset's size, we want to avoid downloading it; instead, we'll use the streaming feature to filter it on the fly. To help us filter the code samples using the libraries we mentioned earlier, we'll use the following function:

```py
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False
```

Let's test it on two examples:

```py
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"

print(any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters))
```

```python out
False True
```

We can use this to create a function that will stream the dataset and filter the elements we want:

```py
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content']) / total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)
```

Then we can simply apply this function to the streaming dataset:

```py
# This cell will take a very long time to execute, so you should skip it and go to
# the next one!
from datasets import load_dataset

split = "train"  # "valid"
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
filtered_data = filter_streaming_dataset(data, filters)
```

```python out
3.26% of data after filtering.
```

This leaves us with about 3% of the original dataset, which is still quite sizable -- the resulting dataset is 6 GB and consists of 600,000 Python scripts!

Filtering the full dataset can take 2-3h depending on your machine and bandwidth. If you don't want to go through this lengthy process yourself, we provide the filtered dataset on the Hub for you to download:

```py
from datasets import load_dataset, DatasetDict

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
        num_rows: 606720
    })
    valid: Dataset({
        features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
        num_rows: 3322
    })
})
```

> [!TIP]
> Pretraining the language model will take a while. We suggest that you first run the training loop on a sample of the data by uncommenting the two partial lines above, and make sure that the training successfully completes and the models are stored. Nothing is more frustrating than a training run failing at the last step because you forgot to create a folder or because there's a typo at the end of the training loop!

Let's look at an example from the dataset. We'll just show the first 200 characters of each field:

```py
for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")
```

```python out
"REPO_NAME: kmike/scikit-learn"

"PATH: sklearn/utils/__init__.py"
"COPIES: 3"
"SIZE: 10094"
'''CONTENT: """
The :mod:`sklearn.utils` module includes various utilites.
"""

from collections import Sequence

import numpy as np
from scipy.sparse import issparse
import warnings

from .murmurhash import murm
LICENSE: bsd-3-clause'''
```

We can see that the `content` field contains the code that we want our model to train on. Now that we have a dataset, we need to prepare the texts so they're in a format suitable for pretraining.

## Preparing the dataset[[preparing-the-dataset]]

<Youtube id="ma1TrR7gE7I"/>

The first step will be to tokenize the data, so we can use it for training. Since our goal is to mainly autocomplete short function calls, we can keep the context size relatively small. This has the benefit that we can train the model much faster and it requires significantly less memory. If it is important for your application to have more context (for example, if you want the model to write unit tests based on a file with the function definition), make sure you increase that number, but also keep in mind that this comes with a greater GPU memory footprint. For now, let's fix the context size at 128 tokens, as opposed to the 1,024 or 2,048 used in GPT-2 or GPT-3, respectively.

Most documents contain many more than 128 tokens, so simply truncating the inputs to the maximum length would eliminate a large fraction of our dataset. Instead, we'll use the `return_overflowing_tokens` option to tokenize the whole input and split it into several chunks, as we did in [Chapter 6](/course/chapter6/4). We'll also use the `return_length` option to return the length of each created chunk automatically. Often the last chunk will be smaller than the context size, and we'll get rid of these pieces to avoid padding issues; we don't really need them as we have plenty of data anyway.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/chunking_texts.svg" alt="Chunking a large texts in several pieces."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/chunking_texts-dark.svg" alt="Chunking a large texts in several pieces."/>
</div>

Let's see exactly how this works by looking at the first two examples:

```py
from transformers import AutoTokenizer

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
```

```python out
Input IDs length: 34
Input chunk lengths: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 117, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 41]
Chunk mapping: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

We can see that we get 34 segments in total from those two examples. Looking at the chunk lengths, we can see that the chunks at the ends of both documents have less than 128 tokens (117 and 41, respectively). These represent just a small fraction of the total chunks that we have, so we can safely throw them away. With the `overflow_to_sample_mapping` field, we can also reconstruct which chunks belonged to which input samples.

With this operation we're using a handy feature of the `Dataset.map()` function in 🤗 Datasets, which is that it does not require one-to-one maps; as we saw in [section 3](/course/chapter7/3), we can create batches with more or fewer elements than the input batch. This is useful when doing operations like data augmentation or data filtering that change the number of elements. In our case, when tokenizing each element into chunks of the specified context size, we create many samples from each document. We just need to make sure to delete the existing columns, since they have a conflicting size. If we wanted to keep them, we could repeat them appropriately and return them within the `Dataset.map()` call:

```py
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['input_ids'],
        num_rows: 16702061
    })
    valid: Dataset({
        features: ['input_ids'],
        num_rows: 93164
    })
})
```

We now have 16.7 million examples with 128 tokens each, which corresponds to about 2.1 billion tokens in total. For reference, OpenAI's GPT-3 and Codex models are trained on 300 and 100 billion tokens, respectively, where the Codex models are initialized from the GPT-3 checkpoints. Our goal in this section is not to compete with these models, which can generate long, coherent texts, but to create a scaled-down version providing a quick autocomplete function for data scientists.

Now that we have the dataset ready, let's set up the model!

> [!TIP]
> ✏️ **Try it out!** Getting rid of all the chunks that are smaller than the context size wasn't a big issue here because we're using small context windows. As you increase the context size (or if you have a corpus of short documents), the fraction of chunks that are thrown away will also grow. A more efficient way to prepare the data is to join all the tokenized samples in a batch with an `eos_token_id` token in between, and then perform the chunking on the concatenated sequences. As an exercise, modify the `tokenize()` function to make use of that approach. Note that you'll want to set `truncation=False` and remove the other arguments from the tokenizer to get the full sequence of token IDs.


## Initializing a new model[[initializing-a-new-model]]

Our first step is to freshly initialize a GPT-2 model. We'll use the same configuration for our model as for the small GPT-2 model, so we load the pretrained configuration, make sure that the tokenizer size matches the model vocabulary size and pass the `bos` and `eos` (beginning and end of sequence) token IDs:

{#if fw === 'pt'}

```py
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

With that configuration, we can load a new model. Note that this is the first time we don't use the `from_pretrained()` function, since we're actually initializing a model ourself:

```py
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size / 1000**2:.1f}M parameters")
```

```python out
GPT-2 size: 124.2M parameters
```

{:else}

```py
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

With that configuration, we can load a new model. Note that this is the first time we don't use the `from_pretrained()` function, since we're actually initializing a model ourself:

```py
model = TFGPT2LMHeadModel(config)
model(model.dummy_inputs)  # Builds the model
model.summary()
```

```python out
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
transformer (TFGPT2MainLayer multiple                  124242432 
=================================================================
Total params: 124,242,432
Trainable params: 124,242,432
Non-trainable params: 0
_________________________________________________________________
```

{/if}

Our model has 124M parameters that we'll have to tune. Before we can start training, we need to set up a data collator that will take care of creating the batches. We can use the `DataCollatorForLanguageModeling` collator, which is designed specifically for language modeling (as the name subtly suggests). Besides stacking and padding batches, it also takes care of creating the language model labels -- in causal language modeling the inputs serve as labels too (just shifted by one element), and this data collator creates them on the fly during training so we don't need to duplicate the `input_ids`.

Note that `DataCollatorForLanguageModeling` supports both masked language modeling (MLM) and causal language modeling (CLM). By default it prepares data for MLM, but we can switch to CLM by setting the argument `mlm=False`:

{#if fw === 'pt'}

```py
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

{:else}

```py
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")
```

{/if}

Let's have a look at an example:

```py
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
```

{#if fw === 'pt'}

```python out
input_ids shape: torch.Size([5, 128])
attention_mask shape: torch.Size([5, 128])
labels shape: torch.Size([5, 128])
```

{:else}

```python out
input_ids shape: (5, 128)
attention_mask shape: (5, 128)
labels shape: (5, 128)
```

{/if}

We can see that the examples have been stacked and all the tensors have the same shape.

{#if fw === 'tf'}

Now we can use the `prepare_tf_dataset()` method to convert our datasets to TensorFlow datasets with the data collator we created above:

```python
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["valid"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)
```

{/if}

> [!WARNING]
> ⚠️ Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.


Now we have everything in place to actually train our model -- that wasn't so much work after all! Before we start training we should log in to Hugging Face. If you're working in a notebook, you can do so with the following utility function:

```python
from huggingface_hub import notebook_login

notebook_login()
```

This will display a widget where you can enter your Hugging Face login credentials.

If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

{#if fw === 'pt'}

All that's left to do is configure the training arguments and fire up the `Trainer`. We'll use a cosine learning rate schedule with some warmup and an effective batch size of 256 (`per_device_train_batch_size` * `gradient_accumulation_steps`). Gradient accumulation is used when a single batch does not fit into memory, and incrementally builds up the gradient through several forward/backward passes. We'll see this in action when we create the training loop with 🤗 Accelerate.

```py
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
```

Now we can just start the `Trainer` and wait for training to finish. Depending on whether you run it on the full or a subset of the training set this will take 20 or 2 hours, respectively, so grab a few coffees and a good book to read!

```py
trainer.train()
```

After training completes, we can push the model and tokenizer to the Hub:

```py
trainer.push_to_hub()
```

{:else}

All that's left to do is configure the training hyperparameters and call `compile()` and `fit()`. We'll use a learning rate schedule with some warmup to improve the stability of training:

```py
from transformers import create_optimizer
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Now we can just call `model.fit()` and wait for training to finish. Depending on whether you run it on the full or a subset of the training set this will take 20 or 2 hours, respectively, so grab a few coffees and a good book to read! After training completes we can push the model and tokenizer to the Hub:

```py
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="codeparrot-ds", tokenizer=tokenizer)

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback])
```

{/if}

> [!TIP]
> ✏️ **Try it out!** It only took us about 30 lines of code in addition to the `TrainingArguments` to get from raw texts to training GPT-2. Try it out with your own dataset and see if you can get good results!

> [!TIP]
> {#if fw === 'pt'}
>
> 💡 If you have access to a machine with multiple GPUs, try to run the code there. The `Trainer` automatically manages multiple machines, and this can speed up training tremendously.
>
> {:else}
>
> 💡 If you have access to a machine with multiple GPUs, you can try using a `MirroredStrategy` context to substantially speed up training. You'll need to create a `tf.distribute.MirroredStrategy` object, and make sure that any `to_tf_dataset()` or `prepare_tf_dataset()` methods as well as model creation and the call to `fit()` are all run in its `scope()` context. You can see documentation on this [here](https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_keras_modelfit).
>
> {/if}

## Code generation with a pipeline[[code-generation-with-a-pipeline]]

Now is the moment of truth: let's see how well the trained model actually works! We can see in the logs that the loss went down steadily, but to put the model to the test let's take a look at how well it works on some prompts. To do that we'll wrap the model in a text generation `pipeline`, and we'll put it on the GPU for fast generations if there is one available:

{#if fw === 'pt'}

```py
import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline("text-generation", model="huggingface-course/codeparrot-ds", device=device)
```

{:else}

```py
from transformers import pipeline

course_model = TFGPT2LMHeadModel.from_pretrained("huggingface-course/codeparrot-ds")
course_tokenizer = AutoTokenizer.from_pretrained("huggingface-course/codeparrot-ds")
pipe = pipeline("text-generation", model=course_model, tokenizer=course_tokenizer, device=0)
```

{/if}

Let's start with the simple task of creating a scatter plot:

```py
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
plt.scatter(x, y)

# create scatter
```

The result looks correct. Does it also work for a `pandas` operation? Let's see if we can create a `DataFrame` from two arrays:

```py
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
df = pd.DataFrame({'x': x, 'y': y})
df.insert(0,'x', x)
for
```

Nice, that's the correct answer -- although it then inserts the column `x` again. Since the number of generated tokens is limited, the following `for` loop is cut off. Let's see if we can do something a bit more complex and have the model help us use the `groupby` operation: 

```py
txt = """\
# dataframe with profession, income and name
df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

# calculate the mean income per profession
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# dataframe with profession, income and name
df = pd.DataFrame({"profession": x, "income": y, "name": z})

# calculate the mean income per profession
profession = df.groupby(["profession"]).mean()

# compute the
```

Not bad; that's the right way to do it. Finally, let's see if we can also use it for `scikit-learn` and set up a Random Forest model:

```py
txt = """
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
```

```python out
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
rf = RandomForestRegressor(n_estimators=300, random_state=random_state, max_depth=3)
rf.fit(X, y)
rf
```

{#if fw === 'tf'}

Looking at these few examples, it seems that the model has learned some of the syntax of the Python data science stack. Of course, we would need to evaluate the model more thoroughly before deploying it in the real world, but this is still an impressive prototype.

{:else}

Looking at these few examples, it seems that the model has learned some of the syntax of the Python data science stack (of course, we would need to evaluate it more thoroughly before deploying the model in the real world). Sometimes it requires more customization of the model training to achieve the necessary performance for a given use case, however. For example, what if we would like to dynamically update the batch size or have a conditional training loop that skips bad examples on the fly? One option would be to subclass the `Trainer` and add the necessary changes, but sometimes it's simpler to write the training loop from scratch. That's where 🤗 Accelerate comes in.

{/if}

{#if fw === 'pt'}

## Training with 🤗 Accelerate[[training-with-accelerate]]

We've seen how to train a model with the `Trainer`, which can allow for some customization. However, sometimes we want full control over the training loop, or we want to make some exotic changes. In this case 🤗 Accelerate is a great choice, and in this section we'll go through the steps to use it to train our model. To make things more interesting, we'll also add a twist to the training loop.

<Youtube id="Hm8_PgVTFuc"/>

Since we are mainly interested in sensible autocompletion for the the data science libraries, it makes sense to give more weight to training samples that make more use of these libraries. We can easily identify these examples through the use of keywords such as `plt`, `pd`, `sk`, `fit`, and `predict`, which are the most frequent import names for `matplotlib.pyplot`, `pandas`, and `sklearn` as well as the fit/predict pattern of the latter. If these are each represented as a single token, we can easily check if they occur in the input sequence. Tokens can have a whitespace prefix, so we'll also check for those versions in the tokenizer vocabulary. To verify that it works, we'll add one test token which should be split into multiple tokens:

```py
keytoken_ids = []
for keyword in [
    "plt",
    "pd",
    "sk",
    "fit",
    "predict",
    " plt",
    " pd",
    " sk",
    " fit",
    " predict",
    "testtest",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")
```

```python out
"Keyword has not single token: testtest"
```

Great, that seems to work nicely! We can now write a custom loss function that takes the input sequence, the logits, and the key tokens we just selected as inputs. First we need to align the logits and inputs: the input sequence shifted by one to the right forms the labels, since the next token is the label for the current token. We can achieve this by starting the labels from the second token of the input sequence, since the model does not make a prediction for the first token anyway. Then we cut off the last logit, as we don't have a label for the token that follows the full input sequence. With that we can compute the loss per sample and count the occurrences of all keywords in each sample. Finally, we calculate the weighted average over all samples using the occurrences as weights. Since we don't want to throw away all the samples that have no keywords, we add 1 to the weights:

```py
from torch.nn import CrossEntropyLoss
import torch


def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss
```

Before we can start training with this awesome new loss function, we need to prepare a few things:

- We need dataloaders to load the data in batches.
- We need to set up weight decay parameters.
- From time to time we want to evaluate, so it makes sense to wrap the evaluation code in a function.

Let's start with the dataloaders. We only need to set the dataset's format to `"torch"`, and then we can pass it to a PyTorch `DataLoader` with the appropriate batch size:

```py
from torch.utils.data.dataloader import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)
```

Next, we group the parameters so that the optimizer knows which ones will get an additional weight decay. Usually, all bias and LayerNorm weights terms are exempt from this; here's how we can do this:

```py
weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
```

Since we want to evaluate the model regularly on the validation set during training, let's write a function for that as well. It just runs through the evaluation dataloader and gathers all the losses across processes:

```py
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
```

With the `evaluate()` function we can report loss and [perplexity](/course/chapter7/3) at regular intervals. Next, we redefine our model to make sure we train from scratch again:

```py
model = GPT2LMHeadModel(config)
```

We can then define our optimizer, using the function from before to split the parameters for weight decay:

```py
from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)
```

Now let's prepare the model, optimizer, and dataloaders so we can start training:

```py
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

> [!TIP]
> 🚨 If you're training on a TPU, you'll need to move all the code starting at the cell above into a dedicated training function. See [Chapter 3](/course/chapter3) for more details.

Now that we have sent our `train_dataloader` to `accelerator.prepare()`, we can use its length to compute the number of training steps. Remember that we should always do this after preparing the dataloader, as that method will change its length. We use a classic linear schedule from the learning rate to 0:

```py
from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)
```

Lastly, to push our model to the Hub, we will need to create a `Repository` object in a working folder. First log in to the Hugging Face Hub, if you aren't logged in already. We'll determine the repository name from the model ID we want to give our model (feel free to replace the `repo_name` with your own choice; it just needs to contain your username, which is what the function `get_full_repo_name()` does):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "codeparrot-ds-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
"sgugger/codeparrot-ds-accelerate"
```

Then we can clone that repository in a local folder. If it already exists, this local folder should be an existing clone of the repository we are working with:

```py
output_dir = "codeparrot-ds-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

We can now upload anything we save in `output_dir` by calling the `repo.push_to_hub()` method. This will help us upload the intermediate models at the end of each epoch.

Before we train, let's run a quick test to see if the evaluation function works properly:

```py
evaluate()
```

```python out
(10.934126853942871, 56057.14453125)
```

Those are very high values for loss and perplexity, but that's not surprising as we haven't trained the model yet. With that, we have everything prepared to write the core part of the training script: the training loop. In the training loop we iterate over the dataloader and pass the batches to the model. With the logits, we can then evaluate our custom loss function. We scale the loss by the number of gradient accumulation steps so as not to create larger losses when aggregating more steps. Before we optimize, we also clip the gradients for better convergence. Finally, every few steps we evaluate the model on the evaluation set with our new `evaluate()` function:

```py
from tqdm.notebook import tqdm

gradient_accumulation_steps = 8
eval_steps = 5_000

model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
        logits = model(batch["input_ids"]).logits
        loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print(
                {
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(commit_message=f"Training in progress step {step}", blocking=False)
```

And that's it -- you now have your own custom training loop for causal language models such as GPT-2 that you can further customize to your needs. 

> [!TIP]
> ✏️ **Try it out!** Either create your own custom loss function tailored to your use case, or add another custom step into the training loop.

> [!TIP]
> ✏️ **Try it out!** When running long training experiments it's a good idea to log important metrics using tools such as TensorBoard or Weights & Biases. Add proper logging to the training loop so you can always check how the training is going.

{/if}


---

<!-- Section 7.7 -->

<FrameworkSwitchCourse {fw} />

# Question answering[[question-answering]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section7_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section7_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={7}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section7_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter7/section7_tf.ipynb"},
]} />

{/if}

Time to look at question answering! This task comes in many flavors, but the one we'll focus on in this section is called *extractive* question answering. This involves posing questions about a document and identifying the answers as _spans of text_ in the document itself.

<Youtube id="ajPx5LwJD-I"/>

We will fine-tune a BERT model on the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/), which consists of questions posed by crowdworkers on a set of Wikipedia articles. This will give us a model able to compute predictions like this one:

<iframe src="https://course-demos-bert-finetuned-squad.hf.space" frameBorder="0" height="450" title="Gradio app" class="block dark:hidden container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

This is actually showcasing the model that was trained and uploaded to the Hub using the code shown in this section. You can find it and double-check the predictions [here](https://huggingface.co/huggingface-course/bert-finetuned-squad?context=%F0%9F%A4%97+Transformers+is+backed+by+the+three+most+popular+deep+learning+libraries+%E2%80%94+Jax%2C+PyTorch+and+TensorFlow+%E2%80%94+with+a+seamless+integration+between+them.+It%27s+straightforward+to+train+your+models+with+one+before+loading+them+for+inference+with+the+other.&question=Which+deep+learning+libraries+back+%F0%9F%A4%97+Transformers%3F).

> [!TIP]
> 💡 Encoder-only models like BERT tend to be great at extracting answers to factoid questions like "Who invented the Transformer architecture?" but fare poorly when given open-ended questions like "Why is the sky blue?" In these more challenging cases, encoder-decoder models like T5 and BART are typically used to synthesize the information in a way that's quite similar to [text summarization](/course/chapter7/5). If you're interested in this type of *generative* question answering, we recommend checking out our [demo](https://yjernite.github.io/lfqa.html) based on the [ELI5 dataset](https://huggingface.co/datasets/eli5).

## Preparing the data[[preparing-the-data]]

The dataset that is used the most as an academic benchmark for extractive question answering is [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), so that's the one we'll use here. There is also a harder [SQuAD v2](https://huggingface.co/datasets/squad_v2) benchmark, which includes questions that don't have an answer. As long as your own dataset contains a column for contexts, a column for questions, and a column for answers, you should be able to adapt the steps below.

### The SQuAD dataset[[the-squad-dataset]]

As usual, we can download and cache the dataset in just one step thanks to `load_dataset()`:

```py
from datasets import load_dataset

raw_datasets = load_dataset("squad")
```

We can then have a look at this object to learn more about the SQuAD dataset:

```py
raw_datasets
```

```python out
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
```

It looks like we have everything we need with the `context`, `question`, and `answers` fields, so let's print those for the first element of our training set:

```py
print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])
```

```python out
Context: 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
Question: "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
Answer: {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}
```

The `context` and `question` fields are very straightforward to use. The `answers` field is a bit trickier as it comports a dictionary with two fields that are both lists. This is the format that will be expected by the `squad` metric during evaluation; if you are using your own data, you don't necessarily need to worry about putting the answers in the same format. The `text` field is rather obvious, and the `answer_start` field contains the starting character index of each answer in the context.

During training, there is only one possible answer. We can double-check this by using the `Dataset.filter()` method:

```py
raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
```

```python out
Dataset({features: ["id", "title", "context", "question", "answers"], num_rows: 0})
```

For evaluation, however, there are several possible answers for each sample, which may be the same or different:

```py
print(raw_datasets["validation"][0]["answers"])
print(raw_datasets["validation"][2]["answers"])
```

```python out
{"text": ["Denver Broncos", "Denver Broncos", "Denver Broncos"], "answer_start": [177, 177, 177]}
{
    "text": [
        "Santa Clara, California",
        "Levi's Stadium",
        "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
    ],
    "answer_start": [403, 355, 355],
}
```

We won't dive into the evaluation script as it will all be wrapped up by a 🤗 Datasets metric for us, but the short version is that some of the questions have several possible answers, and this script will compare a predicted answer to all the acceptable answers and take the best score. If we take a look at the sample at index 2, for instance:

```py
print(raw_datasets["validation"][2]["context"])
print(raw_datasets["validation"][2]["question"])
```

```python out
'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'

"Where did Super Bowl 50 take place?"
```

we can see that the answer can indeed be one of the three possibilities we saw before.

### Processing the training data[[processing-the-training-data]]

<Youtube id="qgaM0weJHpA"/>

Let's start with preprocessing the training data. The hard part will be to generate labels for the question's answer, which will be the start and end positions of the tokens corresponding to the answer inside the context.

But let's not get ahead of ourselves. First, we need to convert the text in the input into IDs the model can make sense of, using a tokenizer:

```py
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

As mentioned previously, we'll be fine-tuning a BERT model, but you can use any other model type as long as it has a fast tokenizer implemented. You can see all the architectures that come with a fast version in [this big table](https://huggingface.co/transformers/#supported-frameworks), and to check that the `tokenizer` object you're using is indeed backed by 🤗 Tokenizers you can look at its `is_fast` attribute:

```py
tokenizer.is_fast
```

```python out
True
```

We can pass to our tokenizer the question and the context together, and it will properly insert the special tokens to form a sentence like this:

```
[CLS] question [SEP] context [SEP]
```

Let's double-check:

```py
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])
```

```python out
"[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Architecturally,"

"the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin "
"Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms "
'upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred '
"Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a "
"replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette "
"Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 statues "
"and the Gold Dome ), is a simple, modern stone statue of Mary. [SEP]"
```

The labels will then be the index of the tokens starting and ending the answer, and the model will be tasked to predicted one start and end logit per token in the input, with the theoretical labels being as follow:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/qa_labels.svg" alt="One-hot encoded labels for question answering."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/qa_labels-dark.svg" alt="One-hot encoded labels for question answering."/>
</div>

In this case the context is not too long, but some of the examples in the dataset have very long contexts that will exceed the maximum length we set (which is 384 in this case). As we saw in [Chapter 6](/course/chapter6/4) when we explored the internals of the `question-answering` pipeline, we will deal with long contexts by creating several training features from one sample of our dataset, with a sliding window between them.

To see how this works using the current example, we can limit the length to 100 and use a sliding window of 50 tokens. As a reminder, we use:

- `max_length` to set the maximum length (here 100)
- `truncation="only_second"` to truncate the context (which is in the second position) when the question with its context is too long
- `stride` to set the number of overlapping tokens between two successive chunks (here 50)
- `return_overflowing_tokens=True` to let the tokenizer know we want the overflowing tokens

```py
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```

```python out
'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basi [SEP]'

'[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin [SEP]'
"[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 [SEP]"
"[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP]. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 statues and the Gold Dome ), is a simple, modern stone statue of Mary. [SEP]"
```

As we can see, our example has been in split into four inputs, each of them containing the question and some part of the context. Note that the answer to the question ("Bernadette Soubirous") only appears in the third and last inputs, so by dealing with long contexts in this way we will create some training examples where the answer is not included in the context. For those examples, the labels will be `start_position = end_position = 0` (so we predict the `[CLS]` token). We will also set those labels in the unfortunate case where the answer has been truncated so that we only have the start (or end) of it. For the examples where the answer is fully in the context, the labels will be the index of the token where the answer starts and the index of the token where the answer ends.

The dataset provides us with the start character of the answer in the context, and by adding the length of the answer, we can find the end character in the context. To map those to token indices, we will need to use the offset mappings we studied in [Chapter 6](/course/chapter6/4). We can have our tokenizer return these by passing along `return_offsets_mapping=True`:

```py
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
inputs.keys()
```

```python out
dict_keys(
    [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "offset_mapping",
        "overflow_to_sample_mapping",
    ]
)
```

As we can see, we get back the usual input IDs, token type IDs, and attention mask, as well as the offset mapping we required and an extra key, `overflow_to_sample_mapping`. The corresponding value will be of use to us when we tokenize several texts at the same time (which we should do to benefit from the fact that our tokenizer is backed by Rust). Since one sample can give several features, it maps each feature to the example it originated from. Because here we only tokenized one example, we get a list of `0`s:

```py
inputs["overflow_to_sample_mapping"]
```

```python out
[0, 0, 0, 0]
```

But if we tokenize more examples, this will become more useful:

```py
inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
```

```python out
"The 4 examples gave 19 features."

"Here is where each comes from: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]."
```

As we can see, the first three examples (at indices 2, 3, and 4 in the training set) each gave four features and the last example (at index 5 in the training set) gave 7 features.

This information will be useful to map each feature we get to its corresponding label. As mentioned earlier, those labels are:

- `(0, 0)` if the answer is not in the corresponding span of the context
- `(start_position, end_position)` if the answer is in the corresponding span of the context, with `start_position` being the index of the token (in the input IDs) at the start of the answer and `end_position` being the index of the token (in the input IDs) where the answer ends

To determine which of these is the case and, if relevant, the positions of the tokens, we first find the indices that start and end the context in the input IDs. We could use the token type IDs to do this, but since those do not necessarily exist for all models (DistilBERT does not require them, for instance), we'll instead use the `sequence_ids()` method of the `BatchEncoding` our tokenizer returns. 

Once we have those token indices, we look at the corresponding offsets, which are tuples of two integers representing the span of characters inside the original context. We can thus detect if the chunk of the context in this feature starts after the answer or ends before the answer begins (in which case the label is `(0, 0)`). If that's not the case, we loop to find the first and last token of the answer:

```py
answers = raw_datasets["train"][2:6]["answers"]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # If the answer is not fully inside the context, label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # Otherwise it's the start and end token positions
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

start_positions, end_positions
```

```python out
(
    [83, 51, 19, 0, 0, 64, 27, 0, 34, 0, 0, 0, 67, 34, 0, 0, 0, 0, 0],
    [85, 53, 21, 0, 0, 70, 33, 0, 40, 0, 0, 0, 68, 35, 0, 0, 0, 0, 0],
)
```

Let's take a look at a few results to verify that our approach is correct. For the first feature we find `(83, 85)` as labels, so let's compare the theoretical answer with the decoded span of tokens from 83 to 85 (inclusive):

```py
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")
```

```python out
"Theoretical answer: the Main Building, labels give: the Main Building"
```

So that's a match! Now let's check index 4, where we set the labels to `(0, 0)`, which means the answer is not in the context chunk of that feature:

```py
idx = 4
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

decoded_example = tokenizer.decode(inputs["input_ids"][idx])
print(f"Theoretical answer: {answer}, decoded example: {decoded_example}")
```

```python out
'Theoretical answer: a Marian place of prayer and reflection, decoded example: [CLS] What is the Grotto at Notre Dame? [SEP] Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grot [SEP]'
```

Indeed, we don't see the answer inside the context.

> [!TIP]
> ✏️ **Your turn!** When using the XLNet architecture, padding is applied on the left and the question and context are switched. Adapt all the code we just saw to the XLNet architecture (and add `padding=True`). Be aware that the `[CLS]` token may not be at the 0 position with padding applied.

Now that we have seen step by step how to preprocess our training data, we can group it in a function we will apply on the whole training dataset. We'll pad every feature to the maximum length we set, as most of the contexts will be long (and the corresponding samples will be split into several features), so there is no real benefit to applying dynamic padding here:

```py
max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
```

Note that we defined two constants to determine the maximum length used as well as the length of the sliding window, and that we added a tiny bit of cleanup before tokenizing: some of the questions in the SQuAD dataset have extra spaces at the beginning and the end that don't add anything (and take up space when being tokenized if you use a model like RoBERTa), so we removed those extra spaces.

To apply this function to the whole training set, we use the `Dataset.map()` method with the `batched=True` flag. It's necessary here as we are changing the length of the dataset (since one example can give several training features):

```py
train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
len(raw_datasets["train"]), len(train_dataset)
```

```python out
(87599, 88729)
```

As we can see, the preprocessing added roughly 1,000 features. Our training set is now ready to be used -- let's dig into the preprocessing of the validation set!

### Processing the validation data[[processing-the-validation-data]]

Preprocessing the validation data will be slightly easier as we don't need to generate labels (unless we want to compute a validation loss, but that number won't really help us understand how good the model is). The real joy will be to interpret the predictions of the model into spans of the original context. For this, we will just need to store both the offset mappings and some way to match each created feature to the original example it comes from. Since there is an ID column in the original dataset, we'll use that ID.

The only thing we'll add here is a tiny bit of cleanup of the offset mappings. They will contain offsets for the question and the context, but once we're in the post-processing stage we won't have any way to know which part of the input IDs corresponded to the context and which part was the question (the `sequence_ids()` method we used is available for the output of the tokenizer only). So, we'll set the offsets corresponding to the question to `None`:

```py
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs
```

We can apply this function on the whole validation dataset like before:

```py
validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
len(raw_datasets["validation"]), len(validation_dataset)
```

```python out
(10570, 10822)
```

In this case we've only added a couple of hundred samples, so it appears the contexts in the validation dataset are a bit shorter.

Now that we have preprocessed all the data, we can get to the training. 

{#if fw === 'pt'}

## Fine-tuning the model with the `Trainer` API[[fine-tuning-the-model-with-the-trainer-api]]

The training code for this example will look a lot like the code in the previous sections -- the hardest thing will be to write the `compute_metrics()` function. Since we padded all the samples to the maximum length we set, there is no data collator to define, so this metric computation is really the only thing we have to worry about. The difficult part will be to post-process the model predictions into spans of text in the original examples; once we have done that, the metric from the 🤗 Datasets library will do most of the work for us.

{:else}

## Fine-tuning the model with Keras[[fine-tuning-the-model-with-keras]]

The training code for this example will look a lot like the code in the previous sections, but computing the metrics will be uniquely challenging. Since we padded all the samples to the maximum length we set, there is no data collator to define, so this metric computation is really the only thing we have to worry about. The hard part will be to post-process the model predictions into spans of text in the original examples; once we have done that, the metric from the 🤗 Datasets library will do most of the work for us.

{/if}

### Post-processing[[post-processing]]

{#if fw === 'pt'}

<Youtube id="BNy08iIWVJM"/>

{:else}

<Youtube id="VN67ZpN33Ss"/>

{/if}

The model will output logits for the start and end positions of the answer in the input IDs, as we saw during our exploration of the [`question-answering` pipeline](/course/chapter6/3b). The post-processing step will be similar to what we did there, so here's a quick reminder of the actions we took:

- We masked the start and end logits corresponding to tokens outside of the context.
- We then converted the start and end logits into probabilities using a softmax.
- We attributed a score to each `(start_token, end_token)` pair by taking the product of the corresponding two probabilities.
- We looked for the pair with the maximum score that yielded a valid answer (e.g., a `start_token` lower than `end_token`).

Here we will change this process slightly because we don't need to compute actual scores (just the predicted answer). This means we can skip the softmax step. To go faster, we also won't score all the possible `(start_token, end_token)` pairs, but only the ones corresponding to the highest `n_best` logits (with `n_best=20`). Since we will skip the softmax, those scores will be logit scores, and will be obtained by taking the sum of the start and end logits (instead of the product, because of the rule \\(\log(ab) = \log(a) + \log(b)\\)).

To demonstrate all of this, we will need some kind of predictions. Since we have not trained our model yet, we are going to use the default model for the QA pipeline to generate some predictions on a small part of the validation set. We can use the same processing function as before; because it relies on the global constant `tokenizer`, we just have to change that object to the tokenizer of the model we want to use temporarily:

```python
small_eval_set = raw_datasets["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
```

Now that the preprocessing is done, we change the tokenizer back to the one we originally picked:

```python
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

We then remove the columns of our `eval_set` that are not expected by the model, build a batch with all of that small validation set, and pass it through the model. If a GPU is available, we use it to go faster:

{#if fw === 'pt'}

```python
import torch
from transformers import AutoModelForQuestionAnswering

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)

with torch.no_grad():
    outputs = trained_model(**batch)
```

Since the `Trainer` will give us predictions as NumPy arrays, we grab the start and end logits and convert them to that format:

```python
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()
```

{:else}

```python
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("numpy")

batch = {k: eval_set_for_model[k] for k in eval_set_for_model.column_names}
trained_model = TFAutoModelForQuestionAnswering.from_pretrained(trained_checkpoint)

outputs = trained_model(**batch)
```

For ease of experimentation, let's convert these outputs to NumPy arrays:

```python
start_logits = outputs.start_logits.numpy()
end_logits = outputs.end_logits.numpy()
```

{/if}

Now, we need to find the predicted answer for each example in our `small_eval_set`. One example may have been split into several features in `eval_set`, so the first step is to map each example in `small_eval_set` to the corresponding features in `eval_set`:

```python
import collections

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)
```

With this in hand, we can really get to work by looping through all the examples and, for each example, through all the associated features. As we said before, we'll look at the logit scores for the `n_best` start logits and end logits, excluding positions that give:

- An answer that wouldn't be inside the context
- An answer with negative length
- An answer that is too long (we limit the possibilities at `max_answer_length=30`)

Once we have all the scored possible answers for one example, we just pick the one with the best logit score:

```python
import numpy as np

n_best = 20
max_answer_length = 30
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
```

The final format of the predicted answers is the one that will be expected by the metric we will use. As usual, we can load it with the help of the 🤗 Evaluate library:

```python
import evaluate

metric = evaluate.load("squad")
```

This metric expects the predicted answers in the format we saw above (a list of dictionaries with one key for the ID of the example and one key for the predicted text) and the theoretical answers in the format below (a list of dictionaries with one key for the ID of the example and one key for the possible answers):

```python
theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set]
```

We can now check that we get sensible results by looking at the first element of both lists:

```python
print(predicted_answers[0])
print(theoretical_answers[0])
```

```python out
{"id": "56be4db0acb8001400a502ec", "prediction_text": "Denver Broncos"}
{
    "id": "56be4db0acb8001400a502ec",
    "answers": {
        "text": ["Denver Broncos", "Denver Broncos", "Denver Broncos"],
        "answer_start": [177, 177, 177],
    },
}
```

Not too bad! Now let's have a look at the score the metric gives us:

```python
metric.compute(predictions=predicted_answers, references=theoretical_answers)
```

```python out
{"exact_match": 83.0, "f1": 88.25}
```

Again, that's rather good considering that according to [its paper](https://arxiv.org/abs/1910.01108v2) DistilBERT fine-tuned on SQuAD obtains 79.1 and 86.9 for those scores on the whole dataset.

{#if fw === 'pt'}

Now let's put everything we just did in a `compute_metrics()` function that we will use in the `Trainer`. Normally, that `compute_metrics()` function only receives a tuple `eval_preds` with logits and labels. Here we will need a bit more, as we have to look in the dataset of features for the offset and in the dataset of examples for the original contexts, so we won't be able to use this function to get regular evaluation results during training. We will only use it at the end of training to check the results.

The `compute_metrics()` function groups the same steps as before; we just add a small check in case we don't come up with any valid answers (in which case we predict an empty string).

{:else}

Now let's put everything we just did in a `compute_metrics()` function that we will use after training our model. We will need to pass a bit more than just the output logits, as we have to look in the dataset of features for the offset and in the dataset of examples for the original contexts:

{/if}

```python
from tqdm.auto import tqdm


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
```

We can check it works on our predictions:

```python
compute_metrics(start_logits, end_logits, eval_set, small_eval_set)
```

```python out
{"exact_match": 83.0, "f1": 88.25}
```

Looking good! Now let's use this to fine-tune our model.

### Fine-tuning the model[[fine-tuning-the-model]]

{#if fw === 'pt'}

We are now ready to train our model. Let's create it first, using the `AutoModelForQuestionAnswering` class like before:

```python
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

{:else}

We are now ready to train our model. Let's create it first, using the `TFAutoModelForQuestionAnswering` class like before:

```python
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

{/if}

As usual, we get a warning that some weights are not used (the ones from the pretraining head) and some others are initialized randomly (the ones for the question answering head). You should be used to this by now, but that means this model is not ready to be used just yet and needs fine-tuning -- good thing we're about to do that!

To be able to push our model to the Hub, we'll need to log in to Hugging Face. If you're running this code in a notebook, you can do so with the following utility function, which displays a widget where you can enter your login credentials:

```python
from huggingface_hub import notebook_login

notebook_login()
```

If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

{#if fw === 'pt'}

Once this is done, we can define our `TrainingArguments`. As we said when we defined our function to compute the metric, we won't be able to have a regular evaluation loop because of the signature of the `compute_metrics()` function. We could write our own subclass of `Trainer` to do this (an approach you can find in the [question answering example script](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py)), but that's a bit too long for this section. Instead, we will only evaluate the model at the end of training here and show you how to do a regular evaluation in "A custom training loop" below.

This is really where the `Trainer` API shows its limits and the 🤗 Accelerate library shines: customizing the class to a specific use case can be painful, but tweaking a fully exposed training loop is easy.

Let's take a look at our `TrainingArguments`:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)
```

We've seen most of these before: we set some hyperparameters (like the learning rate, the number of epochs we train for, and some weight decay) and indicate that we want to save the model at the end of every epoch, skip evaluation, and upload our results to the Model Hub. We also enable mixed-precision training with `fp16=True`, as it can speed up the training nicely on a recent GPU.

{:else}

Now that's done, we can create our TF Datasets. We can use the simple default data collator this time:

```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")
```

And now we create the datasets as usual.

```python
tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)
tf_eval_dataset = model.prepare_tf_dataset(
    validation_dataset,
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
```

Next, we set up our training hyperparameters and compile our model:

```python
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_epochs = 3
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

Finally, we're ready to train with `model.fit()`. We use a `PushToHubCallback` to upload the model to the Hub after each epoch.

{/if}

By default, the repository used will be in your namespace and named after the output directory you set, so in our case it will be in `"sgugger/bert-finetuned-squad"`. We can override this by passing a `hub_model_id`; for instance, to push the model to the `huggingface_course` organization we used `hub_model_id="huggingface_course/bert-finetuned-squad"` (which is the model we linked to at the beginning of this section).

{#if fw === 'pt'}

> [!TIP]
> 💡 If the output directory you are using exists, it needs to be a local clone of the repository you want to push to (so set a new name if you get an error when defining your `Trainer`).

Finally, we just pass everything to the `Trainer` class and launch the training:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

{:else}

```python
from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="bert-finetuned-squad", tokenizer=tokenizer)

# We're going to do validation afterwards, so no validation mid-training
model.fit(tf_train_dataset, callbacks=[callback], epochs=num_train_epochs)
```

{/if}

Note that while the training happens, each time the model is saved (here, every epoch) it is uploaded to the Hub in the background. This way, you will be able to to resume your training on another machine if necessary. The whole training takes a while (a little over an hour on a Titan RTX), so you can grab a coffee or reread some of the parts of the course that you've found more challenging while it proceeds. Also note that as soon as the first epoch is finished, you will see some weights uploaded to the Hub and you can start playing with your model on its page.

{#if fw === 'pt'}

Once the training is complete, we can finally evaluate our model (and pray we didn't spend all that compute time on nothing). The `predict()` method of the `Trainer` will return a tuple where the first elements will be the predictions of the model (here a pair with the start and end logits). We send this to our `compute_metrics()` function:

```python
predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
```

{:else}

Once the training is complete, we can finally evaluate our model (and pray we didn't spend all that compute time on nothing). The `predict()` method of our `model` will take care of getting predictions, and since we did all the hard work of defining a `compute_metrics()` function earlier, we can get our results in a single line:

```python
predictions = model.predict(tf_eval_dataset)
compute_metrics(
    predictions["start_logits"],
    predictions["end_logits"],
    validation_dataset,
    raw_datasets["validation"],
)
```

{/if}

```python out
{"exact_match": 81.18259224219489, "f1": 88.67381321905516}
```

Great! As a comparison, the baseline scores reported in the BERT article for this model are 80.8 and 88.5, so we're right where we should be.

{#if fw === 'pt'}

Finally, we use the `push_to_hub()` method to make sure we upload the latest version of the model:

```py
trainer.push_to_hub(commit_message="Training complete")
```

This returns the URL of the commit it just did, if you want to inspect it:

```python out
"https://huggingface.co/sgugger/bert-finetuned-squad/commit/9dcee1fbc25946a6ed4bb32efb1bd71d5fa90b68"
```

The `Trainer` also drafts a model card with all the evaluation results and uploads it.

{/if}

At this stage, you can use the inference widget on the Model Hub to test the model and share it with your friends, family, and favorite pets. You have successfully fine-tuned a model on a question answering task -- congratulations!

> [!TIP]
> ✏️ **Your turn!** Try another model architecture to see if it performs better on this task!

{#if fw === 'pt'}

If you want to dive a bit more deeply into the training loop, we will now show you how to do the same thing using 🤗  Accelerate.

## A custom training loop[[a-custom-training-loop]]

Let's now have a look at the full training loop, so you can easily customize the parts you need. It will look a lot like the training loop in [Chapter 3](/course/chapter3/4), with the exception of the evaluation loop. We will be able to evaluate the model regularly since we're not constrained by the `Trainer` class anymore.

### Preparing everything for training[[preparing-everything-for-training]]

First we need to build the `DataLoader`s from our datasets. We set the format of those datasets to `"torch"`, and remove the columns in the validation set that are not used by the model. Then, we can use the `default_data_collator` provided by Transformers as a `collate_fn` and shuffle the training set, but not the validation set:

```py
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(validation_set, collate_fn=default_data_collator, batch_size=8)
```

Next we reinstantiate our model, to make sure we're not continuing the fine-tuning from before but starting from the BERT pretrained model again:

```py
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

Then we will need an optimizer. As usual we use the classic `AdamW`, which is like Adam, but with a fix in the way weight decay is applied:

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Once we have all those objects, we can send them to the `accelerator.prepare()` method. Remember that if you want to train on TPUs in a Colab notebook, you will need to move all of this code into a training function, and that shouldn't execute any cell that instantiates an `Accelerator`. We can force mixed-precision training by passing `fp16=True` to the `Accelerator` (or, if you are executing the code as a script, just make sure to fill in the 🤗 Accelerate `config` appropriately).

```py
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

As you should know from the previous sections, we can only use the `train_dataloader` length to compute the number of training steps after it has gone through the `accelerator.prepare()` method. We use the same linear schedule as in the previous sections:

```py
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

To push our model to the Hub, we will need to create a `Repository` object in a working folder. First log in to the Hugging Face Hub, if you're not logged in already. We'll determine the repository name from the model ID we want to give our model (feel free to replace the `repo_name` with your own choice; it just needs to contain your username, which is what the function `get_full_repo_name()` does):

```py
from huggingface_hub import Repository, get_full_repo_name

model_name = "bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

```python out
"sgugger/bert-finetuned-squad-accelerate"
```

Then we can clone that repository in a local folder. If it already exists, this local folder should be a clone of the repository we are working with:

```py
output_dir = "bert-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
```

We can now upload anything we save in `output_dir` by calling the `repo.push_to_hub()` method. This will help us upload the intermediate models at the end of each epoch.

## Training loop[[training-loop]]

We are now ready to write the full training loop. After defining a progress bar to follow how training goes, the loop has three parts:

- The training in itself, which is the classic iteration over the `train_dataloader`, forward pass through the model, then backward pass and optimizer step.
- The evaluation, in which we gather all the values for `start_logits` and `end_logits` before converting them to NumPy arrays. Once the evaluation loop is finished, we concatenate all the results. Note that we need to truncate because the `Accelerator` may have added a few samples at the end to ensure we have the same number of examples in each process.
- Saving and uploading, where we first save the model and the tokenizer, then call `repo.push_to_hub()`. As we did before, we use the argument `blocking=False` to tell the 🤗 Hub library to push in an asynchronous process. This way, training continues normally and this (long) instruction is executed in the background.

Here's the complete code for the training loop:

```py
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    print(f"epoch {epoch}:", metrics)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
```

In case this is the first time you're seeing a model saved with 🤗 Accelerate, let's take a moment to inspect the three lines of code that go with it:

```py
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
```

The first line is self-explanatory: it tells all the processes to wait until everyone is at that stage before continuing. This is to make sure we have the same model in every process before saving. Then we grab the `unwrapped_model`, which is the base model we defined. The `accelerator.prepare()` method changes the model to work in distributed training, so it won't have the `save_pretrained()` method anymore; the `accelerator.unwrap_model()` method undoes that step. Lastly, we call `save_pretrained()` but tell that method to use `accelerator.save()` instead of `torch.save()`. 

Once this is done, you should have a model that produces results pretty similar to the one trained with the `Trainer`. You can check the model we trained using this code at [*huggingface-course/bert-finetuned-squad-accelerate*](https://huggingface.co/huggingface-course/bert-finetuned-squad-accelerate). And if you want to test out any tweaks to the training loop, you can directly implement them by editing the code shown above!

{/if}

## Using the fine-tuned model[[using-the-fine-tuned-model]]

We've already shown you how you can use the model we fine-tuned on the Model Hub with the inference widget. To use it locally in a `pipeline`, you just have to specify the model identifier:

```py
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)
```

```python out
{"score": 0.9979003071784973, "start": 78, "end": 105, "answer": "Jax, PyTorch and TensorFlow"}
```

Great! Our model is working as well as the default one for this pipeline!


---

<!-- Section 7.8 -->

# Mastering LLMs[[mastering-llms]]

<CourseFloatingBanner
    chapter={7}
    classNames="absolute z-10 right-0 top-0"
/>

If you've made it this far in the course, congratulations -- you now have all the knowledge and tools you need to tackle (almost) any language task with 🤗 Transformers and the Hugging Face ecosystem!

## From NLP to LLMs

While we've covered many traditional NLP tasks in this course, the field has been revolutionized by Large Language Models (LLMs). These models have dramatically expanded what's possible in language processing:

- They can handle multiple tasks without task-specific fine-tuning
- They excel at following instructions and adapting to different contexts
- They can generate coherent, contextually appropriate text for various applications
- They can perform reasoning and solve complex problems through techniques like chain-of-thought prompting

The foundational NLP skills you've learned are still essential for working with LLMs effectively. Understanding tokenization, model architectures, fine-tuning approaches, and evaluation metrics provides the knowledge needed to leverage LLMs to their full potential.

We have seen a lot of different data collators, so we made this little video to help you find which one to use for each task:

<Youtube id="-RPeakdlHYo"/>

After completing this lightning tour through the core language tasks, you should:

* Know which architectures (encoder, decoder, or encoder-decoder) are best suited for each task
* Understand the difference between pretraining and fine-tuning a language model
* Know how to train Transformer models using either the `Trainer` API and distributed training features of 🤗 Accelerate or TensorFlow and Keras, depending on which track you've been following
* Understand the meaning and limitations of metrics like ROUGE and BLEU for text generation tasks
* Know how to interact with your fine-tuned models, both on the Hub and using the `pipeline` from 🤗 Transformers
* Appreciate how LLMs build upon and extend traditional NLP techniques

Despite all this knowledge, there will come a time when you'll either encounter a difficult bug in your code or have a question about how to solve a particular language processing problem. Fortunately, the Hugging Face community is here to help you! In the final chapter of this part of the course, we'll explore how you can debug your Transformer models and ask for help effectively.

---

<!-- Section 7.9 -->

<FrameworkSwitchCourse {fw} />

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={7}
    classNames="absolute z-10 right-0 top-0"
/>

Let's test what you learned in this chapter!

### 1. Which of the following tasks can be framed as a token classification problem?

<Question
	choices={[
		{
			text: "Find the grammatical components in a sentence.",
			explain: "Correct! We can then label each word as a noun, verb, etc.",
			correct: true
		},
		{
			text: "Find whether a sentence is grammatically correct or not.",
			explain: "No, this is a sequence classification problem."
		},
		{
			text: "Find the persons mentioned in a sentence.",
			explain: "Correct! We can label each word as person or not person.",
            correct: true
		},
        {
			text: "Find the chunk of words in a sentence that answers a question.",
			explain: "No, that would be a question answering problem."
		}
	]}
/>

### 2. What part of the preprocessing for token classification differs from the other preprocessing pipelines?

<Question
	choices={[
		{
			text: "There is no need to do anything; the texts are already tokenized.",
			explain: "The texts are indeed given as separate words, but we still need to apply the subword tokenization model."
		},
		{
			text: "The texts are given as words, so we only need to apply subword tokenization.",
			explain: "Correct! This is different from the usual preprocessing, where we need to apply the full tokenization pipeline. Can you think of another difference?",
			correct: true
		},
		{
			text: "We use <code>-100</code> to label the special tokens.",
			explain: "That's not specific to token classification -- we always use <code>-100</code> as the label for tokens we want to ignore in the loss."
		},
		{
			text: "We need to make sure to truncate or pad the labels to the same size as the inputs, when applying truncation/padding.",
			explain: "Indeed! That's not the only difference, though.",
			correct: true
		}
	]}
/>

### 3. What problem arises when we tokenize the words in a token classification problem and want to label the tokens?

<Question
	choices={[
		{
			text: "The tokenizer adds special tokens and we have no labels for them.",
			explain: "We label these <code>-100</code> so they are ignored in the loss."
		},
		{
			text: "Each word can produce several tokens, so we end up with more tokens than we have labels.",
			explain: "That is the main problem, and we need to align the original labels with the tokens.",
			correct: true
		},
		{
			text: "The added tokens have no labels, so there is no problem.",
			explain: "That's incorrect; we need as many labels as we have tokens or our models will error out."
		}
	]}
/>

### 4. What does "domain adaptation" mean?

<Question
	choices={[
		{
			text: "It's when we run a model on a dataset and get the predictions for each sample in that dataset.",
			explain: "No, this is just running inference."
		},
		{
			text: "It's when we train a model on a dataset.",
			explain: "No, this is training a model; there is no adaptation here."
		},
		{
			text: "It's when we fine-tune a pretrained model on a new dataset, and it gives predictions that are more adapted to that dataset",
			explain: "Correct! The model adapted its knowledge to the new dataset.",
            correct: true
		},
        {
			text: "It's when we add misclassified samples to a dataset to make our model more robust.",
			explain: "That's certainly something you should do if you retrain your model regularly, but it's not domain adaptation."
		}
	]}
/>

### 5. What are the labels in a masked language modeling problem?

<Question
	choices={[
		{
			text: "Some of the tokens in the input sentence are randomly masked and the labels are the original input tokens.",
			explain: "That's it!",
            correct: true
		},
		{
			text: "Some of the tokens in the input sentence are randomly masked and the labels are the original input tokens, shifted to the left.",
			explain: "No, shifting the labels to the left corresponds to predicting the next word, which is causal language modeling."
		},
		{
			text: "Some of the tokens in the input sentence are randomly masked, and the label is whether the sentence is positive or negative.",
			explain: "That's a sequence classification problem with some data augmentation, not masked language modeling."
		},
        {
			text: "Some of the tokens in the two input sentences are randomly masked, and the label is whether the two sentences are similar or not.",
			explain: "That's a sequence classification problem with some data augmentation, not masked language modeling."
		}
	]}
/>

### 6. Which of these tasks can be seen as a sequence-to-sequence problem?

<Question
	choices={[
		{
			text: "Writing short reviews of long documents",
			explain: "Yes, that's a summarization problem. Try another answer!",
            correct: true
		},
		{
			text: "Answering questions about a document",
			explain: "This can be framed as a sequence-to-sequence problem. It's not the only right answer, though.",
            correct: true
		},
		{
			text: "Translating a text in Chinese into English",
			explain: "That's definitely a sequence-to-sequence problem. Can you spot another one?",
            correct: true
		},
        {
			text: "Fixing the messages sent by my nephew/friend so they're in proper English",
			explain: "That's a kind of translation problem, so definitely a sequence-to-sequence task. This isn't the only right answer, though!",
			correct: true
		}
	]}
/>

### 7. What is the proper way to preprocess the data for a sequence-to-sequence problem?

<Question
	choices={[
		{
			text: "The inputs and targets have to be sent together to the tokenizer with <code>inputs=...</code> and <code>targets=...</code>.",
			explain: "This might be an API we add in the future, but that's not possible right now."
		},
		{
			text: "The inputs and the targets both have to be preprocessed, in two separate calls to the tokenizer.",
			explain: "That is true, but incomplete. There is something you need to do to make sure the tokenizer processes both properly."
		},
		{
			text: "As usual, we just have to tokenize the inputs.",
			explain: "Not in a sequence classification problem; the targets are also texts we need to convert into numbers!"
		},
        {
			text: "The inputs have to be sent to the tokenizer, and the targets too, but under a special context manager.",
			explain: "That's correct, the tokenizer needs to be put into target mode by that context manager.",
			correct: true
		}
	]}
/>

{#if fw === 'pt'}

### 8. Why is there a specific subclass of `Trainer` for sequence-to-sequence problems?

<Question
	choices={[
		{
			text: "Because sequence-to-sequence problems use a custom loss, to ignore the labels set to <code>-100</code>",
			explain: "That's not a custom loss at all, but the way the loss is always computed."
		},
		{
			text: "Because sequence-to-sequence problems require a special evaluation loop",
			explain: "That's correct. Sequence-to-sequence models' predictions are often run using the <code>generate()</code> method.",
			correct: true
		},
		{
			text: "Because the targets are texts in sequence-to-sequence problems",
			explain: "The <code>Trainer</code> doesn't really care about that since they have been preprocessed before."
		},
        {
			text: "Because we use two models in sequence-to-sequence problems",
			explain: "We do use two models in a way, an encoder and a decoder, but they are grouped together in one model."
		}
	]}
/>

{:else}

### 9. Why is it often unnecessary to specify a loss when calling `compile()` on a Transformer model?

<Question
	choices={[
		{
			text: "Because Transformer models are trained with unsupervised learning",
			explain: "Not quite -- even unsupervised learning needs a loss function!"
		},
		{
			text: "Because the model's internal loss output is used by default",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "Because we compute metrics after training instead",
			explain: "We do often do that, but it doesn't explain where we get the loss value we optimize in training."
		},
        {
			text: "Because loss is specified in `model.fit()` instead",
			explain: "No, the loss function is always fixed once you run `model.compile()`, and can't be changed in `model.fit()`."
		}
	]}
/>

{/if}

### 10. When should you pretrain a new model?

<Question
	choices={[
		{
			text: "When there is no pretrained model available for your specific language",
			explain: "That's correct.",
			correct: true
		},
		{
			text: "When you have lots of data available, even if there is a pretrained model that could work on it",
			explain: "In this case, you should probably use the pretrained model and fine-tune it on your data, to avoid huge compute costs."
		},
		{
			text: "When you have concerns about the bias of the pretrained model you are using",
			explain: "That is true, but you have to make very sure the data you will use for training is really better.",
			correct: true
		},
        {
			text: "When the pretrained models available are just not good enough",
			explain: "Are you sure you've properly debugged your training, then?"
		}
	]}
/>

### 11. Why is it easy to pretrain a language model on lots and lots of texts?

<Question
	choices={[
		{
			text: "Because there are plenty of texts available on the internet",
			explain: "Although true, that doesn't really answer the question. Try again!"
		},
		{
			text: "Because the pretraining objective does not require humans to label the data",
			explain: "That's correct, language modeling is a self-supervised problem.",
			correct: true
		},
		{
			text: "Because the 🤗 Transformers library only requires a few lines of code to start the training",
			explain: "Although true, that doesn't really answer the question asked. Try another answer!"
		}
	]}
/>

### 12. What are the main challenges when preprocessing data for a question answering task?

<Question
	choices={[
		{
			text: "You need to tokenize the inputs.",
			explain: "That's correct, but is it really a main challenge?"
		},
		{
			text: "You need to deal with very long contexts, which give several training features that may or may not have the answer in them.",
			explain: "This is definitely one of the challenges.",
			correct: true
		},
		{
			text: "You need to tokenize the answers to the question as well as the inputs.",
			explain: "No, unless you are framing your question answering problem as a sequence-to-sequence task."
		},
       {
			text: "From the answer span in the text, you have to find the start and end token in the tokenized input.",
			explain: "That's one of the hard parts, yes!",
			correct: true
		}
	]}
/>

### 13. How is post-processing usually done in question answering?

<Question
	choices={[
		{
			text: "The model gives you the start and end positions of the answer, and you just have to decode the corresponding span of tokens.",
			explain: "That could be one way to do it, but it's a bit too simplistic."
		},
		{
			text: "The model gives you the start and end positions of the answer for each feature created by one example, and you just have to decode the corresponding span of tokens in the one that has the best score.",
			explain: "That's close to the post-processing we studied, but it's not entirely right."
		},
		{
			text: "The model gives you the start and end positions of the answer for each feature created by one example, and you just have to match them to the span in the context for the one that has the best score.",
			explain: "That's it in a nutshell!",
			correct: true
		},
        {
			text: "The model generates an answer, and you just have to decode it.",
			explain: "No, unless you are framing your question answering problem as a sequence-to-sequence task."
		}
	]}
/>

