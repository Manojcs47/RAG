# Hugging Face NLP Course — Chapter 5

Source: https://huggingface.co/learn/nlp-course/chapter5


---

<!-- Section 5.1 -->

# Introduction[[introduction]]

<CourseFloatingBanner
    chapter={5}
    classNames="absolute z-10 right-0 top-0"
/>

In [Chapter 3](/course/chapter3) you got your first taste of the 🤗 Datasets library and saw that there were three main steps when it came to fine-tuning a model:

1. Load a dataset from the Hugging Face Hub.
2. Preprocess the data with `Dataset.map()`.
3. Load and compute metrics.

But this is just scratching the surface of what 🤗 Datasets can do! In this chapter, we will take a deep dive into the library. Along the way, we'll find answers to the following questions:

* What do you do when your dataset is not on the Hub?
* How can you slice and dice a dataset? (And what if you _really_ need to use Pandas?)
* What do you do when your dataset is huge and will melt your laptop's RAM?
* What the heck are "memory mapping" and Apache Arrow?
* How can you create your own dataset and push it to the Hub?

The techniques you learn here will prepare you for the advanced tokenization and fine-tuning tasks in [Chapter 6](/course/chapter6) and [Chapter 7](/course/chapter7) -- so grab a coffee and let's get started!

---

<!-- Section 5.2 -->

# What if my dataset isn't on the Hub?[[what-if-my-dataset-isnt-on-the-hub]]

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section2.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section2.ipynb"},
]} />

You know how to use the [Hugging Face Hub](https://huggingface.co/datasets) to download datasets, but you'll often find yourself working with data that is stored either on your laptop or on a remote server. In this section we'll show you how 🤗 Datasets can be used to load datasets that aren't available on the Hugging Face Hub.

<Youtube id="HyQgpJTkRdE"/>

## Working with local and remote datasets[[working-with-local-and-remote-datasets]]

🤗 Datasets provides loading scripts to handle the loading of local and remote datasets. It supports several common data formats, such as:

|    Data format     | Loading script |                         Example                         |
| :----------------: | :------------: | :-----------------------------------------------------: |
|     CSV & TSV      |     `csv`      |     `load_dataset("csv", data_files="my_file.csv")`     |
|     Text files     |     `text`     |    `load_dataset("text", data_files="my_file.txt")`     |
| JSON & JSON Lines  |     `json`     |   `load_dataset("json", data_files="my_file.jsonl")`    |
| Pickled DataFrames |    `pandas`    | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

As shown in the table, for each data format we just need to specify the type of loading script in the `load_dataset()` function, along with a `data_files` argument that specifies the path to one or more files. Let's start by loading a dataset from local files; later we'll see how to do the same with remote files.

## Loading a local dataset[[loading-a-local-dataset]]

For this example we'll use the [SQuAD-it dataset](https://github.com/crux82/squad-it/), which is a large-scale dataset for question answering in Italian.

The training and test splits are hosted on GitHub, so we can download them with a simple `wget` command:

```python
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
```

This will download two compressed files called *SQuAD_it-train.json.gz* and *SQuAD_it-test.json.gz*, which we can decompress with the Linux `gzip` command:

```python
!gzip -dkv SQuAD_it-*.json.gz
```

```bash
SQuAD_it-test.json.gz:	   87.4% -- replaced with SQuAD_it-test.json
SQuAD_it-train.json.gz:	   82.2% -- replaced with SQuAD_it-train.json
```

We can see that the compressed files have been replaced with _SQuAD_it-train.json_ and _SQuAD_it-test.json_, and that the data is stored in the JSON format.

> [!TIP]
> ✎ If you're wondering why there's a `!` character in the above shell commands, that's because we're running them within a Jupyter notebook. Simply remove the prefix if you want to download and unzip the dataset within a terminal.

To load a JSON file with the `load_dataset()` function, we just need to know if we're dealing with ordinary JSON (similar to a nested dictionary) or JSON Lines (line-separated JSON). Like many question answering datasets, SQuAD-it uses the nested format, with all the text stored in a `data` field. This means we can load the dataset by specifying the `field` argument as follows:

```py
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
```

By default, loading local files creates a `DatasetDict` object with a `train` split. We can see this by inspecting the `squad_it_dataset` object:

```py
squad_it_dataset
```

```python out
DatasetDict({train: Dataset({features: ["title", "paragraphs"], num_rows: 442})})
```

This shows us the number of rows and the column names associated with the training set. We can view one of the examples by indexing into the `train` split as follows:

```py
squad_it_dataset["train"][0]
```

```python out
{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...,
            ],
        },
        ...,
    ],
}
```

Great, we've loaded our first local dataset! But while this worked for the training set, what we really want is to include both the `train` and `test` splits in a single `DatasetDict` object so we can apply `Dataset.map()` functions across both splits at once. To do this, we can provide a dictionary to the `data_files` argument that maps each split name to a file associated with that split:

```py
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
```

This is exactly what we wanted. Now, we can apply various preprocessing techniques to clean up the data, tokenize the reviews, and so on.

> [!TIP]
> The `data_files` argument of the `load_dataset()` function is quite flexible and can be either a single file path, a list of file paths, or a dictionary that maps split names to file paths. You can also glob files that match a specified pattern according to the rules used by the Unix shell (e.g., you can glob all the JSON files in a directory as a single split by setting `data_files="*.json"`). See the 🤗 Datasets [documentation](https://huggingface.co/docs/datasets/loading#local-and-remote-files) for more details.

The loading scripts in 🤗 Datasets actually support automatic decompression of the input files, so we could have skipped the use of `gzip` by pointing the `data_files` argument directly to the compressed files:

```py
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

This can be useful if you don't want to manually decompress many GZIP files. The automatic decompression also applies to other common formats like ZIP and TAR, so you just need to point `data_files` to the compressed files and you're good to go!

Now that you know how to load local files on your laptop or desktop, let's take a look at loading remote files.

## Loading a remote dataset[[loading-a-remote-dataset]]

If you're working as a data scientist or coder in a company, there's a good chance the datasets you want to analyze are stored on some remote server. Fortunately, loading remote files is just as simple as loading local ones! Instead of providing a path to local files, we point the `data_files` argument of `load_dataset()` to one or more URLs where the remote files are stored. For example, for the SQuAD-it dataset hosted on GitHub, we can just point `data_files` to the _SQuAD_it-*.json.gz_ URLs as follows:

```py
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

This returns the same `DatasetDict` object obtained above, but saves us the step of manually downloading and decompressing the _SQuAD_it-*.json.gz_ files. This wraps up our foray into the various ways to load datasets that aren't hosted on the Hugging Face Hub. Now that we've got a dataset to play with, let's get our hands dirty with various data-wrangling techniques!

> [!TIP]
> ✏️ **Try it out!** Pick another dataset hosted on GitHub or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) and try loading it both locally and remotely using the techniques introduced above. For bonus points, try loading a dataset that’s stored in a CSV or text format (see the [documentation](https://huggingface.co/docs/datasets/loading#local-and-remote-files) for more information on these formats).




---

<!-- Section 5.3 -->

# Time to slice and dice[[time-to-slice-and-dice]]

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section3.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section3.ipynb"},
]} />

Most of the time, the data you work with won't be perfectly prepared for training models. In this section we'll explore the various features that 🤗 Datasets provides to clean up your datasets.

<Youtube id="tqfSFcPMgOI"/>

## Slicing and dicing our data[[slicing-and-dicing-our-data]]

Similar to Pandas, 🤗 Datasets provides several functions to manipulate the contents of `Dataset` and `DatasetDict` objects. We already encountered the `Dataset.map()` method in [Chapter 3](/course/chapter3), and in this section we'll explore some of the other functions at our disposal.

For this example we'll use the [Drug Review Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29) that's hosted on the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), which contains patient reviews on various drugs, along with the condition being treated and a 10-star rating of the patient's satisfaction.

First we need to download and extract the data, which can be done with the `wget` and `unzip` commands:

```py
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip drugsCom_raw.zip
```

Since TSV is just a variant of CSV that uses tabs instead of commas as the separator, we can load these files by using the `csv` loading script and specifying the `delimiter` argument in the `load_dataset()` function as follows:

```py
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
```

A good practice when doing any sort of data analysis is to grab a small random sample to get a quick feel for the type of data you're working with. In 🤗 Datasets, we can create a random sample by chaining the `Dataset.shuffle()` and `Dataset.select()` functions together:

```py
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]
```

```python out
{
    "Unnamed: 0": [87571, 178045, 80482],
    "drugName": ["Naproxen", "Duloxetine", "Mobic"],
    "condition": ["Gout, Acute", "ibromyalgia", "Inflammatory Conditions"],
    "review": [
        '"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
        '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
        '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."',
    ],
    "rating": [9.0, 3.0, 10.0],
    "date": ["September 2, 2015", "November 7, 2011", "June 5, 2013"],
    "usefulCount": [36, 13, 128],
}
```

Note that we've fixed the seed in `Dataset.shuffle()` for reproducibility purposes. `Dataset.select()` expects an iterable of indices, so we've passed `range(1000)` to grab the first 1,000 examples from the shuffled dataset. From this sample we can already see a few quirks in our dataset:

* The `Unnamed: 0` column looks suspiciously like an anonymized ID for each patient.
* The `condition` column includes a mix of uppercase and lowercase labels.
* The reviews are of varying length and contain a mix of Python line separators (`\r\n`) as well as HTML character codes like `&\#039;`.

Let's see how we can use 🤗 Datasets to deal with each of these issues. To test the patient ID hypothesis for the `Unnamed: 0` column, we can use the `Dataset.unique()` function to verify that the number of IDs matches the number of rows in each split:

```py
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
```

This seems to confirm our hypothesis, so let's clean up the dataset a bit by renaming the `Unnamed: 0` column to something a bit more interpretable. We can use the `DatasetDict.rename_column()` function to rename the column across both splits in one go:

```py
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
drug_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
```

> [!TIP]
> ✏️ **Try it out!** Use the `Dataset.unique()` function to find the number of unique drugs and conditions in the training and test sets.

Next, let's normalize all the `condition` labels using `Dataset.map()`. As we did with tokenization in [Chapter 3](/course/chapter3), we can define a simple function that can be applied across all the rows of each split in `drug_dataset`:

```py
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset.map(lowercase_condition)
```

```python out
AttributeError: 'NoneType' object has no attribute 'lower'
```

Oh no, we've run into a problem with our map function! From the error we can infer that some of the entries in the `condition` column are `None`, which cannot be lowercased as they're not strings. Let's drop these rows using `Dataset.filter()`, which works in a similar way to `Dataset.map()` and expects a function that receives a single example of the dataset. Instead of writing an explicit function like:

```py
def filter_nones(x):
    return x["condition"] is not None
```

and then running `drug_dataset.filter(filter_nones)`, we can do this in one line using a _lambda function_. In Python, lambda functions are small functions that you can define without explicitly naming them. They take the general form:

```
lambda <arguments> : <expression>
```

where `lambda` is one of Python's special [keywords](https://docs.python.org/3/reference/lexical_analysis.html#keywords), `<arguments>` is a list/set of comma-separated values that define the inputs to the function, and `<expression>` represents the operations you wish to execute. For example, we can define a simple lambda function that squares a number as follows:

```
lambda x : x * x
```

To apply this function to an input, we need to wrap it and the input in parentheses:

```py
(lambda x: x * x)(3)
```

```python out
9
```

Similarly, we can define lambda functions with multiple arguments by separating them with commas. For example, we can compute the area of a triangle as follows:

```py
(lambda base, height: 0.5 * base * height)(4, 8)
```

```python out
16.0
```

Lambda functions are handy when you want to define small, single-use functions (for more information about them, we recommend reading the excellent [Real Python tutorial](https://realpython.com/python-lambda/) by Andre Burgaud). In the 🤗 Datasets context, we can use lambda functions to define simple map and filter operations, so let's use this trick to eliminate the `None` entries in our dataset:

```py
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
```

With the `None` entries removed, we can normalize our `condition` column:

```py
drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
drug_dataset["train"]["condition"][:3]
```

```python out
["left ventricular dysfunction", "adhd", "birth control"]
```

It works! Now that we've cleaned up the labels, let's take a look at cleaning up the reviews themselves.

## Creating new columns[[creating-new-columns]]

Whenever you're dealing with customer reviews, a good practice is to check the number of words in each review. A review might be just a single word like "Great!" or a full-blown essay with thousands of words, and depending on the use case you'll need to handle these extremes differently. To compute the number of words in each review, we'll use a rough heuristic based on splitting each text by whitespace.

Let's define a simple function that counts the number of words in each review:

```py
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
```

Unlike our `lowercase_condition()` function, `compute_review_length()` returns a dictionary whose key does not correspond to one of the column names in the dataset. In this case, when `compute_review_length()` is passed to `Dataset.map()`, it will be applied to all the rows in the dataset to create a new `review_length` column:

```py
drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
drug_dataset["train"][0]
```

```python out
{
    "patient_id": 206461,
    "drugName": "Valsartan",
    "condition": "left ventricular dysfunction",
    "review": '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
    "rating": 9.0,
    "date": "May 20, 2012",
    "usefulCount": 27,
    "review_length": 17,
}
```

As expected, we can see a `review_length` column has been added to our training set. We can sort this new column with `Dataset.sort()` to see what the extreme values look like:

```py
drug_dataset["train"].sort("review_length")[:3]
```

```python out
{
    "patient_id": [103488, 23627, 20558],
    "drugName": ["Loestrin 21 1 / 20", "Chlorzoxazone", "Nucynta"],
    "condition": ["birth control", "muscle spasm", "pain"],
    "review": ['"Excellent."', '"useless"', '"ok"'],
    "rating": [10.0, 1.0, 6.0],
    "date": ["November 4, 2008", "March 24, 2017", "August 20, 2016"],
    "usefulCount": [5, 2, 10],
    "review_length": [1, 1, 1],
}
```

As we suspected, some reviews contain just a single word, which, although it may be okay for sentiment analysis, would not be informative if we want to predict the condition.

> [!TIP]
> 🙋 An alternative way to add new columns to a dataset is with the `Dataset.add_column()` function. This allows you to provide the column as a Python list or NumPy array and can be handy in situations where `Dataset.map()` is not well suited for your analysis.

Let's use the `Dataset.filter()` function to remove reviews that contain fewer than 30 words. Similarly to what we did with the `condition` column, we can filter out the very short reviews by requiring that the reviews have a length above this threshold:

```py
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)
```

```python out
{"train": 138514, "test": 46108}
```

As you can see, this has removed around 15% of the reviews from our original training and test sets.

> [!TIP]
> ✏️ **Try it out!** Use the `Dataset.sort()` function to inspect the reviews with the largest numbers of words. See the [documentation](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.sort) to see which argument you need to use sort the reviews by length in descending order.

The last thing we need to deal with is the presence of HTML character codes in our reviews. We can use Python's `html` module to unescape these characters, like so:

```py
import html

text = "I&#039;m a transformer called BERT"
html.unescape(text)
```

```python out
"I'm a transformer called BERT"
```

We'll use `Dataset.map()` to unescape all the HTML characters in our corpus:

```python
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
```

As you can see, the `Dataset.map()` method is quite useful for processing data -- and we haven't even scratched the surface of everything it can do!

## The `map()` method's superpowers[[the-map-methods-superpowers]]

The `Dataset.map()` method takes a `batched` argument that, if set to `True`, causes it to send a batch of examples to the map function at once (the batch size is configurable but defaults to 1,000). For instance, the previous map function that unescaped all the HTML took a bit of time to run (you can read the time taken from the progress bars). We can speed this up by processing several elements at the same time using a list comprehension.

When you specify `batched=True` the function receives a dictionary with the fields of the dataset, but each value is now a _list of values_, and not just a single value. The return value of `Dataset.map()` should be the same: a dictionary with the fields we want to update or add to our dataset, and a list of values. For example, here is another way to unescape all HTML characters, but using `batched=True`:

```python
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
```

If you're running this code in a notebook, you'll see that this command executes way faster than the previous one. And it's not because our reviews have already been HTML-unescaped -- if you re-execute the instruction from the previous section (without `batched=True`), it will take the same amount of time as before. This is because list comprehensions are usually faster than executing the same code in a `for` loop, and we also gain some performance by accessing lots of elements at the same time instead of one by one.

Using `Dataset.map()` with `batched=True` will be essential to unlock the speed of the "fast" tokenizers that we'll encounter in [Chapter 6](/course/chapter6), which can quickly tokenize big lists of texts. For instance, to tokenize all the drug reviews with a fast tokenizer, we could use a function like this:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)
```

As you saw in [Chapter 3](/course/chapter3), we can pass one or several examples to the tokenizer, so we can use this function with or without `batched=True`. Let's take this opportunity to compare the performance of the different options. In a notebook, you can time a one-line instruction by adding `%time` before the line of code you wish to measure:

```python no-format
%time tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
```

You can also time a whole cell by putting `%%time` at the beginning of the cell. On the hardware we executed this on, it showed 10.8s for this instruction (it's the number written after "Wall time").

> [!TIP]
> ✏️ **Try it out!** Execute the same instruction with and without `batched=True`, then try it with a slow tokenizer (add `use_fast=False` in the `AutoTokenizer.from_pretrained()` method) so you can see what numbers you get on your hardware.

Here are the results we obtained with and without batching, with a fast and a slow tokenizer:

Options         | Fast tokenizer | Slow tokenizer
:--------------:|:--------------:|:-------------:
`batched=True`  | 10.8s          | 4min41s
`batched=False` | 59.2s          | 5min3s

This means that using a fast tokenizer with the `batched=True` option is 30 times faster than its slow counterpart with no batching -- this is truly amazing! That's the main reason why fast tokenizers are the default when using `AutoTokenizer` (and why they are called "fast"). They're able to achieve such a speedup because behind the scenes the tokenization code is executed in Rust, which is a language that makes it easy to parallelize code execution.

Parallelization is also the reason for the nearly 6x speedup the fast tokenizer achieves with batching: you can't parallelize a single tokenization operation, but when you want to tokenize lots of texts at the same time you can just split the execution across several processes, each responsible for its own texts.

`Dataset.map()` also has some parallelization capabilities of its own. Since they are not backed by Rust, they won't let a slow tokenizer catch up with a fast one, but they can still be helpful (especially if you're using a tokenizer that doesn't have a fast version). To enable multiprocessing, use the `num_proc` argument and specify the number of processes to use in your call to `Dataset.map()`:

```py
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
```

You can experiment a little with timing to determine the optimal number of processes to use; in our case 8 seemed to produce the best speed gain. Here are the numbers we got with and without multiprocessing:

Options         | Fast tokenizer | Slow tokenizer
:--------------:|:--------------:|:-------------:
`batched=True`  | 10.8s          | 4min41s
`batched=False` | 59.2s          | 5min3s
`batched=True`, `num_proc=8`  | 6.52s          | 41.3s
`batched=False`, `num_proc=8` | 9.49s          | 45.2s

Those are much more reasonable results for the slow tokenizer, but the performance of the fast tokenizer was also substantially improved. Note, however, that won't always be the case -- for values of `num_proc` other than 8, our tests showed that it was faster to use `batched=True` without that option. In general, we don't recommend using Python multiprocessing for fast tokenizers with `batched=True`.

> [!TIP]
> Using `num_proc` to speed up your processing is usually a great idea, as long as the function you are using is not already doing some kind of multiprocessing of its own.

All of this functionality condensed into a single method is already pretty amazing, but there's more! With `Dataset.map()` and `batched=True` you can change the number of elements in your dataset. This is super useful in many situations where you want to create several training features from one example, and we will need to do this as part of the preprocessing for several of the NLP tasks we'll undertake in [Chapter 7](/course/chapter7).

> [!TIP]
> 💡 In machine learning, an _example_ is usually defined as the set of _features_ that we feed to the model. In some contexts, these features will be the set of columns in a `Dataset`, but in others (like here and for question answering), multiple features can be extracted from a single example and belong to a single column.

Let's have a look at how it works! Here we will tokenize our examples and truncate them to a maximum length of 128, but we will ask the tokenizer to return *all* the chunks of the texts instead of just the first one. This can be done with `return_overflowing_tokens=True`:

```py
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
```

Let's test this on one example before using `Dataset.map()` on the whole dataset:

```py
result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]
```

```python out
[128, 49]
```

So, our first example in the training set became two features because it was tokenized to more than the maximum number of tokens we specified: the first one of length 128 and the second one of length 49. Now let's do this for all elements of the dataset!

```py
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
```

```python out
ArrowInvalid: Column 1 named condition expected length 1463 but got length 1000
```

Oh no! That didn't work! Why not? Looking at the error message will give us a clue: there is a mismatch in the lengths of one of the columns, one being of length 1,463 and the other of length 1,000. If you've looked at the `Dataset.map()` [documentation](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map), you may recall that it's the number of samples passed to the function that we are mapping; here those 1,000 examples gave 1,463 new features, resulting in a shape error.

The problem is that we're trying to mix two different datasets of different sizes: the `drug_dataset` columns will have a certain number of examples (the 1,000 in our error), but the `tokenized_dataset` we are building will have more (the 1,463 in the error message; it is more than 1,000 because we are tokenizing long reviews into more than one example by using `return_overflowing_tokens=True`). That doesn't work for a `Dataset`, so we need to either remove the columns from the old dataset or make them the same size as they are in the new dataset. We can do the former with the `remove_columns` argument:

```py
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
```

Now this works without error. We can check that our new dataset has many more elements than the original dataset by comparing the lengths:

```py
len(tokenized_dataset["train"]), len(drug_dataset["train"])
```

```python out
(206772, 138514)
```

We mentioned that we can also deal with the mismatched length problem by making the old columns the same size as the new ones. To do this, we will need the `overflow_to_sample_mapping` field the tokenizer returns when we set `return_overflowing_tokens=True`. It gives us a mapping from a new feature index to the index of the sample it originated from. Using this, we can associate each key present in our original dataset with a list of values of the right size by repeating the values of each example as many times as it generates new features:

```py
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result
```

We can see it works with `Dataset.map()` without us needing to remove the old columns:

```py
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset
```

```python out
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 206772
    })
    test: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 68876
    })
})
```

We get the same number of training features as before, but here we've kept all the old fields. If you need them for some post-processing after applying your model, you might want to use this approach.

You've now seen how 🤗 Datasets can be used to preprocess a dataset in various ways. Although the processing functions of 🤗 Datasets will cover most of your model training needs,
there may be times when you'll need to switch to Pandas to access more powerful features, like `DataFrame.groupby()` or high-level APIs for visualization. Fortunately, 🤗 Datasets is designed to be interoperable with libraries such as Pandas, NumPy, PyTorch, TensorFlow, and JAX. Let's take a look at how this works.

## From `Dataset`s to `DataFrame`s and back[[from-datasets-to-dataframes-and-back]]

<Youtube id="tfcY1067A5Q"/>

To enable the conversion between various third-party libraries, 🤗 Datasets provides a `Dataset.set_format()` function. This function only changes the _output format_ of the dataset, so you can easily switch to another format without affecting the underlying _data format_, which is Apache Arrow. The formatting is done in place. To demonstrate, let's convert our dataset to Pandas:

```py
drug_dataset.set_format("pandas")
```

Now when we access elements of the dataset we get a `pandas.DataFrame` instead of a dictionary:

```py
drug_dataset["train"][:3]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>drugName</th>
      <th>condition</th>
      <th>review</th>
      <th>rating</th>
      <th>date</th>
      <th>usefulCount</th>
      <th>review_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>95260</td>
      <td>Guanfacine</td>
      <td>adhd</td>
      <td>"My son is halfway through his fourth week of Intuniv..."</td>
      <td>8.0</td>
      <td>April 27, 2010</td>
      <td>192</td>
      <td>141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>92703</td>
      <td>Lybrel</td>
      <td>birth control</td>
      <td>"I used to take another oral contraceptive, which had 21 pill cycle, and was very happy- very light periods, max 5 days, no other side effects..."</td>
      <td>5.0</td>
      <td>December 14, 2009</td>
      <td>17</td>
      <td>134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138000</td>
      <td>Ortho Evra</td>
      <td>birth control</td>
      <td>"This is my first time using any form of birth control..."</td>
      <td>8.0</td>
      <td>November 3, 2015</td>
      <td>10</td>
      <td>89</td>
    </tr>
  </tbody>
</table>

Let's create a `pandas.DataFrame` for the whole training set by selecting all the elements of `drug_dataset["train"]`:

```py
train_df = drug_dataset["train"][:]
```

> [!TIP]
> 🚨 Under the hood, `Dataset.set_format()` changes the return format for the dataset's `__getitem__()` dunder method. This means that when we want to create a new object like `train_df` from a `Dataset` in the `"pandas"` format, we need to slice the whole dataset to obtain a `pandas.DataFrame`. You can verify for yourself that the type of `drug_dataset["train"]` is `Dataset`, irrespective of the output format.


From here we can use all the Pandas functionality that we want. For example, we can do fancy chaining to compute the class distribution among the `condition` entries:

```py
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)
frequencies.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>birth control</td>
      <td>27655</td>
    </tr>
    <tr>
      <th>1</th>
      <td>depression</td>
      <td>8023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>acne</td>
      <td>5209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>anxiety</td>
      <td>4991</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pain</td>
      <td>4744</td>
    </tr>
  </tbody>
</table>


And once we're done with our Pandas analysis, we can always create a new `Dataset` object by using the `Dataset.from_pandas()` function as follows:


```py
from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
freq_dataset
```

```python out
Dataset({features: ["condition", "frequency"], num_rows: 819})
```

> [!TIP]
> ✏️ **Try it out!** Compute the average rating per drug and store the result in a new `Dataset`.

This wraps up our tour of the various preprocessing techniques available in 🤗 Datasets. To round out the section, let's create a validation set to prepare the dataset for training a classifier on. Before doing so, we'll reset the output format of `drug_dataset` from `"pandas"` to `"arrow"`:

```python
drug_dataset.reset_format()
```

## Creating a validation set[[creating-a-validation-set]]

Although we have a test set we could use for evaluation, it's a good practice to leave the test set untouched and create a separate validation set during development. Once you are happy with the performance of your models on the validation set, you can do a final sanity check on the test set. This process helps mitigate the risk that you'll overfit to the test set and deploy a model that fails on real-world data.

🤗 Datasets provides a `Dataset.train_test_split()` function that is based on the famous functionality from `scikit-learn`. Let's use it to split our training set into `train` and `validation` splits (we set the `seed` argument for reproducibility):

```py
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean
```

```python out
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 46108
    })
})
```

Great, we've now prepared a dataset that's ready for training some models on! In [section 5](/course/chapter5/5) we'll show you how to upload datasets to the Hugging Face Hub, but for now let's cap off our analysis by looking at a few ways you can save datasets on your local machine.

## Saving a dataset[[saving-a-dataset]]

<Youtube id="blF9uxYcKHo"/>

Although 🤗 Datasets will cache every downloaded dataset and the operations performed on it, there are times when you'll want to save a dataset to disk (e.g., in case the cache gets deleted). As shown in the table below, 🤗 Datasets provides three main functions to save your dataset in different formats:

| Data format |        Function        |
| :---------: | :--------------------: |
|    Arrow    | `Dataset.save_to_disk()` |
|     CSV     |    `Dataset.to_csv()`    |
|    JSON     |   `Dataset.to_json()`    |

For example, let's save our cleaned dataset in the Arrow format:

```py
drug_dataset_clean.save_to_disk("drug-reviews")
```

This will create a directory with the following structure:

```
drug-reviews/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
```

where we can see that each split is associated with its own *dataset.arrow* table, and some metadata in *dataset_info.json* and *state.json*. You can think of the Arrow format as a fancy table of columns and rows that is optimized for building high-performance applications that process and transport large datasets.

Once the dataset is saved, we can load it by using the `load_from_disk()` function as follows:

```py
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
drug_dataset_reloaded
```

```python out
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 46108
    })
})
```

For the CSV and JSON formats, we have to store each split as a separate file. One way to do this is by iterating over the keys and values in the `DatasetDict` object:

```py
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
```

This saves each split in [JSON Lines format](https://jsonlines.org), where each row in the dataset is stored as a single line of JSON. Here's what the first example looks like:

```py
!head -n 1 drug-reviews-train.jsonl
```

```python out
{
    "patient_id": 141780,
    "drugName": "Escitalopram",
    "condition": "depression",
    "review": '"I seemed to experience the regular side effects of LEXAPRO, insomnia, low sex drive, sleepiness during the day. I am taking it at night because my doctor said if it made me tired to take it at night. I assumed it would and started out taking it at night. Strange dreams, some pleasant. I was diagnosed with fibromyalgia. Seems to be helping with the pain. Have had anxiety and depression in my family, and have tried quite a few other medications that haven\'t worked. Only have been on it for two weeks but feel more positive in my mind, want to accomplish more in my life. Hopefully the side effects will dwindle away, worth it to stick with it from hearing others responses. Great medication."',
    "rating": 9.0,
    "date": "May 29, 2011",
    "usefulCount": 10,
    "review_length": 125,
}
```

We can then use the techniques from [section 2](/course/chapter5/2) to load the JSON files as follows:

```py
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)
```

And that's it for our excursion into data wrangling with 🤗 Datasets! Now that we have a cleaned dataset for training a model on, here are a few ideas that you could try out:

1. Use the techniques from [Chapter 3](/course/chapter3) to train a classifier that can predict the patient condition based on the drug review.
2. Use the `summarization` pipeline from [Chapter 1](/course/chapter1) to generate summaries of the reviews.

Next, we'll take a look at how 🤗 Datasets can enable you to work with huge datasets without blowing up your laptop!


---

<!-- Section 5.4 -->

# Big data? 🤗 Datasets to the rescue![[big-data-datasets-to-the-rescue]]

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section4.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section4.ipynb"},
]} />


Nowadays it is not uncommon to find yourself working with multi-gigabyte datasets, especially if you're planning to pretrain a transformer like BERT or GPT-2 from scratch. In these cases, even _loading_ the data can be a challenge. For example, the WebText corpus used to pretrain GPT-2 consists of over 8 million documents and 40 GB of text -- loading this into your laptop's RAM is likely to give it a heart attack!

Fortunately, 🤗 Datasets has been designed to overcome these limitations. It frees you from memory management problems by treating datasets as _memory-mapped_ files, and from hard drive limits by _streaming_ the entries in a corpus.

<Youtube id="JwISwTCPPWo"/>

In this section we'll explore these features of 🤗 Datasets with a huge 825 GB corpus known as [the Pile](https://pile.eleuther.ai). Let's get started!

## What is the Pile?[[what-is-the-pile]]

The Pile is an English text corpus that was created by [EleutherAI](https://www.eleuther.ai) for training large-scale language models. It includes a diverse range of datasets, spanning scientific articles, GitHub code repositories, and filtered web text. The training corpus is available in [14 GB chunks](https://the-eye.eu/public/AI/pile/), and you can also download several of the [individual components](https://the-eye.eu/public/AI/pile_preliminary_components/). Let's start by taking a look at the PubMed Abstracts dataset, which is a corpus of abstracts from 15 million biomedical publications on [PubMed](https://pubmed.ncbi.nlm.nih.gov/). The dataset is in [JSON Lines format](https://jsonlines.org) and is compressed using the `zstandard` library, so first we need to install that:

```py
!pip install zstandard
```

Next, we can load the dataset using the method for remote files that we learned in [section 2](/course/chapter5/2):

```py
from datasets import load_dataset

# This takes a few minutes to run, so go grab a tea or coffee while you wait :)
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
pubmed_dataset
```

```python out
Dataset({features: ["meta", "text"], num_rows: 15518009})
```

We can see that there are 15,518,009 rows and 2 columns in our dataset -- that's a lot!

> [!TIP]
> ✎ By default, 🤗 Datasets will decompress the files needed to load a dataset. If you want to preserve hard drive space, you can pass `DownloadConfig(delete_extracted=True)` to the `download_config` argument of `load_dataset()`. See the [documentation](https://huggingface.co/docs/datasets/package_reference/builder_classes#datasets.DownloadConfig) for more details.

Let's inspect the contents of the first example:

```py
pubmed_dataset[0]
```

```python out
{
    "meta": {"pmid": 11409574, "language": "eng"},
    "text": "Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...",
}
```

Okay, this looks like the abstract from a medical article. Now let's see how much RAM we've used to load the dataset!

## The magic of memory mapping[[the-magic-of-memory-mapping]]

A simple way to measure memory usage in Python is with the [`psutil`](https://psutil.readthedocs.io/en/latest/) library, which can be installed with `pip` as follows:

```python
!pip install psutil
```

It provides a `Process` class that allows us to check the memory usage of the current process as follows:

```py
import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
```

```python out
RAM used: 5678.33 MB
```

Here the `rss` attribute refers to the _resident set size_, which is the fraction of memory that a process occupies in RAM. This measurement also includes the memory used by the Python interpreter and the libraries we've loaded, so the actual amount of memory used to load the dataset is a bit smaller. For comparison, let's see how large the dataset is on disk, using the `dataset_size` attribute. Since the result is expressed in bytes like before, we need to manually convert it to gigabytes:

```py
print(f"Dataset size in bytes: {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")
```

```python out
Dataset size in bytes : 20979437051
Dataset size (cache file) : 19.54 GB
```

Nice -- despite it being almost 20 GB large, we're able to load and access the dataset with much less RAM!

> [!TIP]
> ✏️ **Try it out!** Pick one of the [subsets](https://the-eye.eu/public/AI/pile_preliminary_components/) from the Pile that is larger than your laptop or desktop's RAM, load it with 🤗 Datasets, and measure the amount of RAM used. Note that to get an accurate measurement, you'll want to do this in a new process. You can find the decompressed sizes of each subset in Table 1 of [the Pile paper](https://arxiv.org/abs/2101.00027).

If you're familiar with Pandas, this result might come as a surprise because of Wes Kinney's famous [rule of thumb](https://wesmckinney.com/blog/apache-arrow-pandas-internals/) that you typically need 5 to 10 times as much RAM as the size of your dataset. So how does 🤗 Datasets solve this memory management problem? 🤗 Datasets treats each dataset as a [memory-mapped file](https://en.wikipedia.org/wiki/Memory-mapped_file), which provides a mapping between RAM and filesystem storage that allows the library to access and operate on elements of the dataset without needing to fully load it into memory.

Memory-mapped files can also be shared across multiple processes, which enables methods like `Dataset.map()` to be parallelized without needing to move or copy the dataset. Under the hood, these capabilities are all realized by the [Apache Arrow](https://arrow.apache.org) memory format and [`pyarrow`](https://arrow.apache.org/docs/python/index.html) library, which make the data loading and processing lightning fast. (For more details about Apache Arrow and comparisons to Pandas, check out [Dejan Simic's blog post](https://towardsdatascience.com/apache-arrow-read-dataframe-with-zero-memory-69634092b1a).) To see this in action, let's run a little speed test by iterating over all the elements in the PubMed Abstracts dataset:

```py
import timeit

code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb / time:.3f} GB/s"
)
```

```python out
"Iterated over 15518009 examples (about 19.5 GB) in 64.2s, i.e. 0.304 GB/s"
```

Here we've used Python's `timeit` module to measure the execution time taken by `code_snippet`. You'll typically be able to iterate over a dataset at speed of a few tenths of a GB/s to several GB/s. This works great for the vast majority of applications, but sometimes you'll have to work with a dataset that is too large to even store on your laptop's hard drive. For example, if we tried to download the Pile in its entirety, we'd need 825 GB of free disk space! To handle these cases, 🤗 Datasets provides a streaming feature that allows us to download and access elements on the fly, without needing to download the whole dataset. Let's take a look at how this works.

> [!TIP]
> 💡 In Jupyter notebooks you can also time cells using the [`%%timeit` magic function](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit).

## Streaming datasets[[streaming-datasets]]

To enable dataset streaming you just need to pass the `streaming=True` argument to the `load_dataset()` function. For example, let's load the PubMed Abstracts dataset again, but in streaming mode:

```py
pubmed_dataset_streamed = load_dataset("json", data_files=data_files, split="train", streaming=True)
```

Instead of the familiar `Dataset` that we've encountered elsewhere in this chapter, the object returned with `streaming=True` is an `IterableDataset`. As the name suggests, to access the elements of an `IterableDataset` we need to iterate over it. We can access the first element of our streamed dataset as follows:


```py
next(iter(pubmed_dataset_streamed))
```

```python out
{
    "meta": {"pmid": 11409574, "language": "eng"},
    "text": "Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...",
}
```

The elements from a streamed dataset can be processed on the fly using `IterableDataset.map()`, which is useful during training if you need to tokenize the inputs. The process is exactly the same as the one we used to tokenize our dataset in [Chapter 3](/course/chapter3), with the only difference being that outputs are returned one by one:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
next(iter(tokenized_dataset))
```

```python out
{"input_ids": [101, 4958, 5178, 4328, 6779, ...], "attention_mask": [1, 1, 1, 1, 1, ...]}
```

> [!TIP]
> 💡 To speed up tokenization with streaming you can pass `batched=True`, as we saw in the last section. It will process the examples batch by batch; the default batch size is 1,000 and can be specified with the `batch_size` argument.

You can also shuffle a streamed dataset using `IterableDataset.shuffle()`, but unlike `Dataset.shuffle()` this only shuffles the elements in a predefined `buffer_size`:

```py
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
next(iter(shuffled_dataset))
```

```python out
{
    "meta": {"pmid": 11410799, "language": "eng"},
    "text": "Randomized study of dose or schedule modification of granulocyte colony-stimulating factor in platinum-based chemotherapy for elderly patients with lung cancer ...",
}
```

In this example, we selected a random example from the first 10,000 examples in the buffer. Once an example is accessed, its spot in the buffer is filled with the next example in the corpus (i.e., the 10,001st example in the case above). You can also select elements from a streamed dataset using the `IterableDataset.take()` and `IterableDataset.skip()` functions, which act in a similar way to `Dataset.select()`. For example, to select the first 5 examples in the PubMed Abstracts dataset we can do the following:

```py
dataset_head = pubmed_dataset_streamed.take(5)
list(dataset_head)
```

```python out
[
    {
        "meta": {"pmid": 11409574, "language": "eng"},
        "text": "Epidemiology of hypoxaemia in children with acute lower respiratory infection ...",
    },
    {
        "meta": {"pmid": 11409575, "language": "eng"},
        "text": "Clinical signs of hypoxaemia in children with acute lower respiratory infection: indicators of oxygen therapy ...",
    },
    {
        "meta": {"pmid": 11409576, "language": "eng"},
        "text": "Hypoxaemia in children with severe pneumonia in Papua New Guinea ...",
    },
    {
        "meta": {"pmid": 11409577, "language": "eng"},
        "text": "Oxygen concentrators and cylinders ...",
    },
    {
        "meta": {"pmid": 11409578, "language": "eng"},
        "text": "Oxygen supply in rural africa: a personal experience ...",
    },
]
```

Similarly, you can use the `IterableDataset.skip()` function to create training and validation splits from a shuffled dataset as follows:

```py
# Skip the first 1,000 examples and include the rest in the training set
train_dataset = shuffled_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = shuffled_dataset.take(1000)
```

Let's round out our exploration of dataset streaming with a common application: combining multiple datasets together to create a single corpus. 🤗 Datasets provides an `interleave_datasets()` function that converts a list of `IterableDataset` objects into a single `IterableDataset`, where the elements of the new dataset are obtained by alternating among the source examples. This function is especially useful when you're trying to combine large datasets, so as an example let's stream the FreeLaw subset of the Pile, which is a 51 GB dataset of legal opinions from US courts:

```py
law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
next(iter(law_dataset_streamed))
```

```python out
{
    "meta": {
        "case_ID": "110921.json",
        "case_jurisdiction": "scotus.tar.gz",
        "date_created": "2010-04-28T17:12:49Z",
    },
    "text": "\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581.\nSupreme Court of United States.\nArgued January 19, 1983.\nDecided April 26, 1983.\nCERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT\n*239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General...",
}
```

This dataset is large enough to stress the RAM of most laptops, yet we've been able to load and access it without breaking a sweat! Let's now combine the examples from the FreeLaw and PubMed Abstracts datasets with the `interleave_datasets()` function:

```py
from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
list(islice(combined_dataset, 2))
```

```python out
[
    {
        "meta": {"pmid": 11409574, "language": "eng"},
        "text": "Epidemiology of hypoxaemia in children with acute lower respiratory infection ...",
    },
    {
        "meta": {
            "case_ID": "110921.json",
            "case_jurisdiction": "scotus.tar.gz",
            "date_created": "2010-04-28T17:12:49Z",
        },
        "text": "\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581.\nSupreme Court of United States.\nArgued January 19, 1983.\nDecided April 26, 1983.\nCERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT\n*239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General...",
    },
]
```

Here we've used the `islice()` function from Python's `itertools` module to select the first two examples from the combined dataset, and we can see that they match the first examples from each of the two source datasets.

Finally, if you want to stream the Pile in its 825 GB entirety, you can grab all the prepared files as follows:

```py
base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
next(iter(pile_dataset["train"]))
```

```python out
{
    "meta": {"pile_set_name": "Pile-CC"},
    "text": "It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web...",
}
```

> [!TIP]
> ✏️ **Try it out!** Use one of the large Common Crawl corpora like [`mc4`](https://huggingface.co/datasets/mc4) or [`oscar`](https://huggingface.co/datasets/oscar) to create a streaming multilingual dataset that represents the spoken proportions of languages in a country of your choice. For example, the four national languages in Switzerland are German, French, Italian, and Romansh, so you could try creating a Swiss corpus by sampling the Oscar subsets according to their spoken proportion.

You now have all the tools you need to load and process datasets of all shapes and sizes -- but unless you're exceptionally lucky, there will come a point in your NLP journey where you'll have to actually create a dataset to solve the problem at hand. That's the topic of the next section!


---

<!-- Section 5.5 -->

# Creating your own dataset[[creating-your-own-dataset]]

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section5.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section5.ipynb"},
]} />

Sometimes the dataset that you need to build an NLP application doesn't exist, so you'll need to create it yourself. In this section we'll show you how to create a corpus of [GitHub issues](https://github.com/features/issues/), which are commonly used to track bugs or features in GitHub repositories. This corpus could be used for various purposes, including:

* Exploring how long it takes to close open issues or pull requests
* Training a _multilabel classifier_ that can tag issues with metadata based on the issue's description (e.g., "bug," "enhancement," or "question")
* Creating a semantic search engine to find which issues match a user's query

Here we'll focus on creating the corpus, and in the next section we'll tackle the semantic search application. To keep things meta, we'll use the GitHub issues associated with a popular open source project: 🤗 Datasets! Let's take a look at how to get the data and explore the information contained in these issues.

## Getting the data[[getting-the-data]]

You can find all the issues in 🤗 Datasets by navigating to the repository's [Issues tab](https://github.com/huggingface/datasets/issues). As shown in the following screenshot, at the time of writing there were 331 open issues and 668 closed ones.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-issues.png" alt="The GitHub issues associated with 🤗 Datasets." width="80%"/>
</div>

If you click on one of these issues you'll find it contains a title, a description, and a set of labels that characterize the issue. An example is shown in the screenshot below.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-issues-single.png" alt="A typical GitHub issue in the 🤗 Datasets repository." width="80%"/>
</div>

To download all the repository's issues, we'll use the [GitHub REST API](https://docs.github.com/en/rest) to poll the [`Issues` endpoint](https://docs.github.com/en/rest/reference/issues#list-repository-issues). This endpoint returns a list of JSON objects, with each object containing a large number of fields that include the title and description as well as metadata about the status of the issue and so on.

A convenient way to download the issues is via the `requests` library, which is the standard way for making HTTP requests in Python. You can install the library by running:

```python
!pip install requests
```

Once the library is installed, you can make GET requests to the `Issues` endpoint by invoking the `requests.get()` function. For example, you can run the following command to retrieve the first issue on the first page:

```py
import requests

url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
```

The `response` object contains a lot of useful information about the request, including the HTTP status code:

```py
response.status_code
```

```python out
200
```

where a `200` status means the request was successful (you can find a list of possible HTTP status codes [here](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)). What we are really interested in, though, is the _payload_, which can be accessed in various formats like bytes, strings, or JSON. Since we know our issues are in JSON format, let's inspect the payload as follows:

```py
response.json()
```

```python out
[
    {
        "url": "https://api.github.com/repos/huggingface/datasets/issues/2792",
        "repository_url": "https://api.github.com/repos/huggingface/datasets",
        "labels_url": "https://api.github.com/repos/huggingface/datasets/issues/2792/labels{/name}",
        "comments_url": "https://api.github.com/repos/huggingface/datasets/issues/2792/comments",
        "events_url": "https://api.github.com/repos/huggingface/datasets/issues/2792/events",
        "html_url": "https://github.com/huggingface/datasets/pull/2792",
        "id": 968650274,
        "node_id": "MDExOlB1bGxSZXF1ZXN0NzEwNzUyMjc0",
        "number": 2792,
        "title": "Update GooAQ",
        "user": {
            "login": "bhavitvyamalik",
            "id": 19718818,
            "node_id": "MDQ6VXNlcjE5NzE4ODE4",
            "avatar_url": "https://avatars.githubusercontent.com/u/19718818?v=4",
            "gravatar_id": "",
            "url": "https://api.github.com/users/bhavitvyamalik",
            "html_url": "https://github.com/bhavitvyamalik",
            "followers_url": "https://api.github.com/users/bhavitvyamalik/followers",
            "following_url": "https://api.github.com/users/bhavitvyamalik/following{/other_user}",
            "gists_url": "https://api.github.com/users/bhavitvyamalik/gists{/gist_id}",
            "starred_url": "https://api.github.com/users/bhavitvyamalik/starred{/owner}{/repo}",
            "subscriptions_url": "https://api.github.com/users/bhavitvyamalik/subscriptions",
            "organizations_url": "https://api.github.com/users/bhavitvyamalik/orgs",
            "repos_url": "https://api.github.com/users/bhavitvyamalik/repos",
            "events_url": "https://api.github.com/users/bhavitvyamalik/events{/privacy}",
            "received_events_url": "https://api.github.com/users/bhavitvyamalik/received_events",
            "type": "User",
            "site_admin": False,
        },
        "labels": [],
        "state": "open",
        "locked": False,
        "assignee": None,
        "assignees": [],
        "milestone": None,
        "comments": 1,
        "created_at": "2021-08-12T11:40:18Z",
        "updated_at": "2021-08-12T12:31:17Z",
        "closed_at": None,
        "author_association": "CONTRIBUTOR",
        "active_lock_reason": None,
        "pull_request": {
            "url": "https://api.github.com/repos/huggingface/datasets/pulls/2792",
            "html_url": "https://github.com/huggingface/datasets/pull/2792",
            "diff_url": "https://github.com/huggingface/datasets/pull/2792.diff",
            "patch_url": "https://github.com/huggingface/datasets/pull/2792.patch",
        },
        "body": "[GooAQ](https://github.com/allenai/gooaq) dataset was recently updated after splits were added for the same. This PR contains new updated GooAQ with train/val/test splits and updated README as well.",
        "performed_via_github_app": None,
    }
]
```

Whoa, that's a lot of information! We can see useful fields like `title`, `body`, and `number` that describe the issue, as well as information about the GitHub user who opened the issue.

> [!TIP]
> ✏️ **Try it out!** Click on a few of the URLs in the JSON payload above to get a feel for what type of information each GitHub issue is linked to.

As described in the GitHub [documentation](https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting), unauthenticated requests are limited to 60 requests per hour. Although you can increase the `per_page` query parameter to reduce the number of requests you make, you will still hit the rate limit on any repository that has more than a few thousand issues. So instead, you should follow GitHub's [instructions](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) on creating a _personal access token_ so that you can boost the rate limit to 5,000 requests per hour. Once you have your token, you can include it as part of the request header:

```py
GITHUB_TOKEN = xxx  # Copy your GitHub token here
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
```

> [!WARNING]
> ⚠️ Do not share a notebook with your `GITHUB_TOKEN` pasted in it. We recommend you delete the last cell once you have executed it to avoid leaking this information accidentally. Even better, store the token in a *.env* file and use the [`python-dotenv` library](https://github.com/theskumar/python-dotenv) to load it automatically for you as an environment variable.

Now that we have our access token, let's create a function that can download all the issues from a GitHub repository:

```py
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm


def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )
```

Now when we call `fetch_issues()` it will download all the issues in batches to avoid exceeding GitHub's limit on the number of requests per hour; the result will be stored in a _repository_name-issues.jsonl_ file, where each line is a JSON object the represents an issue. Let's use this function to grab all the issues from 🤗 Datasets:

```py
# Depending on your internet connection, this can take several minutes to run...
fetch_issues()
```

Once the issues are downloaded we can load them locally using our newfound skills from [section 2](/course/chapter5/2):

```py
issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
issues_dataset
```

```python out
Dataset(
    {
        features: [
            "url",
            "repository_url",
            "labels_url",
            "comments_url",
            "events_url",
            "html_url",
            "id",
            "node_id",
            "number",
            "title",
            "user",
            "labels",
            "state",
            "locked",
            "assignee",
            "assignees",
            "milestone",
            "comments",
            "created_at",
            "updated_at",
            "closed_at",
            "author_association",
            "active_lock_reason",
            "pull_request",
            "body",
            "timeline_url",
            "performed_via_github_app",
        ],
        num_rows: 3019,
    }
)
```

Great, we've created our first dataset from scratch! But why are there several thousand issues when the [Issues tab](https://github.com/huggingface/datasets/issues) of the 🤗 Datasets repository only shows around 1,000 issues in total 🤔? As described in the GitHub [documentation](https://docs.github.com/en/rest/reference/issues#list-issues-assigned-to-the-authenticated-user), that's because we've downloaded all the pull requests as well:

> GitHub's REST API v3 considers every pull request an issue, but not every issue is a pull request. For this reason, "Issues" endpoints may return both issues and pull requests in the response. You can identify pull requests by the `pull_request` key. Be aware that the `id` of a pull request returned from "Issues" endpoints will be an issue id.

Since the contents of issues and pull requests are quite different, let's do some minor preprocessing to enable us to distinguish between them.

## Cleaning up the data[[cleaning-up-the-data]]

The above snippet from GitHub's documentation tells us that the `pull_request` column can be used to differentiate between issues and pull requests. Let's look at a random sample to see what the difference is. As we did in [section 3](/course/chapter5/3), we'll chain `Dataset.shuffle()` and `Dataset.select()` to create a random sample and then zip the `html_url` and `pull_request` columns so we can compare the various URLs:

```py
sample = issues_dataset.shuffle(seed=666).select(range(3))

# Print out the URL and pull request entries
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")
```

```python out
>> URL: https://github.com/huggingface/datasets/pull/850
>> Pull request: {'url': 'https://api.github.com/repos/huggingface/datasets/pulls/850', 'html_url': 'https://github.com/huggingface/datasets/pull/850', 'diff_url': 'https://github.com/huggingface/datasets/pull/850.diff', 'patch_url': 'https://github.com/huggingface/datasets/pull/850.patch'}

>> URL: https://github.com/huggingface/datasets/issues/2773
>> Pull request: None

>> URL: https://github.com/huggingface/datasets/pull/783
>> Pull request: {'url': 'https://api.github.com/repos/huggingface/datasets/pulls/783', 'html_url': 'https://github.com/huggingface/datasets/pull/783', 'diff_url': 'https://github.com/huggingface/datasets/pull/783.diff', 'patch_url': 'https://github.com/huggingface/datasets/pull/783.patch'}
```

Here we can see that each pull request is associated with various URLs, while ordinary issues have a `None` entry. We can use this distinction to create a new `is_pull_request` column that checks whether the `pull_request` field is `None` or not:

```py
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)
```

> [!TIP]
> ✏️ **Try it out!** Calculate the average time it takes to close issues in 🤗 Datasets. You may find the `Dataset.filter()` function useful to filter out the pull requests and open issues, and you can use the `Dataset.set_format()` function to convert the dataset to a `DataFrame` so you can easily manipulate the `created_at` and `closed_at` timestamps. For bonus points, calculate the average time it takes to close pull requests.

Although we could proceed to further clean up the dataset by dropping or renaming some columns, it is generally a good practice to keep the dataset as "raw" as possible at this stage so that it can be easily used in multiple applications.

Before we push our dataset to the Hugging Face Hub, let's deal with one thing that's missing from it: the comments associated with each issue and pull request. We'll add them next with -- you guessed it -- the GitHub REST API!

## Augmenting the dataset[[augmenting-the-dataset]]

As shown in the following screenshot, the comments associated with an issue or pull request provide a rich source of information, especially if we're interested in building a search engine to answer user queries about the library.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-issues-comment.png" alt="Comments associated with an issue about 🤗 Datasets." width="80%"/>
</div>

The GitHub REST API provides a [`Comments` endpoint](https://docs.github.com/en/rest/reference/issues#list-issue-comments) that returns all the comments associated with an issue number. Let's test the endpoint to see what it returns:

```py
issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)
response.json()
```

```python out
[
    {
        "url": "https://api.github.com/repos/huggingface/datasets/issues/comments/897594128",
        "html_url": "https://github.com/huggingface/datasets/pull/2792#issuecomment-897594128",
        "issue_url": "https://api.github.com/repos/huggingface/datasets/issues/2792",
        "id": 897594128,
        "node_id": "IC_kwDODunzps41gDMQ",
        "user": {
            "login": "bhavitvyamalik",
            "id": 19718818,
            "node_id": "MDQ6VXNlcjE5NzE4ODE4",
            "avatar_url": "https://avatars.githubusercontent.com/u/19718818?v=4",
            "gravatar_id": "",
            "url": "https://api.github.com/users/bhavitvyamalik",
            "html_url": "https://github.com/bhavitvyamalik",
            "followers_url": "https://api.github.com/users/bhavitvyamalik/followers",
            "following_url": "https://api.github.com/users/bhavitvyamalik/following{/other_user}",
            "gists_url": "https://api.github.com/users/bhavitvyamalik/gists{/gist_id}",
            "starred_url": "https://api.github.com/users/bhavitvyamalik/starred{/owner}{/repo}",
            "subscriptions_url": "https://api.github.com/users/bhavitvyamalik/subscriptions",
            "organizations_url": "https://api.github.com/users/bhavitvyamalik/orgs",
            "repos_url": "https://api.github.com/users/bhavitvyamalik/repos",
            "events_url": "https://api.github.com/users/bhavitvyamalik/events{/privacy}",
            "received_events_url": "https://api.github.com/users/bhavitvyamalik/received_events",
            "type": "User",
            "site_admin": False,
        },
        "created_at": "2021-08-12T12:21:52Z",
        "updated_at": "2021-08-12T12:31:17Z",
        "author_association": "CONTRIBUTOR",
        "body": "@albertvillanova my tests are failing here:\r\n```\r\ndataset_name = 'gooaq'\r\n\r\n    def test_load_dataset(self, dataset_name):\r\n        configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]\r\n>       self.dataset_tester.check_load_dataset(dataset_name, configs, is_local=True, use_local_dummy_data=True)\r\n\r\ntests/test_dataset_common.py:234: \r\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \r\ntests/test_dataset_common.py:187: in check_load_dataset\r\n    self.parent.assertTrue(len(dataset[split]) > 0)\r\nE   AssertionError: False is not true\r\n```\r\nWhen I try loading dataset on local machine it works fine. Any suggestions on how can I avoid this error?",
        "performed_via_github_app": None,
    }
]
```

We can see that the comment is stored in the `body` field, so let's write a simple function that returns all the comments associated with an issue by picking out the `body` contents for each element in `response.json()`:

```py
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


# Test our function works as expected
get_comments(2792)
```

```python out
[
    "@albertvillanova my tests are failing here:\r\n```\r\ndataset_name = 'gooaq'\r\n\r\n    def test_load_dataset(self, dataset_name):\r\n        configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]\r\n>       self.dataset_tester.check_load_dataset(dataset_name, configs, is_local=True, use_local_dummy_data=True)\r\n\r\ntests/test_dataset_common.py:234: \r\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \r\ntests/test_dataset_common.py:187: in check_load_dataset\r\n    self.parent.assertTrue(len(dataset[split]) > 0)\r\nE   AssertionError: False is not true\r\n```\r\nWhen I try loading dataset on local machine it works fine. Any suggestions on how can I avoid this error?"
]
```

This looks good, so let's use `Dataset.map()` to add a new `comments` column to each issue in our dataset:

```py
# Depending on your internet connection, this can take a few minutes...
issues_with_comments_dataset = issues_dataset.map(lambda x: {"comments": get_comments(x["number"])})
```

The final step is to push our dataset to the Hub. Let's take a look at how we can do that.

## Uploading the dataset to the Hugging Face Hub[[uploading-the-dataset-to-the-hugging-face-hub]]

<Youtube id="HaN6qCr_Afc"/>

Now that we have our augmented dataset, it's time to push it to the Hub so we can share it with the community! Uploading a dataset is very simple: just like models and tokenizers from 🤗 Transformers, we can use a `push_to_hub()` method to push a dataset. To do that we need an authentication token, which can be obtained by first logging into the Hugging Face Hub with the `notebook_login()` function:

```py
from huggingface_hub import notebook_login

notebook_login()
```

This will create a widget where you can enter your username and password, and an API token will be saved in *~/.huggingface/token*. If you're running the code in a terminal, you can log in via the CLI instead:

```bash
huggingface-cli login
```

Once we've done this, we can upload our dataset by running:

```py
issues_with_comments_dataset.push_to_hub("github-issues")
```

From here, anyone can download the dataset by simply providing `load_dataset()` with the repository ID as the `path` argument:

```py
remote_dataset = load_dataset("lewtun/github-issues", split="train")
remote_dataset
```

```python out
Dataset(
    {
        features: [
            "url",
            "repository_url",
            "labels_url",
            "comments_url",
            "events_url",
            "html_url",
            "id",
            "node_id",
            "number",
            "title",
            "user",
            "labels",
            "state",
            "locked",
            "assignee",
            "assignees",
            "milestone",
            "comments",
            "created_at",
            "updated_at",
            "closed_at",
            "author_association",
            "active_lock_reason",
            "pull_request",
            "body",
            "performed_via_github_app",
            "is_pull_request",
        ],
        num_rows: 2855,
    }
)
```

Cool, we've pushed our dataset to the Hub and it's available for others to use! There's just one important thing left to do: adding a _dataset card_ that explains how the corpus was created and provides other useful information for the community.

> [!TIP]
> 💡 You can also upload a dataset to the Hugging Face Hub directly from the terminal by using `huggingface-cli` and a bit of Git magic. See the [🤗 Datasets guide](https://huggingface.co/docs/datasets/share#share-a-dataset-using-the-cli) for details on how to do this.

## Creating a dataset card[[creating-a-dataset-card]]

Well-documented datasets are more likely to be useful to others (including your future self!), as they provide the context to enable users to decide whether the dataset is relevant to their task and to evaluate any potential biases in or risks associated with using the dataset.

On the Hugging Face Hub, this information is stored in each dataset repository's *README.md* file. There are two main steps you should take before creating this file:

1. Use the [`datasets-tagging` application](https://huggingface.co/datasets/tagging/) to create metadata tags in YAML format. These tags are used for a variety of search features on the Hugging Face Hub and ensure your dataset can be easily found by members of the community. Since we have created a custom dataset here, you'll need to clone the `datasets-tagging` repository and run the application locally. Here's what the interface looks like:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/datasets-tagger.png" alt="The `datasets-tagging` interface." width="80%"/>
</div>

2. Read the [🤗 Datasets guide](https://github.com/huggingface/datasets/blob/master/templates/README_guide.md) on creating informative dataset cards and use it as a template.

You can create the *README.md* file directly on the Hub, and you can find a template dataset card in the `lewtun/github-issues` dataset repository. A screenshot of the filled-out dataset card is shown below.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/dataset-card.png" alt="A dataset card." width="80%"/>
</div>

> [!TIP]
> ✏️ **Try it out!** Use the `dataset-tagging` application and [🤗 Datasets guide](https://github.com/huggingface/datasets/blob/master/templates/README_guide.md) to complete the *README.md* file for your GitHub issues dataset.

That's it! We've seen in this section that creating a good dataset can be quite involved, but fortunately uploading it and sharing it with the community is not. In the next section we'll use our new dataset to create a semantic search engine with 🤗 Datasets that can match questions to the most relevant issues and comments.

> [!TIP]
> ✏️ **Try it out!** Go through the steps we took in this section to create a dataset of GitHub issues for your favorite open source library (pick something other than 🤗 Datasets, of course!). For bonus points, fine-tune a multilabel classifier to predict the tags present in the `labels` field.




---

<!-- Section 5.6 -->

<FrameworkSwitchCourse {fw} />

# Semantic search with FAISS[[semantic-search-with-faiss]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_tf.ipynb"},
]} />

{/if}

In [section 5](/course/chapter5/5), we created a dataset of GitHub issues and comments from the 🤗 Datasets repository. In this section we'll use this information to build a search engine that can help us find answers to our most pressing questions about the library!

<Youtube id="OATCgQtNX2o"/>

## Using embeddings for semantic search[[using-embeddings-for-semantic-search]]

As we saw in [Chapter 1](/course/chapter1), Transformer-based language models represent each token in a span of text as an _embedding vector_. It turns out that one can "pool" the individual embeddings to create a vector representation for whole sentences, paragraphs, or (in some cases) documents. These embeddings can then be used to find similar documents in the corpus by computing the dot-product similarity (or some other similarity metric) between each embedding and returning the documents with the greatest overlap.

In this section we'll use embeddings to develop a semantic search engine. These search engines offer several advantages over conventional approaches that are based on matching keywords in a query with the documents.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/semantic-search.svg" alt="Semantic search."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/semantic-search-dark.svg" alt="Semantic search."/>
</div>

## Loading and preparing the dataset[[loading-and-preparing-the-dataset]]

The first thing we need to do is download our dataset of GitHub issues, so let's use `load_dataset()` function as usual:

```py
from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset
```

```python out
Dataset(
    {
        features: [
            "url",
            "repository_url",
            "labels_url",
            "comments_url",
            "events_url",
            "html_url",
            "id",
            "node_id",
            "number",
            "title",
            "user",
            "labels",
            "state",
            "locked",
            "assignee",
            "assignees",
            "milestone",
            "comments",
            "created_at",
            "updated_at",
            "closed_at",
            "author_association",
            "active_lock_reason",
            "pull_request",
            "body",
            "performed_via_github_app",
            "is_pull_request",
        ],
        num_rows: 2855,
    }
)
```

Here we've specified the default `train` split in `load_dataset()`, so it returns a `Dataset` instead of a `DatasetDict`. The first order of business is to filter out the pull requests, as these tend to be rarely used for answering user queries and will introduce noise in our search engine. As should be familiar by now, we can use the `Dataset.filter()` function to exclude these rows in our dataset. While we're at it, let's also filter out rows with no comments, since these provide no answers to user queries:

```py
issues_dataset = issues_dataset.filter(
    lambda x: x["is_pull_request"] == False and len(x["comments"]) > 0
)
issues_dataset
```

```python out
Dataset(
    {
        features: [
            "url",
            "repository_url",
            "labels_url",
            "comments_url",
            "events_url",
            "html_url",
            "id",
            "node_id",
            "number",
            "title",
            "user",
            "labels",
            "state",
            "locked",
            "assignee",
            "assignees",
            "milestone",
            "comments",
            "created_at",
            "updated_at",
            "closed_at",
            "author_association",
            "active_lock_reason",
            "pull_request",
            "body",
            "performed_via_github_app",
            "is_pull_request",
        ],
        num_rows: 771,
    }
)
```

We can see that there are a lot of columns in our dataset, most of which we don't need to build our search engine. From a search perspective, the most informative columns are `title`, `body`, and `comments`, while `html_url` provides us with a link back to the source issue. Let's use the `Dataset.remove_columns()` function to drop the rest:

```py
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset
```

```python out
Dataset({features: ["html_url", "title", "comments", "body"], num_rows: 771})
```

To create our embeddings we'll augment each comment with the issue's title and body, since these fields often include useful contextual information. Because our `comments` column is currently a list of comments for each issue, we need to "explode" the column so that each row consists of an `(html_url, title, body, comment)` tuple. In Pandas we can do this with the [`DataFrame.explode()` function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html), which creates a new row for each element in a list-like column, while replicating all the other column values. To see this in action, let's first switch to the Pandas  `DataFrame` format:

```py
issues_dataset.set_format("pandas")
df = issues_dataset[:]
```

If we inspect the first row in this `DataFrame` we can see there are four comments associated with this issue:

```py
df["comments"][0].tolist()
```

```python out
[
    'the bug code locate in ：\r\n    if data_args.task_name is not None:\r\n        # Downloading and loading a dataset from the hub.\r\n        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)',
    "Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com\r\n\r\nNormally, it should work if you wait a little and then retry.\r\n\r\nCould you please confirm if the problem persists?",
    "cannot connect，even by Web browser，please check that  there is some  problems。",
    "I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...",
]
```

When we explode `df`, we expect to get one row for each of these comments. Let's check if that's the case:

```py
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)
```

<table border="1" class="dataframe" style="table-layout: fixed; word-wrap:break-word; width: 100%;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>html_url</th>
      <th>title</th>
      <th>comments</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>the bug code locate in ：\r\n    if data_args.task_name is not None...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>cannot connect，even by Web browser，please check that  there is some  problems。</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
  </tbody>
</table>

Great, we can see the rows have been replicated, with the `comments` column containing the individual comments! Now that we're finished with Pandas, we can quickly switch back to a `Dataset` by loading the `DataFrame` in memory:

```py
from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset
```

```python out
Dataset({features: ["html_url", "title", "comments", "body"], num_rows: 2842})
```

Okay, this has given us a few thousand comments to work with!


> [!TIP]
> ✏️ **Try it out!** See if you can use `Dataset.map()` to explode the `comments` column of `issues_dataset` _without_ resorting to the use of Pandas. This is a little tricky; you might find the ["Batch mapping"](https://huggingface.co/docs/datasets/about_map_batch#batch-mapping) section of the 🤗 Datasets documentation useful for this task.

Now that we have one comment per row, let's create a new `comments_length` column that contains the number of words per comment:

```py
comments_dataset = comments_dataset.map(lambda x: {"comment_length": len(x["comments"].split())})
```

We can use this new column to filter out short comments, which typically include things like "cc @lewtun" or "Thanks!" that are not relevant for our search engine. There's no precise number to select for the filter, but around 15 words seems like a good start:

```py
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset
```

```python out
Dataset({features: ["html_url", "title", "comments", "body", "comment_length"], num_rows: 2098})
```

Having cleaned up our dataset a bit, let's concatenate the issue title, description, and comments together in a new `text` column. As usual, we'll write a simple function that we can pass to `Dataset.map()`:

```py
def concatenate_text(examples):
    return {"text": examples["title"] + " \n " + examples["body"] + " \n " + examples["comments"]}


comments_dataset = comments_dataset.map(concatenate_text)
```

We're finally ready to create some embeddings! Let's take a look.

## Creating text embeddings[[creating-text-embeddings]]

We saw in [Chapter 2](/course/chapter2) that we can obtain token embeddings by using the `AutoModel` class. All we need to do is pick a suitable checkpoint to load the model from. Fortunately, there's a library called `sentence-transformers` that is dedicated to creating embeddings. As described in the library's [documentation](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search), our use case is an example of _asymmetric semantic search_ because we have a short query whose answer we'd like to find in a longer document, like a an issue comment. The handy [model overview table](https://www.sbert.net/docs/pretrained_models.html#model-overview) in the documentation indicates that the `multi-qa-mpnet-base-dot-v1` checkpoint has the best performance for semantic search, so we'll use that for our application. We'll also load the tokenizer using the same checkpoint:

{#if fw === 'pt'}

```py
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
```

To speed up the embedding process, it helps to place the model and inputs on a GPU device, so let's do that now:

```py
import torch

device = torch.device("cuda")
model.to(device)
```

{:else}

```py
from transformers import AutoTokenizer, TFAutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
```

Note that we've set `from_pt=True` as an argument of the `from_pretrained()` method. That's because the `multi-qa-mpnet-base-dot-v1` checkpoint only has PyTorch weights, so setting `from_pt=True` will automatically convert them to the TensorFlow format for us. As you can see, it is very simple to switch between frameworks in 🤗 Transformers!

{/if}

As we mentioned earlier, we'd like to represent each entry in our GitHub issues corpus as a single vector, so we need to "pool" or average our token embeddings in some way. One popular approach is to perform *CLS pooling* on our model's outputs, where we simply collect the last hidden state for the special `[CLS]` token. The following function does the trick for us:

```py
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]
```

Next, we'll create a helper function that will tokenize a list of documents, place the tensors on the GPU, feed them to the model, and finally apply CLS pooling to the outputs:

{#if fw === 'pt'}

```py
def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

We can test the function works by feeding it the first text entry in our corpus and inspecting the output shape:

```py
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
```

```python out
torch.Size([1, 768])
```

Great, we've converted the first entry in our corpus into a 768-dimensional vector! We can use `Dataset.map()` to apply our `get_embeddings()` function to each row in our corpus, so let's create a new `embeddings` column as follows:

```py
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
```

{:else}

```py
def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="tf")
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

We can test the function works by feeding it the first text entry in our corpus and inspecting the output shape:

```py
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
```

```python out
TensorShape([1, 768])
```

Great, we've converted the first entry in our corpus into a 768-dimensional vector! We can use `Dataset.map()` to apply our `get_embeddings()` function to each row in our corpus, so let's create a new `embeddings` column as follows:

```py
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
)
```

{/if}

Notice that we've converted the embeddings to NumPy arrays -- that's because 🤗 Datasets requires this format when we try to index them with FAISS, which we'll do next.


## Using FAISS for efficient similarity search[[using-faiss-for-efficient-similarity-search]]

Now that we have a dataset of embeddings, we need some way to search over them. To do this, we'll use a special data structure in 🤗 Datasets called a _FAISS index_. [FAISS](https://faiss.ai/) (short for Facebook AI Similarity Search) is a library that provides efficient algorithms to quickly search and cluster embedding vectors.

The basic idea behind FAISS is to create a special data structure called an _index_ that allows one to find which embeddings are similar to an input embedding. Creating a FAISS index in 🤗 Datasets is simple -- we use the `Dataset.add_faiss_index()` function and specify which column of our dataset we'd like to index:

```py
embeddings_dataset.add_faiss_index(column="embeddings")
```

We can now perform queries on this index by doing a nearest neighbor lookup with the `Dataset.get_nearest_examples()` function. Let's test this out by first embedding a question as follows:

{#if fw === 'pt'}

```py
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape
```

```python out
torch.Size([1, 768])
```

{:else}

```py
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).numpy()
question_embedding.shape
```

```python out
(1, 768)
```

{/if}

Just like with the documents, we now have a 768-dimensional vector representing the query, which we can compare against the whole corpus to find the most similar embeddings:

```py
scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=5)
```

The `Dataset.get_nearest_examples()` function returns a tuple of scores that rank the overlap between the query and the document, and a corresponding set of samples (here, the 5 best matches). Let's collect these in a `pandas.DataFrame` so we can easily sort them:

```py
import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
```

Now we can iterate over the first few rows to see how well our query matched the available comments:

```py
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
```

```python out
"""
COMMENT: Requiring online connection is a deal breaker in some cases unfortunately so it'd be great if offline mode is added similar to how `transformers` loads models offline fine.

@mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?
SCORE: 25.505046844482422
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: The local dataset builders (csv, text , json and pandas) are now part of the `datasets` package since #1726 :)
You can now use them offline
\`\`\`python
datasets = load_dataset("text", data_files=data_files)
\`\`\`

We'll do a new release soon
SCORE: 24.555509567260742
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: I opened a PR that allows to reload modules that have already been loaded once even if there's no internet.

Let me know if you know other ways that can make the offline mode experience better. I'd be happy to add them :)

I already note the "freeze" modules option, to prevent local modules updates. It would be a cool feature.

----------

> @mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?

Indeed `load_dataset` allows to load remote dataset script (squad, glue, etc.) but also you own local ones.
For example if you have a dataset script at `./my_dataset/my_dataset.py` then you can do
\`\`\`python
load_dataset("./my_dataset")
\`\`\`
and the dataset script will generate your dataset once and for all.

----------

About I'm looking into having `csv`, `json`, `text`, `pandas` dataset builders already included in the `datasets` package, so that they are available offline by default, as opposed to the other datasets that require the script to be downloaded.
cf #1724
SCORE: 24.14896583557129
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: > here is my way to load a dataset offline, but it **requires** an online machine
>
> 1. (online machine)
>
> ```
>
> import datasets
>
> data = datasets.load_dataset(...)
>
> data.save_to_disk(/YOUR/DATASET/DIR)
>
> ```
>
> 2. copy the dir from online to the offline machine
>
> 3. (offline machine)
>
> ```
>
> import datasets
>
> data = datasets.load_from_disk(/SAVED/DATA/DIR)
>
> ```
>
>
>
> HTH.


SCORE: 22.893993377685547
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: here is my way to load a dataset offline, but it **requires** an online machine
1. (online machine)
\`\`\`
import datasets
data = datasets.load_dataset(...)
data.save_to_disk(/YOUR/DATASET/DIR)
\`\`\`
2. copy the dir from online to the offline machine
3. (offline machine)
\`\`\`
import datasets
data = datasets.load_from_disk(/SAVED/DATA/DIR)
\`\`\`

HTH.
SCORE: 22.406635284423828
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================
"""
```

Not bad! Our second hit seems to match the query.

> [!TIP]
> ✏️ **Try it out!** Create your own query and see whether you can find an answer in the retrieved documents. You might have to increase the `k` parameter in `Dataset.get_nearest_examples()` to broaden the search.

---

<!-- Section 5.7 -->

# 🤗 Datasets, check![[datasets-check]]

<CourseFloatingBanner
    chapter={5}
    classNames="absolute z-10 right-0 top-0"
/>

Well, that was quite a tour through the 🤗 Datasets library -- congratulations on making it this far! With the knowledge that you've gained from this chapter, you should be able to:

- Load datasets from anywhere, be it the Hugging Face Hub, your laptop, or a remote server at your company.
- Wrangle your data using a mix of the `Dataset.map()` and `Dataset.filter()` functions.
- Quickly switch between data formats like Pandas and NumPy using `Dataset.set_format()`.
- Create your very own dataset and push it to the Hugging Face Hub.
- Embed your documents using a Transformer model and build a semantic search engine using FAISS.

In [Chapter 7](/course/chapter7), we'll put all of this to good use as we take a deep dive into the core NLP tasks that Transformer models are great for. Before jumping ahead, though, put your knowledge of 🤗 Datasets to the test with a quick quiz!

---

<!-- Section 5.8 -->

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={5}
    classNames="absolute z-10 right-0 top-0"
/>

This chapter covered a lot of ground! Don't worry if you didn't grasp all the details; the next chapters will help you understand how things work under the hood.

Before moving on, though, let's test what you learned in this chapter.

### 1. The `load_dataset()` function in 🤗 Datasets allows you to load a dataset from which of the following locations? 

<Question
	choices={[
		{
			text: "Locally, e.g. on your laptop",
			explain: "Correct! You can pass the paths of local files to the <code>data_files</code> argument of <code>load_dataset()</code> to load local datasets.",
			correct: true
		},
		{
			text: "The Hugging Face Hub",
			explain: "Correct! You can load datasets on the Hub by providing the dataset ID, e.g. <code>load_dataset('emotion')</code>.",
			correct: true
		},
		{
			text: "A remote server",
			explain: "Correct! You can pass URLs to the <code>data_files</code> argument of <code>load_dataset()</code> to load remote files.",
			correct: true
		},
	]}
/>

### 2. Suppose you load one of the GLUE tasks as follows:

```py
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")
```

Which of the following commands will produce a random sample of 50 elements from `dataset`?

<Question
	choices={[
		{
			text: "<code>dataset.sample(50)</code>",
			explain: "This is incorrect -- there is no <code>Dataset.sample()</code> method."
		},
		{
			text: "<code>dataset.shuffle().select(range(50))</code>",
			explain: "Correct! As you saw in this chapter, you first shuffle the dataset and then select the samples from it.",
			correct: true
		},
		{
			text: "<code>dataset.select(range(50)).shuffle()</code>",
			explain: "This is incorrect -- although the code will run, it will only shuffle the first 50 elements in the dataset."
		}
	]}
/>

### 3. Suppose you have a dataset about household pets called `pets_dataset`, which has a `name` column that denotes the name of each pet. Which of the following approaches would allow you to filter the dataset for all pets whose names start with the letter "L"?

<Question
	choices={[
		{
			text: "<code>pets_dataset.filter(lambda x : x['name'].startswith('L'))</code>",
			explain: "Correct! Using a Python lambda function for these quick filters is a great idea. Can you think of another solution?",
			correct: true
		},
		{
			text: "<code>pets_dataset.filter(lambda x['name'].startswith('L'))</code>",
			explain: "This is incorrect -- a lambda function takes the general form <code>lambda *arguments* : *expression*</code>, so you need to provide arguments in this case."
		},
		{
			text: "Create a function like <code>def filter_names(x): return x['name'].startswith('L')</code> and run <code>pets_dataset.filter(filter_names)</code>.",
			explain: "Correct! Just like with <code>Dataset.map()</code>, you can pass explicit functions to <code>Dataset.filter()</code>. This is useful when you have some complex logic that isn't suitable for a short lambda function. Which of the other solutions would work?",
			correct: true
		}
	]}
/>

### 4. What is memory mapping?

<Question
	choices={[
		{
			text: "A mapping between CPU and GPU RAM",
			explain: "That's not it -- try again!",
		},
		{
			text: "A mapping between RAM and filesystem storage",
			explain: "Correct! 🤗 Datasets treats each dataset as a memory-mapped file. This allows the library to access and operate on elements of the dataset without needing to fully load it into memory.",
			correct: true
		},
		{
			text: "A mapping between two files in the 🤗 Datasets cache",
			explain: "This is not correct - try again!"
		}
	]}
/>

### 5. Which of the following are the main benefits of memory mapping?

<Question
	choices={[
		{
			text: "Accessing memory-mapped files is faster than reading from or writing to disk.",
			explain: "Correct! This allows 🤗 Datasets to be blazing fast. That's not the only benefit, though.",
			correct: true
		},
		{
			text: "Applications can access segments of data in an extremely large file without having to read the whole file into RAM first.",
			explain: "Correct! This allows 🤗 Datasets to load multi-gigabyte datasets on your laptop without blowing up your CPU. What other advantage does memory mapping offer?",
			correct: true
		},
		{
			text: "It consumes less energy, so your battery lasts longer.",
			explain: "This is not correct -- try again!"
		}
	]}
/>

### 6. Why does the following code fail?

```py
from datasets import load_dataset

dataset = load_dataset("allocine", streaming=True, split="train")
dataset[0]
```

<Question
	choices={[
		{
			text: "It tries to stream a dataset that's too large to fit in RAM.",
			explain: "This is not correct -- streaming datasets are decompressed on the fly, and you can process terabyte-sized datasets with very little RAM!",
		},
		{
			text: "It tries to access an <code>IterableDataset</code>.",
			explain: "Correct! An <code>IterableDataset</code> is a generator, not a container, so you should access its elements using <code>next(iter(dataset))</code>.",
			correct: true
		},
		{
			text: "The <code>allocine</code> dataset doesn't have a <code>train</code> split.",
			explain: "This is incorrect -- check out the [<code>allocine</code> dataset card](https://huggingface.co/datasets/allocine) on the Hub to see which splits it contains."
		}
	]}
/>

### 7. Which of the following are the main benefits of creating a dataset card?

<Question
	choices={[
		{
			text: "It provides information about the intended use and supported tasks of the dataset so others in the community can make an informed decision about using it.",
			explain: "Correct! Undocumented datasets may be used to train models that may not reflect the intentions of the dataset creators, or may produce models whose legal status is murky if they're trained on data that violates privacy or licensing restrictions. This isn't the only benefit, though!",
			correct : true
		},
		{
			text: "It helps draw attention to the biases that are present in a corpus.",
			explain: "Correct! Almost all datasets have some form of bias, which can produce negative consequences downstream. Being aware of them helps model builders understand how to address the inherent biases. What else do dataset cards help with?",
			correct : true
		},
		{
			text: "It improves the chances that others in the community will use my dataset.",
			explain: "Correct! A well-written dataset card will tend to lead to higher usage of your precious dataset. What other benefits does it offer?",
			correct: true
		},
	]}
/>


### 8. What is semantic search?

<Question
	choices={[
		{
			text: "A way to search for exact matches between the words in a query and the documents in a corpus",
			explain: "This is incorrect -- this type of search is called *lexical search*, and it's what you typically see with traditional search engines."
		},
		{
			text: "A way to search for matching documents by understanding the contextual meaning of a query",
			explain: "Correct! Semantic search uses embedding vectors to represent queries and documents, and uses a similarity metric to measure the amount of overlap between them. How else might you describe it?",
			correct: true
		},
		{
			text: "A way to improve search accuracy",
			explain: "Correct! Semantic search engines can capture the intent of a query much better than keyword matching and typically retrieve documents with higher precision. But this isn't the only right answer - what else does semantic search provide?",
			correct: true
		}
	]}
/>

### 9. For asymmetric semantic search, you usually have:

<Question
	choices={[
		{
			text: "A short query and a longer paragraph that answers the query",
			explain: "Correct!",
			correct : true
		},
		{
			text: "Queries and paragraphs that are of about the same length",
			explain: "This is actually an example of symmetric semantic search -- try again!"
		},
		{
			text: "A long query and a shorter paragraph that answers the query",
			explain: "This is incorrect -- try again!"
		}
	]}
/>

### 10. Can I use 🤗 Datasets to load data for use in other domains, like speech processing?

<Question
	choices={[
		{
			text: "No",
			explain: "This is incorrect -- 🤗 Datasets currently supports tabular data, audio, and computer vision. Check out the <a  href='https://huggingface.co/datasets/mnist'>MNIST dataset</a> on the Hub for a computer vision example."
		},
		{
			text: "Yes",
			explain: "Correct! Check out the exciting developments with speech and vision in the 🤗 Transformers library to see how 🤗 Datasets is used in these domains.",
			correct : true
		},
	]}
/>

