# Hugging Face NLP Course — Chapter 1

Source: https://huggingface.co/learn/nlp-course/chapter1


---

<!-- Section 1.1 -->

# Introduction[[introduction]]

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

## Welcome to the 🤗 Course![[welcome-to-the-course]]

<Youtube id="00GKzGyWFEs" />

This course will teach you about large language models (LLMs) and natural language processing (NLP) using libraries from the [Hugging Face](https://huggingface.co/) ecosystem — [🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Tokenizers](https://github.com/huggingface/tokenizers), and [🤗 Accelerate](https://github.com/huggingface/accelerate) — as well as the [Hugging Face Hub](https://huggingface.co/models). 

We'll also cover libraries outside the Hugging Face ecosystem. These are amazing contributions to the AI community and incredibly useful tools.

It's completely free and without ads.

## Understanding NLP and LLMs[[understanding-nlp-and-llms]]

While this course was originally focused on NLP (Natural Language Processing), it has evolved to emphasize Large Language Models (LLMs), which represent the latest advancement in the field. 

**What's the difference?**
- **NLP (Natural Language Processing)** is the broader field focused on enabling computers to understand, interpret, and generate human language. NLP encompasses many techniques and tasks such as sentiment analysis, named entity recognition, and machine translation.
- **LLMs (Large Language Models)** are a powerful subset of NLP models characterized by their massive size, extensive training data, and ability to perform a wide range of language tasks with minimal task-specific training. Models like the Llama, GPT, or Claude series are examples of LLMs that have revolutionized what's possible in NLP.

Throughout this course, you'll learn about both traditional NLP concepts and cutting-edge LLM techniques, as understanding the foundations of NLP is crucial for working effectively with LLMs.

## What to expect?[[what-to-expect]]

Here is a brief overview of the course:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/summary.svg" alt="Brief overview of the chapters of the course.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/summary-dark.svg" alt="Brief overview of the chapters of the course.">
</div>

- Chapters 1 to 4 provide an introduction to the main concepts of the 🤗 Transformers library. By the end of this part of the course, you will be familiar with how Transformer models work and will know how to use a model from the [Hugging Face Hub](https://huggingface.co/models), fine-tune it on a dataset, and share your results on the Hub!
- Chapters 5 to 8 teach the basics of 🤗 Datasets and 🤗 Tokenizers before diving into classic NLP tasks and LLM techniques. By the end of this part, you will be able to tackle the most common language processing challenges by yourself.
- Chapter 9 goes beyond NLP to cover how to build and share demos of your models on the 🤗 Hub. By the end of this part, you will be ready to showcase your 🤗 Transformers application to the world!
- Chapters 10 to 12 dive into advanced LLM topics like fine-tuning, curating high-quality datasets, and building reasoning models.

This course:

* Requires a good knowledge of Python
* Is better taken after an introductory deep learning course, such as [fast.ai's](https://www.fast.ai/) [Practical Deep Learning for Coders](https://course.fast.ai/) or one of the programs developed by [DeepLearning.AI](https://www.deeplearning.ai/)
* Does not expect prior [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/) knowledge, though some familiarity with either of those will help

After you've completed this course, we recommend checking out DeepLearning.AI's [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearning-ai&utm_medium=institutions&utm_campaign=20211011-nlp-2-hugging_face-page-nlp-refresh), which covers a wide range of traditional NLP models like naive Bayes and LSTMs that are well worth knowing about!

## Who are we?[[who-are-we]]

About the authors:

[**Abubakar Abid**](https://huggingface.co/abidlabs) completed his PhD at Stanford in applied machine learning. During his PhD, he founded [Gradio](https://github.com/gradio-app/gradio), an open-source Python library that has been used to build over 600,000 machine learning demos. Gradio was acquired by Hugging Face, which is where Abubakar now serves as a machine learning team lead.

[**Ben Burtenshaw**](https://huggingface.co/burtenshaw) is a Machine Learning Engineer at Hugging Face. He completed his PhD in Natural Language Processing at the University of Antwerp, where he applied Transformer models to generate children stories for the purpose of improving literacy skills. Since then, he has focused on educational materials and tools for the wider community.

[**Matthew Carrigan**](https://huggingface.co/Rocketknight1) is a Machine Learning Engineer at Hugging Face. He lives in Dublin, Ireland and previously worked as an ML engineer at Parse.ly and before that as a post-doctoral researcher at Trinity College Dublin. He does not believe we're going to get to AGI by scaling existing architectures, but has high hopes for robot immortality regardless.

[**Lysandre Debut**](https://huggingface.co/lysandre) is a Machine Learning Engineer at Hugging Face and has been working on the 🤗 Transformers library since the very early development stages. His aim is to make NLP accessible for everyone by developing tools with a very simple API.

[**Sylvain Gugger**](https://huggingface.co/sgugger) is a Research Engineer at Hugging Face and one of the core maintainers of the 🤗 Transformers library. Previously he was a Research Scientist at fast.ai, and he co-wrote _[Deep Learning for Coders with fastai and PyTorch](https://learning.oreilly.com/library/view/deep-learning-for/9781492045519/)_ with Jeremy Howard. The main focus of his research is on making deep learning more accessible, by designing and improving techniques that allow models to train fast on limited resources.

[**Dawood Khan**](https://huggingface.co/dawoodkhan82) is a Machine Learning Engineer at Hugging Face. He's from NYC and graduated from New York University studying Computer Science. After working as an iOS Engineer for a few years, Dawood quit to start Gradio with his fellow co-founders. Gradio was eventually acquired by Hugging Face.

[**Merve Noyan**](https://huggingface.co/merve) is a developer advocate at Hugging Face, working on developing tools and building content around them to democratize machine learning for everyone.

[**Lucile Saulnier**](https://huggingface.co/SaulLu) is a machine learning engineer at Hugging Face, developing and supporting the use of open source tools. She is also actively involved in many research projects in the field of Natural Language Processing such as collaborative training and BigScience.

[**Lewis Tunstall**](https://huggingface.co/lewtun) is a machine learning engineer at Hugging Face, focused on developing open-source tools and making them accessible to the wider community. He is also a co-author of the O'Reilly book [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/).

[**Leandro von Werra**](https://huggingface.co/lvwerra) is a machine learning engineer in the open-source team at Hugging Face and also a co-author of the O'Reilly book [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/). He has several years of industry experience bringing NLP projects to production by working across the whole machine learning stack..

## FAQ[[faq]]

Here are some answers to frequently asked questions:

- **Does taking this course lead to a certification?**
Currently we do not have any certification for this course. However, we are working on a certification program for the Hugging Face ecosystem -- stay tuned!

- **How much time should I spend on this course?**
Each chapter in this course is designed to be completed in 1 week, with approximately 6-8 hours of work per week. However, you can take as much time as you need to complete the course.

- **Where can I ask a question if I have one?**
If you have a question about any section of the course, just click on the "*Ask a question*" banner at the top of the page to be automatically redirected to the right section of the [Hugging Face forums](https://discuss.huggingface.co/):

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/forum-button.png" alt="Link to the Hugging Face forums" width="75%">

Note that a list of [project ideas](https://discuss.huggingface.co/c/course/course-event/25) is also available on the forums if you wish to practice more once you have completed the course.

- **Where can I get the code for the course?**
For each section, click on the banner at the top of the page to run the code in either Google Colab or Amazon SageMaker Studio Lab:

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/notebook-buttons.png" alt="Link to the Hugging Face course notebooks" width="75%">

The Jupyter notebooks containing all the code from the course are hosted on the [`huggingface/notebooks`](https://github.com/huggingface/notebooks) repo. If you wish to generate them locally, check out the instructions in the [`course`](https://github.com/huggingface/course#-jupyter-notebooks) repo on GitHub.


- **How can I contribute to the course?**
There are many ways to contribute to the course! If you find a typo or a bug, please open an issue on the [`course`](https://github.com/huggingface/course) repo. If you would like to help translate the course into your native language, check out the instructions [here](https://github.com/huggingface/course#translating-the-course-into-your-language).

- ** What were the choices made for each translation?**
Each translation has a glossary and `TRANSLATING.txt` file that details the choices that were made for machine learning jargon etc. You can find an example for German [here](https://github.com/huggingface/course/blob/main/chapters/de/TRANSLATING.txt).


- **Can I reuse this course?**
Of course! The course is released under the permissive [Apache 2 license](https://www.apache.org/licenses/LICENSE-2.0.html). This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. If you would like to cite the course, please use the following BibTeX:

```
@misc{huggingfacecourse,
  author = {Hugging Face},
  title = {The Hugging Face Course, 2022},
  howpublished = "\url{https://huggingface.co/course}",
  year = {2022},
  note = "[Online; accessed <today>]"
}
```

## Languages and translations[[languages-and-translations]]

Thanks to our wonderful community, the course is available in many languages beyond English 🔥! Check out the table below to see which languages are available and who contributed to the translations:

| Language                                                                      | Authors                                                                                                                                                                                                                                                                                                                                                  |
|:------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [French](https://huggingface.co/course/fr/chapter1/1)                         | [@lbourdois](https://github.com/lbourdois), [@ChainYo](https://github.com/ChainYo), [@melaniedrevet](https://github.com/melaniedrevet), [@abdouaziz](https://github.com/abdouaziz)                                                                                                                                                                       |
| [Vietnamese](https://huggingface.co/course/vi/chapter1/1)                     | [@honghanhh](https://github.com/honghanhh)                                                                                                                                                                                                                                                                                                               |
| [Chinese (simplified)](https://huggingface.co/course/zh-CN/chapter1/1)        | [@zhlhyx](https://github.com/zhlhyx), [petrichor1122](https://github.com/petrichor1122), [@yaoqih](https://github.com/yaoqih)                                                                                                                                                                                                                    |
| [Bengali](https://huggingface.co/course/bn/chapter1/1) (WIP)                  | [@avishek-018](https://github.com/avishek-018), [@eNipu](https://github.com/eNipu)                                                                                                                                                                                                                                                                       |
| [German](https://huggingface.co/course/de/chapter1/1) (WIP)                   | [@JesperDramsch](https://github.com/JesperDramsch), [@MarcusFra](https://github.com/MarcusFra), [@fabridamicelli](https://github.com/fabridamicelli)                                                                                                                                                                                                     |
| [Spanish](https://huggingface.co/course/es/chapter1/1) (WIP)                  | [@camartinezbu](https://github.com/camartinezbu), [@munozariasjm](https://github.com/munozariasjm), [@fordaz](https://github.com/fordaz)                                                                                                                                                                                                                 |
| [Persian](https://huggingface.co/course/fa/chapter1/1) (WIP)                  | [@jowharshamshiri](https://github.com/jowharshamshiri), [@schoobani](https://github.com/schoobani)                                                                                                                                                                                                                                                       |
| [Gujarati](https://huggingface.co/course/gu/chapter1/1) (WIP)                 | [@pandyaved98](https://github.com/pandyaved98)                                                                                                                                                                                                                                                                                                           |
| [Hebrew](https://huggingface.co/course/he/chapter1/1) (WIP)                   | [@omer-dor](https://github.com/omer-dor)                                                                                                                                                                                                                                                                                                                 |
| [Hindi](https://huggingface.co/course/hi/chapter1/1) (WIP)                    | [@pandyaved98](https://github.com/pandyaved98)                                                                                                                                                                                                                                                                                                           |
| [Bahasa Indonesia](https://huggingface.co/course/id/chapter1/1) (WIP)         | [@gstdl](https://github.com/gstdl)                                                                                                                                                                                                                                                                                                                       |
| [Italian](https://huggingface.co/course/it/chapter1/1) (WIP)                  | [@CaterinaBi](https://github.com/CaterinaBi), [@ClonedOne](https://github.com/ClonedOne),    [@Nolanogenn](https://github.com/Nolanogenn), [@EdAbati](https://github.com/EdAbati), [@gdacciaro](https://github.com/gdacciaro)                                                                                                                            |
| [Japanese](https://huggingface.co/course/ja/chapter1/1) (WIP)                 | [@hiromu166](https://github.com/@hiromu166), [@younesbelkada](https://github.com/@younesbelkada), [@HiromuHota](https://github.com/@HiromuHota)                                                                                                                                                                                                          |
| [Korean](https://huggingface.co/course/ko/chapter1/1) (WIP)                   | [@Doohae](https://github.com/Doohae), [@wonhyeongseo](https://github.com/wonhyeongseo), [@dlfrnaos19](https://github.com/dlfrnaos19)                                                                                                                                                                                                                     |
| [Portuguese](https://huggingface.co/course/pt/chapter1/1) (WIP)               | [@johnnv1](https://github.com/johnnv1), [@victorescosta](https://github.com/victorescosta), [@LincolnVS](https://github.com/LincolnVS)                                                                                                                                                                                                                   |
| [Russian](https://huggingface.co/course/ru/chapter1/1) (WIP)                  | [@pdumin](https://github.com/pdumin), [@svv73](https://github.com/svv73)                                                                                                                                                                                                                                                                                 |
| [Thai](https://huggingface.co/course/th/chapter1/1) (WIP)                     | [@peeraponw](https://github.com/peeraponw), [@a-krirk](https://github.com/a-krirk), [@jomariya23156](https://github.com/jomariya23156), [@ckingkan](https://github.com/ckingkan)                                                                                                                                                                         |
| [Turkish](https://huggingface.co/course/tr/chapter1/1) (WIP)                  | [@tanersekmen](https://github.com/tanersekmen), [@mertbozkir](https://github.com/mertbozkir), [@ftarlaci](https://github.com/ftarlaci), [@akkasayaz](https://github.com/akkasayaz)                                                                                                                                                                       |
| [Chinese (traditional)](https://huggingface.co/course/zh-TW/chapter1/1) (WIP) | [@davidpeng86](https://github.com/davidpeng86)                                                                                                                                                                                                                                                                                                           |

For some languages, the [course YouTube videos](https://youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o) have subtitles in the language. You can enable them by first clicking the _CC_ button in the bottom right corner of the video. Then, under the settings icon ⚙️, you can select the language you want by selecting the _Subtitles/CC_ option.

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/subtitles.png" alt="Activating subtitles for the Hugging Face course YouTube videos" width="75%">

> [!TIP]
> Don't see your language in the above table or you'd like to contribute to an existing translation? You can help us translate the course by following the instructions <a href="https://github.com/huggingface/course#translating-the-course-into-your-language">here</a>.

## Let's go 🚀

Are you ready to roll? In this chapter, you will learn:

* How to use the `pipeline()` function to solve NLP tasks such as text generation and classification
* About the Transformer architecture
* How to distinguish between encoder, decoder, and encoder-decoder architectures and use cases



---

<!-- Section 1.2 -->

# Natural Language Processing and Large Language Models[[natural-language-processing-and-large-language-models]]

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

Before jumping into Transformer models, let's do a quick overview of what natural language processing is, how large language models have transformed the field, and why we care about it.

## What is NLP?[[what-is-nlp]]

<Youtube id="iNzlxWUAjd4" />

NLP is a field of linguistics and machine learning focused on understanding everything related to human language. The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words.

The following is a list of common NLP tasks, with some examples of each:

- **Classifying whole sentences**: Getting the sentiment of a review, detecting if an email is spam, determining if a sentence is grammatically correct or whether two sentences are logically related or not
- **Classifying each word in a sentence**: Identifying the grammatical components of a sentence (noun, verb, adjective), or the named entities (person, location, organization)
- **Generating text content**: Completing a prompt with auto-generated text, filling in the blanks in a text with masked words
- **Extracting an answer from a text**: Given a question and a context, extracting the answer to the question based on the information provided in the context
- **Generating a new sentence from an input text**: Translating a text into another language, summarizing a text

NLP isn't limited to written text though. It also tackles complex challenges in speech recognition and computer vision, such as generating a transcript of an audio sample or a description of an image.

## The Rise of Large Language Models (LLMs)[[rise-of-llms]]

In recent years, the field of NLP has been revolutionized by Large Language Models (LLMs). These models, which include architectures like GPT (Generative Pre-trained Transformer) and [Llama](https://huggingface.co/meta-llama), have transformed what's possible in language processing.

> [!TIP]
> A large language model (LLM) is an AI model trained on massive amounts of text data that can understand and generate human-like text, recognize patterns in language, and perform a wide variety of language tasks without task-specific training. They represent a significant advancement in the field of natural language processing (NLP).

LLMs are characterized by:
- **Scale**: They contain millions, billions, or even hundreds of billions of parameters
- **General capabilities**: They can perform multiple tasks without task-specific training
- **In-context learning**: They can learn from examples provided in the prompt
- **Emergent abilities**: As these models grow in size, they demonstrate capabilities that weren't explicitly programmed or anticipated

The advent of LLMs has shifted the paradigm from building specialized models for specific NLP tasks to using a single, large model that can be prompted or fine-tuned to address a wide range of language tasks. This has made sophisticated language processing more accessible while also introducing new challenges in areas like efficiency, ethics, and deployment. 

However, LLMs also have important limitations:
- **Hallucinations**: They can generate incorrect information confidently
- **Lack of true understanding**: They lack true understanding of the world and operate purely on statistical patterns
- **Bias**: They may reproduce biases present in their training data or inputs.
- **Context windows**: They have limited context windows (though this is improving)
- **Computational resources**: They require significant computational resources

## Why is language processing challenging?[[why-is-it-challenging]]

Computers don't process information in the same way as humans. For example, when we read the sentence "I am hungry," we can easily understand its meaning. Similarly, given two sentences such as "I am hungry" and "I am sad," we're able to easily determine how similar they are. For machine learning (ML) models, such tasks are more difficult. The text needs to be processed in a way that enables the model to learn from it. And because language is complex, we need to think carefully about how this processing must be done. There has been a lot of research done on how to represent text, and we will look at some methods in the next chapter.

Even with the advances in LLMs, many fundamental challenges remain. These include understanding ambiguity, cultural context, sarcasm, and humor. LLMs address these challenges through massive training on diverse datasets, but still often fall short of human-level understanding in many complex scenarios.


---

<!-- Section 1.3 -->

# Transformers, what can they do?[[transformers-what-can-they-do]]

<CourseFloatingBanner chapter={1}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter1/section3.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter1/section3.ipynb"},
]} />

In this section, we will look at what Transformer models can do and use our first tool from the 🤗 Transformers library: the `pipeline()` function.

> [!TIP]
> 👀 See that <em>Open in Colab</em> button on the top right? Click on it to open a Google Colab notebook with all the code samples of this section. This button will be present in any section containing code examples. 
>
> If you want to run the examples locally, we recommend taking a look at the <a href="/course/chapter0">setup</a>.

## Transformers are everywhere![[transformers-are-everywhere]]

Transformer models are used to solve all kinds of tasks across different modalities, including natural language processing (NLP), computer vision, audio processing, and more. Here are some of the companies and organizations using Hugging Face and Transformer models, who also contribute back to the community by sharing their models:

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/companies.PNG" alt="Companies using Hugging Face" width="100%">

The [🤗 Transformers library](https://github.com/huggingface/transformers) provides the functionality to create and use those shared models. The [Model Hub](https://huggingface.co/models) contains millions of pretrained models that anyone can download and use. You can also upload your own models to the Hub!

> [!TIP]
> ⚠️ The Hugging Face Hub is not limited to Transformer models. Anyone can share any kind of models or datasets they want! <a href="https://huggingface.co/join">Create a huggingface.co</a> account to benefit from all available features!

Before diving into how Transformer models work under the hood, let's look at a few examples of how they can be used to solve some interesting NLP problems.

## Working with pipelines[[working-with-pipelines]]

<Youtube id="tiZFewofSLM" />

The most basic object in the 🤗 Transformers library is the `pipeline()` function. It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

```python out
[{"label": "POSITIVE", "score": 0.9598047137260437}]
```

We can even pass several sentences!

```python
classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"])
```

```python out
[
    {"label": "POSITIVE", "score": 0.9598047137260437},
    {"label": "NEGATIVE", "score": 0.9994558095932007},
]
```

By default, this pipeline selects a particular pretrained model that has been fine-tuned for sentiment analysis in English. The model is downloaded and cached when you create the `classifier` object. If you rerun the command, the cached model will be used instead and there is no need to download the model again.

There are three main steps involved when you pass some text to a pipeline:

1. The text is preprocessed into a format the model can understand.
2. The preprocessed inputs are passed to the model.
3. The predictions of the model are post-processed, so you can make sense of them.

## Available pipelines for different modalities

The `pipeline()` function supports multiple modalities, allowing you to work with text, images, audio, and even multimodal tasks. In this course we'll focus on text tasks, but it's useful to understand the transformer architecture's potential, so we'll briefly outline it. 

Here's an overview of what's available:

> [!TIP]
> For a full and updated list of pipelines, see the [🤗 Transformers documentation](https://huggingface.co/docs/hub/en/models-tasks).

### Text pipelines

- `text-generation`: Generate text from a prompt
- `text-classification`: Classify text into predefined categories
- `summarization`: Create a shorter version of a text while preserving key information
- `translation`: Translate text from one language to another
- `zero-shot-classification`: Classify text without prior training on specific labels
- `feature-extraction`: Extract vector representations of text

### Image pipelines

- `image-to-text`: Generate text descriptions of images
- `image-classification`: Identify objects in an image
- `object-detection`: Locate and identify objects in images

### Audio pipelines

- `automatic-speech-recognition`: Convert speech to text
- `audio-classification`: Classify audio into categories
- `text-to-speech`: Convert text to spoken audio

### Multimodal pipelines

- `image-text-to-text`: Respond to an image based on a text prompt

Let's explore some of these pipelines in more detail!

## Zero-shot classification[[zero-shot-classification]]

We'll start by tackling a more challenging task where we need to classify texts that haven't been labelled. This is a common scenario in real-world projects because annotating text is usually time-consuming and requires domain expertise. For this use case, the `zero-shot-classification` pipeline is very powerful: it allows you to specify which labels to use for the classification, so you don't have to rely on the labels of the pretrained model. You've already seen how the model can classify a sentence as positive or negative using those two labels — but it can also classify the text using any other set of labels you like.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

```python out
{
    "sequence": "This is a course about the Transformers library",
    "labels": ["education", "business", "politics"],
    "scores": [0.8445963859558105, 0.111976258456707, 0.043427448719739914],
}
```

This pipeline is called _zero-shot_ because you don't need to fine-tune the model on your data to use it. It can directly return probability scores for any list of labels you want!

> [!TIP]
> ✏️ **Try it out!** Play around with your own sequences and labels and see how the model behaves.


## Text generation[[text-generation]]

Now let's see how to use a pipeline to generate some text. The main idea here is that you provide a prompt and the model will auto-complete it by generating the remaining text. This is similar to the predictive text feature that is found on many phones. Text generation involves randomness, so it's normal if you don't get the same results as shown below.

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

```python out
[
    {
        "generated_text": "In this course, we will teach you how to understand and use "
        "data flow and data interchange when handling user data. We "
        "will be working with one or more of the most commonly used "
        "data flows — data flows of various types, as seen by the "
        "HTTP"
    }
]
```

You can control how many different sequences are generated with the argument `num_return_sequences` and the total length of the output text with the argument `max_length`.

> [!TIP]
> ✏️ **Try it out!** Use the `num_return_sequences` and `max_length` arguments to generate two sentences of 15 words each.

## Using any model from the Hub in a pipeline[[using-any-model-from-the-hub-in-a-pipeline]]

The previous examples used the default model for the task at hand, but you can also choose a particular model from the Hub to use in a pipeline for a specific task — say, text generation. Go to the [Model Hub](https://huggingface.co/models) and click on the corresponding tag on the left to display only the supported models for that task. You should get to a page like [this one](https://huggingface.co/models?pipeline_tag=text-generation).

Let's try the [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) model! Here's how to load it in the same pipeline as before:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

```python out
[
    {
        "generated_text": "In this course, we will teach you how to manipulate the world and "
        "move your mental and physical capabilities to your advantage."
    },
    {
        "generated_text": "In this course, we will teach you how to become an expert and "
        "practice realtime, and with a hands on experience on both real "
        "time and real"
    },
]
```

You can refine your search for a model by clicking on the language tags, and pick a model that will generate text in another language. The Model Hub even contains checkpoints for multilingual models that support several languages.

Once you select a model by clicking on it, you'll see that there is a widget enabling you to try it directly online. This way you can quickly test the model's capabilities before downloading it.

> [!TIP]
> ✏️ **Try it out!** Use the filters to find a text generation model for another language. Feel free to play with the widget and use it in a pipeline!

### Inference Providers[[inference-providers]]

All the models can be tested directly through your browser using the Inference Providers, which is available on the Hugging Face [website](https://huggingface.co/docs/inference-providers/en/index). You can play with the model directly on this page by inputting custom text and watching the model process the input data.

Inference Providers that powers the widget is also available as a paid product, which comes in handy if you need it for your workflows. See the [pricing page](https://huggingface.co/docs/inference-providers/en/pricing) for more details.

## Mask filling[[mask-filling]]

The next pipeline you'll try is `fill-mask`. The idea of this task is to fill in the blanks in a given text:

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

```python out
[
    {
        "sequence": "This course will teach you all about mathematical models.",
        "score": 0.19619831442832947,
        "token": 30412,
        "token_str": " mathematical",
    },
    {
        "sequence": "This course will teach you all about computational models.",
        "score": 0.04052725434303284,
        "token": 38163,
        "token_str": " computational",
    },
]
```

The `top_k` argument controls how many possibilities you want to be displayed. Note that here the model fills in the special `<mask>` word, which is often referred to as a *mask token*. Other mask-filling models might have different mask tokens, so it's always good to verify the proper mask word when exploring other models. One way to check it is by looking at the mask word used in the widget.

> [!TIP]
> ✏️ **Try it out!** Search for the `bert-base-cased` model on the Hub and identify its mask word in the Inference API widget. What does this model predict for the sentence in our `pipeline` example above?

## Named entity recognition[[named-entity-recognition]]

Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to entities such as persons, locations, or organizations. Let's look at an example:

```python
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python out
[
    {"entity_group": "PER", "score": 0.99816, "word": "Sylvain", "start": 11, "end": 18},
    {"entity_group": "ORG", "score": 0.97960, "word": "Hugging Face", "start": 33, "end": 45},
    {"entity_group": "LOC", "score": 0.99321, "word": "Brooklyn", "start": 49, "end": 57},
]
```

Here the model correctly identified that Sylvain is a person (PER), Hugging Face an organization (ORG), and Brooklyn a location (LOC).

We pass the option `aggregation_strategy="simple"` in the pipeline creation function to tell the pipeline to regroup together the parts of the sentence that correspond to the same entity: here the model correctly grouped "Hugging" and "Face" as a single organization, even though the name consists of multiple words. In fact, as we will see in the next chapter, the preprocessing even splits some words into smaller parts. For instance, `Sylvain` is split into four pieces: `S`, `##yl`, `##va`, and `##in`. In the post-processing step, the pipeline successfully regrouped those pieces.

> [!TIP]
> ✏️ **Try it out!** Search the Model Hub for a model able to do part-of-speech tagging (usually abbreviated as POS) in English. What does this model predict for the sentence in the example above?

## Question answering[[question-answering]]

The `question-answering` pipeline answers questions using information from a given context:

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

```python out
{"score": 0.6385916471481323, "start": 33, "end": 45, "answer": "Hugging Face"}
```

Note that this pipeline works by extracting information from the provided context; it does not generate the answer.

## Summarization[[summarization]]

Summarization is the task of reducing a text into a shorter text while keeping all (or most) of the important aspects referenced in the text. Here's an example:

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
```

```python out
[
    {
        "summary_text": " America has changed dramatically during recent years . The "
        "number of engineering graduates in the U.S. has declined in "
        "traditional engineering disciplines such as mechanical, civil "
        ", electrical, chemical, and aeronautical engineering . Rapidly "
        "developing economies such as China and India, as well as other "
        "industrial countries in Europe and Asia, continue to encourage "
        "and advance engineering ."
    }
]
```

Like with text generation, you can specify a `max_length` or a `min_length` for the result.


## Translation[[translation]]

For translation, you can use a default model if you provide a language pair in the task name (such as `"translation_en_to_fr"`), but the easiest way is to pick the model you want to use on the [Model Hub](https://huggingface.co/models). Here we'll try translating from French to English:

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

```python out
[{"translation_text": "This course is produced by Hugging Face."}]
```

Like with text generation and summarization, you can specify a `max_length` or a `min_length` for the result.

> [!TIP]
> ✏️ **Try it out!** Search for translation models in other languages and try to translate the previous sentence into a few different languages.

## Image and audio pipelines

Beyond text, Transformer models can also work with images and audio. Here are a few examples:

### Image classification

```python
from transformers import pipeline

image_classifier = pipeline(task="image-classification", model="google/vit-base-patch16-224")
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)
```

```python out
[
    {"label": "lynx, catamount", "score": 0.43350091576576233},
    {
        "label": "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
        "score": 0.034796204417943954,
    },
    {"label": "snow leopard, ounce, Panthera uncia", "score": 0.03240183740854263},
    {"label": "Egyptian cat", "score": 0.02394474856555462},
    {"label": "tiger cat", "score": 0.02288915030658245},
]
```

### Automatic speech recognition

```python
from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result)
```

```python out
{
    "text": " I have a dream that one day this nation will rise up and live out the true meaning of its creed."
}
```

## Combining data from multiple sources

One powerful application of Transformer models is their ability to combine and process data from multiple sources. This is especially useful when you need to:

1. Search across multiple databases or repositories
2. Consolidate information from different formats (text, images, audio)
3. Create a unified view of related information

For example, you could build a system that:
- Searches for information across databases in multiple modalities like text and image.
- Combines results from different sources into a single coherent response. For example, from an audio file and text description.
- Presents the most relevant information from a database of documents and metadata.

## Conclusion

The pipelines shown in this chapter are mostly for demonstrative purposes. They were programmed for specific tasks and cannot perform variations of them. In the next chapter, you'll learn what's inside a `pipeline()` function and how to customize its behavior.


---

<!-- Section 1.4 -->

# How do Transformers work?[[how-do-transformers-work]]

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

In this section, we will take a look at the architecture of Transformer models and dive deeper into the concepts of attention, encoder-decoder architecture, and more.

> [!WARNING]
> 🚀 We're taking things up a notch here. This section is detailed and technical, so don't worry if you don't understand everything right away. We'll come back to these concepts later in the course.

## A bit of Transformer history[[a-bit-of-transformer-history]]

Here are some reference points in the (short) history of Transformer models:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg" alt="A brief chronology of Transformers models.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono-dark.svg" alt="A brief chronology of Transformers models.">
</div>

The [Transformer architecture](https://arxiv.org/abs/1706.03762) was introduced in June 2017. The focus of the original research was on translation tasks. This was followed by the introduction of several influential models, including:

- **June 2018**: [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results

- **October 2018**: [BERT](https://arxiv.org/abs/1810.04805), another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!)

- **February 2019**: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns

- **October 2019**: [T5](https://huggingface.co/papers/1910.10683), A multi-task focused implementation of the sequence-to-sequence Transformer architecture.

- **May 2020**, [GPT-3](https://huggingface.co/papers/2005.14165), an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called _zero-shot learning_)

- **January 2022**: [InstructGPT](https://huggingface.co/papers/2203.02155), a version of GPT-3 that was trained to follow instructions better.

- **January 2023**: [Llama](https://huggingface.co/papers/2302.13971), a large language model that is able to generate text in a variety of languages.

- **March 2023**: [Mistral](https://huggingface.co/papers/2310.06825), a 7-billion-parameter language model that outperforms Llama 2 13B across all evaluated benchmarks, leveraging grouped-query attention for faster inference and sliding window attention to handle sequences of arbitrary length.

- **May 2024**: [Gemma 2](https://huggingface.co/papers/2408.00118), a family of lightweight, state-of-the-art open models ranging from 2B to 27B parameters that incorporate interleaved local-global attentions and group-query attention, with smaller models trained using knowledge distillation to deliver performance competitive with models 2-3 times larger.

- **November 2024**: [SmolLM2](https://huggingface.co/papers/2502.02737), a state-of-the-art small language model (135 million to 1.7 billion parameters) that achieves impressive performance despite its compact size, and unlocking new possibilities for mobile and edge devices.

This list is far from comprehensive, and is just meant to highlight a few of the different kinds of Transformer models. Broadly, they can be grouped into three categories:

- GPT-like (also called _auto-regressive_ Transformer models)
- BERT-like (also called _auto-encoding_ Transformer models) 
- T5-like (also called _sequence-to-sequence_ Transformer models)

We will dive into these families in more depth later on.

## Transformers are language models[[transformers-are-language-models]]

All the Transformer models mentioned above (GPT, BERT, T5, etc.) have been trained as *language models*. This means they have been trained on large amounts of raw text in a self-supervised fashion. 

Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data!

This type of model develops a statistical understanding of the language it has been trained on, but it's less useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called *transfer learning* or *fine-tuning*. During this process, the model is fine-tuned in a supervised way -- that is, using human-annotated labels -- on a given task.

An example of a task is predicting the next word in a sentence having read the *n* previous words. This is called *causal language modeling* because the output depends on the past and present inputs, but not the future ones.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling.svg" alt="Example of causal language modeling in which the next word from a sentence is predicted.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling-dark.svg" alt="Example of causal language modeling in which the next word from a sentence is predicted.">
</div>

Another example is *masked language modeling*, in which the model predicts a masked word in the sentence.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling.svg" alt="Example of masked language modeling in which a masked word from a sentence is predicted.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling-dark.svg" alt="Example of masked language modeling in which a masked word from a sentence is predicted.">
</div>

## Transformers are big models[[transformers-are-big-models]]

Apart from a few outliers (like DistilBERT), the general strategy to achieve better performance is by increasing the models' sizes as well as the amount of data they are pretrained on.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png" alt="Number of parameters of recent Transformers models" width="90%">
</div>

Unfortunately, training a model, especially a large one, requires a large amount of data. This becomes very costly in terms of time and compute resources. It even translates to environmental impact, as can be seen in the following graph.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint.svg" alt="The carbon footprint of a large language model.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint-dark.svg" alt="The carbon footprint of a large language model.">
</div>

<Youtube id="ftWlj4FBHTg"/>

And this is showing a project for a (very big) model led by a team consciously trying to reduce the environmental impact of pretraining. The footprint of running lots of trials to get the best hyperparameters would be even higher.

Imagine if each time a research team, a student organization, or a company wanted to train a model, it did so from scratch. This would lead to huge, unnecessary global costs!

This is why sharing language models is paramount: sharing the trained weights and building on top of already trained weights reduces the overall compute cost and carbon footprint of the community.

By the way, you can evaluate the carbon footprint of your models' training through several tools. For example [ML CO2 Impact](https://mlco2.github.io/impact/) or [Code Carbon]( https://codecarbon.io/) which is integrated in 🤗 Transformers. To learn more about this, you can read this [blog post](https://huggingface.co/blog/carbon-emissions-on-the-hub) which will show you how to generate an `emissions.csv` file with an estimate of the footprint of your training, as well as the [documentation](https://huggingface.co/docs/hub/model-cards-co2) of 🤗 Transformers addressing this topic.


## Transfer Learning[[transfer-learning]]

<Youtube id="BqqfQnyjmgg" />

*Pretraining* is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining.svg" alt="The pretraining of a language model is costly in both time and money.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining-dark.svg" alt="The pretraining of a language model is costly in both time and money.">
</div>

This pretraining is usually done on very large amounts of data. Therefore, it requires a very large corpus of data, and training can take up to several weeks.

*Fine-tuning*, on the other hand, is the training done **after** a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task. Wait -- why not simply train the model for your final use case from the start (**scratch**)? There are a couple of reasons:

*  The pretrained model was already trained on a dataset that has some similarities with the fine-tuning dataset. The fine-tuning process is thus able to take advantage of knowledge acquired by the initial model during pretraining (for instance, with NLP problems, the pretrained model will have some kind of statistical understanding of the language you are using for your task). 
*  Since the pretrained model was already trained on lots of data, the fine-tuning requires way less data to get decent results.
*  For the same reason, the amount of time and resources needed to get good results are much lower.

For example, one could leverage a pretrained model trained on the English language and then fine-tune it on an arXiv corpus, resulting in a science/research-based model. The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is "transferred," hence the term *transfer learning*.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning.svg" alt="The fine-tuning of a language model is cheaper than pretraining in both time and money.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning-dark.svg" alt="The fine-tuning of a language model is cheaper than pretraining in both time and money.">
</div>

Fine-tuning a model therefore has lower time, data, financial, and environmental costs. It is also quicker and easier to iterate over different fine-tuning schemes, as the training is less constraining than a full pretraining.

This process will also achieve better results than training from scratch (unless you have lots of data), which is why you should always try to leverage a pretrained model -- one as close as possible to the task you have at hand -- and fine-tune it.

## General Transformer architecture[[general-transformer-architecture]]

In this section, we'll go over the general architecture of the Transformer model. Don't worry if you don't understand some of the concepts; there are detailed sections later covering each of the components.

<Youtube id="H39Z_720T5s" />

The model is primarily composed of two blocks:

* **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
* **Decoder (right)**: The decoder uses the encoder's representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks.svg" alt="Architecture of a Transformers models">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks-dark.svg" alt="Architecture of a Transformers models">
</div>

Each of these parts can be used independently, depending on the task: 

* **Encoder-only models**: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
* **Decoder-only models**: Good for generative tasks such as text generation.
* **Encoder-decoder models** or **sequence-to-sequence models**: Good for generative tasks that require an input, such as translation or summarization.

We will dive into those architectures independently in later sections.

## Attention layers[[attention-layers]]

A key feature of Transformer models is that they are built with special layers called *attention layers*. In fact, the title of the paper introducing the Transformer architecture was ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)! We will explore the details of attention layers later in the course; for now, all you need to know is that this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.

To put this into context, consider the task of translating text from English to French. Given the input "You like this course", a translation model will need to also attend to the adjacent word "You" to get the proper translation for the word "like", because in French the verb "like" is conjugated differently depending on the subject. The rest of the sentence, however, is not useful for the translation of that word. In the same vein, when translating "this" the model will also need to pay attention to the word "course", because "this" translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of "course". With more complex sentences (and more complex grammar rules), the model would need to pay special attention to words that might appear farther away in the sentence to properly translate each word.

The same concept applies to any task associated with natural language: a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.

Now that you have an idea of what attention layers are all about, let's take a closer look at the Transformer architecture.

## The original architecture[[the-original-architecture]]

The Transformer architecture was originally designed for translation. During training, the encoder receives inputs (sentences) in a certain language, while the decoder receives the same sentences in the desired target language. In the encoder, the attention layers can use all the words in a sentence (since, as we just saw, the translation of a given word can be dependent on what is after as well as before it in the sentence). The decoder, however, works sequentially and can only pay attention to the words in the sentence that it has already translated (so, only the words before the word currently being generated). For example, when we have predicted the first three words of the translated target, we give them to the decoder  which then uses all the inputs of the encoder to try to predict the fourth word.

To speed things up during training (when the model has access to target sentences), the decoder is fed the whole target, but it is not allowed to use future words (if it had access to the word at position 2 when trying to predict the word at position 2, the problem would not be very hard!). For instance, when trying to predict the fourth word, the attention layer will only have access to the words in positions 1 to 3.

The original Transformer architecture looked like this, with the encoder on the left and the decoder on the right:

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg" alt="Architecture of a Transformers models">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg" alt="Architecture of a Transformers models">
</div>

Note that the first attention layer in a decoder block pays attention to all (past) inputs to the decoder, but the second attention layer uses the output of the encoder. It can thus access the whole input sentence to best predict the current word. This is very useful as different languages can have grammatical rules that put the words in different orders, or some context provided later in the sentence may be helpful to determine the best translation of a given word.

The *attention mask* can also be used in the encoder/decoder to prevent the model from paying attention to some special words -- for instance, the special padding word used to make all the inputs the same length when batching together sentences.

##  Architectures vs. checkpoints[[architecture-vs-checkpoints]]

As we dive into Transformer models in this course, you'll see mentions of *architectures* and *checkpoints* as well as *models*. These terms all have slightly different meanings: 

* **Architecture**: This is the skeleton of the model -- the definition of each layer and each operation that happens within the model. 
* **Checkpoints**: These are the weights that will be loaded in a given architecture.
* **Model**: This is an umbrella term that isn't as precise as "architecture" or "checkpoint": it can mean both. This course will specify *architecture* or *checkpoint* when it matters to reduce ambiguity.

For example, BERT is an architecture while `bert-base-cased`, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say "the BERT model" and "the `bert-base-cased` model."


---

<!-- Section 1.5 -->

# How 🤗 Transformers solve tasks

<Youtube id="zsfR7eY9Uho" />

In [Transformers, what can they do?](/course/chapter1/3), you learned about natural language processing (NLP), speech and audio, computer vision tasks, and some important applications of them. This page will look closely at how models solve these tasks and explain what's happening under the hood. There are many ways to solve a given task, some models may implement certain techniques or even approach the task from a new angle, but for Transformer models, the general idea is the same. Owing to its flexible architecture, most models are a variant of an encoder, a decoder, or an encoder-decoder structure. 

> [!TIP]
> Before diving into specific architectural variants, it's helpful to understand that most tasks follow a similar pattern: input data is processed through a model, and the output is interpreted for a specific task. The differences lie in how the data is prepared, what model architecture variant is used, and how the output is processed.

To explain how tasks are solved, we'll walk through what goes on inside the model to output useful predictions. We'll cover the following models and their corresponding tasks:

- [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) for audio classification and automatic speech recognition (ASR)
- [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) and [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext) for image classification
- [DETR](https://huggingface.co/docs/transformers/model_doc/detr) for object detection
- [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) for image segmentation
- [GLPN](https://huggingface.co/docs/transformers/model_doc/glpn) for depth estimation
- [BERT](https://huggingface.co/docs/transformers/model_doc/bert) for NLP tasks like text classification, token classification and question answering that use an encoder
- [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) for NLP tasks like text generation that use a decoder
- [BART](https://huggingface.co/docs/transformers/model_doc/bart) for NLP tasks like summarization and translation that use an encoder-decoder

> [!TIP]
> Before you go further, it is good to have some basic knowledge of the original Transformer architecture. Knowing how encoders, decoders, and attention work will aid you in understanding how different Transformer models work. Be sure to check out our [the previous section](https://huggingface.co/course/chapter1/4?fw=pt) for more information!

## Transformer models for language 

Language models are at the heart of modern NLP. They're designed to understand and generate human language by learning the statistical patterns and relationships between words or tokens in text.

The Transformer was initially designed for machine translation, and since then, it has become the default architecture for solving all AI tasks. Some tasks lend themselves to the Transformer's encoder structure, while others are better suited for the decoder. Still, other tasks make use of both the Transformer's encoder-decoder structure.

### How language models work

Language models work by being trained to predict the probability of a word given the context of surrounding words. This gives them a foundational understanding of language that can generalize to other tasks.

There are two main approaches for training a transformer model:

1. **Masked language modeling (MLM)**: Used by encoder models like BERT, this approach randomly masks some tokens in the input and trains the model to predict the original tokens based on the surrounding context. This allows the model to learn bidirectional context (looking at words both before and after the masked word).

2. **Causal language modeling (CLM)**: Used by decoder models like GPT, this approach predicts the next token based on all previous tokens in the sequence. The model can only use context from the left (previous tokens) to predict the next token.

### Types of language models

In the Transformers library, language models generally fall into three architectural categories:

1. **Encoder-only models** (like BERT): These models use a bidirectional approach to understand context from both directions. They're best suited for tasks that require deep understanding of text, such as classification, named entity recognition, and question answering.

2. **Decoder-only models** (like GPT, Llama): These models process text from left to right and are particularly good at text generation tasks. They can complete sentences, write essays, or even generate code based on a prompt.

3. **Encoder-decoder models** (like T5, BART): These models combine both approaches, using an encoder to understand the input and a decoder to generate output. They excel at sequence-to-sequence tasks like translation, summarization, and question answering.

![transformer-models-for-language](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_architecture.png)

As we covered in the previous section, language models are typically pretrained on large amounts of text data in a self-supervised manner (without human annotations), then fine-tuned on specific tasks. This approach, known as transfer learning, allows these models to adapt to many different NLP tasks with relatively small amounts of task-specific data.

In the following sections, we'll explore specific model architectures and how they're applied to various tasks across speech, vision, and text domains.

> [!TIP]
> Understanding which part of the Transformer architecture (encoder, decoder, or both) is best suited for a particular NLP task is key to choosing the right model. Generally, tasks requiring bidirectional context use encoders, tasks generating text use decoders, and tasks converting one sequence to another use encoder-decoders.

### Text generation

Text generation involves creating coherent and contextually relevant text based on a prompt or input.

[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) is a decoder-only model pretrained on a large amount of text. It can generate convincing (though not always true!) text given a prompt and complete other NLP tasks like question answering despite not being explicitly trained to.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gpt2_architecture.png"/>
</div>

1. GPT-2 uses [byte pair encoding (BPE)](https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe) to tokenize words and generate a token embedding. Positional encodings are added to the token embeddings to indicate the position of each token in the sequence. The input embeddings are passed through multiple decoder blocks to output some final hidden state. Within each decoder block, GPT-2 uses a *masked self-attention* layer which means GPT-2 can't attend to future tokens. It is only allowed to attend to tokens on the left. This is different from BERT's [`mask`] token because, in masked self-attention, an attention mask is used to set the score to `0` for future tokens.

2. The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The label is the next token in the sequence, which are created by shifting the logits to the right by one. The cross-entropy loss is calculated between the shifted logits and the labels to output the next most likely token.

GPT-2's pretraining objective is based entirely on [causal language modeling](https://huggingface.co/docs/transformers/glossary#causal-language-modeling), predicting the next word in a sequence. This makes GPT-2 especially good at tasks that involve generating text.

Ready to try your hand at text generation? Check out our complete [causal language modeling guide](https://huggingface.co/docs/transformers/tasks/language_modeling#causal-language-modeling) to learn how to finetune DistilGPT-2 and use it for inference!

> [!TIP]
> For more information about text generation, check out the [text generation strategies](https://huggingface.co/docs/transformers/generation_strategies#generation-strategies) guide!

### Text classification

Text classification involves assigning predefined categories to text documents, such as sentiment analysis, topic classification, or spam detection.

[BERT](https://huggingface.co/docs/transformers/model_doc/bert) is an encoder-only model and is the first model to effectively implement deep bidirectionality to learn richer representations of the text by attending to words on both sides.

1. BERT uses [WordPiece](https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece) tokenization to generate a token embedding of the text. To tell the difference between a single sentence and a pair of sentences, a special `[SEP]` token is added to differentiate them. A special `[CLS]` token is added to the beginning of every sequence of text. The final output with the `[CLS]` token is used as the input to the classification head for classification tasks. BERT also adds a segment embedding to denote whether a token belongs to the first or second sentence in a pair of sentences.

2. BERT is pretrained with two objectives: masked language modeling and next-sentence prediction. In masked language modeling, some percentage of the input tokens are randomly masked, and the model needs to predict these. This solves the issue of bidirectionality, where the model could cheat and see all the words and "predict" the next word. The final hidden states of the predicted mask tokens are passed to a feedforward network with a softmax over the vocabulary to predict the masked word.

    The second pretraining object is next-sentence prediction. The model must predict whether sentence B follows sentence A. Half of the time sentence B is the next sentence, and the other half of the time, sentence B is a random sentence. The prediction, whether it is the next sentence or not, is passed to a feedforward network with a softmax over the two classes (`IsNext` and `NotNext`).

3. The input embeddings are passed through multiple encoder layers to output some final hidden states.

To use the pretrained model for text classification, add a sequence classification head on top of the base BERT model. The sequence classification head is a linear layer that accepts the final hidden states and performs a linear transformation to convert them into logits. The cross-entropy loss is calculated between the logits and target to find the most likely label.

Ready to try your hand at text classification? Check out our complete [text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) to learn how to finetune DistilBERT and use it for inference!

### Token classification

Token classification involves assigning a label to each token in a sequence, such as in named entity recognition or part-of-speech tagging.

To use BERT for token classification tasks like named entity recognition (NER), add a token classification head on top of the base BERT model. The token classification head is a linear layer that accepts the final hidden states and performs a linear transformation to convert them into logits. The cross-entropy loss is calculated between the logits and each token to find the most likely label.

Ready to try your hand at token classification? Check out our complete [token classification guide](https://huggingface.co/docs/transformers/tasks/token_classification) to learn how to finetune DistilBERT and use it for inference!

### Question answering

Question answering involves finding the answer to a question within a given context or passage.

To use BERT for question answering, add a span classification head on top of the base BERT model. This linear layer accepts the final hidden states and performs a linear transformation to compute the `span` start and end logits corresponding to the answer. The cross-entropy loss is calculated between the logits and the label position to find the most likely span of text corresponding to the answer.

Ready to try your hand at question answering? Check out our complete [question answering guide](https://huggingface.co/docs/transformers/tasks/question_answering) to learn how to finetune DistilBERT and use it for inference!

> [!TIP]
> 💡 Notice how easy it is to use BERT for different tasks once it's been pretrained. You only need to add a specific head to the pretrained model to manipulate the hidden states into your desired output!

### Summarization

Summarization involves condensing a longer text into a shorter version while preserving its key information and meaning.

Encoder-decoder models like [BART](https://huggingface.co/docs/transformers/model_doc/bart) and [T5](model_doc/t5) are designed for the sequence-to-sequence pattern of a summarization task. We'll explain how BART works in this section, and then you can try finetuning T5 at the end.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bart_architecture.png"/>
</div>

1. BART's encoder architecture is very similar to BERT and accepts a token and positional embedding of the text. BART is pretrained by corrupting the input and then reconstructing it with the decoder. Unlike other encoders with specific corruption strategies, BART can apply any type of corruption. The *text infilling* corruption strategy works the best though. In text infilling, a number of text spans are replaced with a **single** [`mask`] token. This is important because the model has to predict the masked tokens, and it teaches the model to predict the number of missing tokens. The input embeddings and masked spans are passed through the encoder to output some final hidden states, but unlike BERT, BART doesn't add a final feedforward network at the end to predict a word.

2. The encoder's output is passed to the decoder, which must predict the masked tokens and any uncorrupted tokens from the encoder's output. This gives additional context to help the decoder restore the original text. The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The cross-entropy loss is calculated between the logits and the label, which is just the token shifted to the right.

Ready to try your hand at summarization? Check out our complete [summarization guide](https://huggingface.co/docs/transformers/tasks/summarization) to learn how to finetune T5 and use it for inference!

> [!TIP]
> For more information about text generation, check out the [text generation strategies](https://huggingface.co/docs/transformers/generation_strategies) guide!

### Translation

Translation involves converting text from one language to another while preserving its meaning. Translation is another example of a sequence-to-sequence task, which means you can use an encoder-decoder model like [BART](https://huggingface.co/docs/transformers/model_doc/bart) or [T5](model_doc/t5) to do it. We'll explain how BART works in this section, and then you can try finetuning T5 at the end.

BART adapts to translation by adding a separate randomly initialized encoder to map a source language to an input that can be decoded into the target language. This new encoder's embeddings are passed to the pretrained encoder instead of the original word embeddings. The source encoder is trained by updating the source encoder, positional embeddings, and input embeddings with the cross-entropy loss from the model output. The model parameters are frozen in this first step, and all the model parameters are trained together in the second step.
BART has since been followed up by a multilingual version, mBART, intended for translation and pretrained on many different languages.

Ready to try your hand at translation? Check out our complete [translation guide](https://huggingface.co/docs/transformers/tasks/translation) to learn how to finetune T5 and use it for inference!

> [!TIP]
> As you've seen throughout this guide, many models follow similar patterns despite addressing different tasks. Understanding these common patterns can help you quickly grasp how new models work and how to adapt existing models to your specific needs.

## Modalities beyond text

Transformers are not limited to text. They can also be applied to other modalities like speech and audio, images, and video. Of course, on this course we will focus on text, but we can briefly introduce the other modalities.

### Speech and audio

Let's start by exploring how Transformer models handle speech and audio data, which presents unique challenges compared to text or images.

[Whisper](https://huggingface.co/docs/transformers/main/en/model_doc/whisper) is a encoder-decoder (sequence-to-sequence) transformer pretrained on 680,000 hours of labeled audio data. This amount of pretraining data enables zero-shot performance on audio tasks in English and many other languages. The decoder allows Whisper to map the encoders learned speech representations to useful outputs, such as text, without additional fine-tuning. Whisper just works out of the box.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/whisper_architecture.png"/>
</div>

Diagram is from [Whisper paper](https://huggingface.co/papers/2212.04356).

This model has two main components:

1. An **encoder** processes the input audio. The raw audio is first converted into a log-Mel spectrogram. This spectrogram is then passed through a Transformer encoder network.

2. A **decoder** takes the encoded audio representation and autoregressively predicts the corresponding text tokens. It's a standard Transformer decoder trained to predict the next text token given the previous tokens and the encoder output. Special tokens are used at the beginning of the decoder input to steer the model towards specific tasks like transcription, translation, or language identification.

Whisper was pretrained on a massive and diverse dataset of 680,000 hours of labeled audio data collected from the web. This large-scale, weakly supervised pretraining is the key to its strong zero-shot performance across many languages and tasks.

Now that Whisper is pretrained, you can use it directly for zero-shot inference or finetune it on your data for improved performance on specific tasks like automatic speech recognition or speech translation!

> [!TIP]
> The key innovation in Whisper is its training on an unprecedented scale of diverse, weakly supervised audio data from the internet. This allows it to generalize remarkably well to different languages, accents, and tasks without task-specific finetuning.

### Automatic speech recognition

To use the pretrained model for automatic speech recognition, you leverage its full encoder-decoder structure. The encoder processes the audio input, and the decoder autoregressively generates the transcript token by token. When fine-tuning, the model is typically trained using a standard sequence-to-sequence loss (like cross-entropy) to predict the correct text tokens based on the audio input.

The easiest way to use a fine-tuned model for inference is within a `pipeline`.

```python
from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-base.en")
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# Output: {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

Ready to try your hand at automatic speech recognition? Check out our complete [automatic speech recognition guide](https://huggingface.co/docs/transformers/tasks/asr) to learn how to finetune Whisper and use it for inference!

### Computer vision

Now let's move on to computer vision tasks, which deal with understanding and interpreting visual information from images or videos.

There are two ways to approach computer vision tasks:

1. Split an image into a sequence of patches and process them in parallel with a Transformer.
2. Use a modern CNN, like [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext), which relies on convolutional layers but adopts modern network designs.

> [!TIP]
> A third approach mixes Transformers with convolutions (for example, [Convolutional Vision Transformer](https://huggingface.co/docs/transformers/model_doc/cvt) or [LeViT](https://huggingface.co/docs/transformers/model_doc/levit)). We won't discuss those because they just combine the two approaches we examine here.

ViT and ConvNeXT are commonly used for image classification, but for other vision tasks like object detection, segmentation, and depth estimation, we'll look at DETR, Mask2Former and GLPN, respectively; these models are better suited for those tasks.

### Image classification

Image classification is one of the fundamental computer vision tasks. Let's see how different model architectures approach this problem.

ViT and ConvNeXT can both be used for image classification; the main difference is that ViT uses an attention mechanism while ConvNeXT uses convolutions.

[ViT](https://huggingface.co/docs/transformers/model_doc/vit) replaces convolutions entirely with a pure Transformer architecture. If you're familiar with the original Transformer, then you're already most of the way toward understanding ViT.  

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg"/>
</div>

The main change ViT introduced was in how images are fed to a Transformer:

1. An image is split into square non-overlapping patches, each of which gets turned into a vector or *patch embedding*. The patch embeddings are generated from a convolutional 2D layer which creates the proper input dimensions (which for a base Transformer is 768 values for each patch embedding). If you had a 224x224 pixel image, you could split it into 196 16x16 image patches. Just like how text is tokenized into words, an image is "tokenized" into a sequence of patches.

2. A *learnable embedding* - a special `[CLS]` token - is added to the beginning of the patch embeddings just like BERT. The final hidden state of the `[CLS]` token is used as the input to the attached classification head; other outputs are ignored. This token helps the model learn how to encode a representation of the image.

3. The last thing to add to the patch and learnable embeddings are the *position embeddings* because the model doesn't know how the image patches are ordered. The position embeddings are also learnable and have the same size as the patch embeddings. Finally, all of the embeddings are passed to the Transformer encoder.

4. The output, specifically only the output with the `[CLS]` token, is passed to a multilayer perceptron head (MLP). ViT's pretraining objective is simply classification. Like other classification heads, the MLP head converts the output into logits over the class labels and calculates the cross-entropy loss to find the most likely class.

Ready to try your hand at image classification? Check out our complete [image classification guide](https://huggingface.co/docs/transformers/tasks/image_classification) to learn how to fine-tune ViT and use it for inference!  


> [!TIP]
> Notice the parallel between ViT and BERT: both use a special token (<code>[CLS]</code>) to capture the overall representation, both add position information to their embeddings, and both use a Transformer encoder to process the sequence of tokens/patches.


---

<!-- Section 1.6 -->

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

# Transformer Architectures[[transformer-architectures]]

In the previous sections, we introduced the general Transformer architecture and explored how these models can solve various tasks. Now, let's take a closer look at the three main architectural variants of Transformer models and understand when to use each one. Then, we look at how those architectures are applied to different language tasks. 

In this section, we're going to dive deeper into the three main architectural variants of Transformer models and understand when to use each one.


> [!TIP]
> Remember that most Transformer models use one of three architectures: encoder-only, decoder-only, or encoder-decoder (sequence-to-sequence). Understanding these differences will help you choose the right model for your specific task.

## Encoder models[[encoder-models]]

<Youtube id="MUqNwgPjJvQ" />

Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having "bi-directional" attention, and are often called *auto-encoding models*.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.

Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition (and more generally word classification), and extractive question answering.

> [!TIP]
> As we saw in [How 🤗 Transformers solve tasks](https://huggingface.co/learn/llm-course/chapter1/5), encoder models like BERT excel at understanding text because they can look at the entire context in both directions. This makes them perfect for tasks where comprehension of the whole input is important.

Representatives of this family of models include:

- [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [ModernBERT](https://huggingface.co/docs/transformers/en/model_doc/modernbert)

## Decoder models[[decoder-models]]

<Youtube id="d_ixlCubqQw" />

Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called *auto-regressive models*.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.

These models are best suited for tasks involving text generation.

> [!TIP]
> Decoder models like GPT are designed to generate text by predicting one token at a time. As we explored in [How 🤗 Transformers solve tasks](https://huggingface.co/learn/llm-course/chapter1/5), they can only see previous tokens, which makes them excellent for creative text generation but less ideal for tasks requiring bidirectional understanding.

Representatives of this family of models include:

- [Hugging Face SmolLM Series](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Meta's Llama Series](https://huggingface.co/docs/transformers/en/model_doc/llama4)
- [Google's Gemma Series](https://huggingface.co/docs/transformers/main/en/model_doc/gemma3)
- [DeepSeek's V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

### Modern Large Language Models (LLMs)

Most modern Large Language Models (LLMs) use the decoder-only architecture. These models have grown dramatically in size and capabilities over the past few years, with some of the largest models containing hundreds of billions of parameters.

Modern LLMs are typically trained in two phases:
1. **Pretraining**: The model learns to predict the next token on vast amounts of text data
2. **Instruction tuning**: The model is fine-tuned to follow instructions and generate helpful responses

This approach has led to models that can understand and generate human-like text across a wide range of topics and tasks.

#### Key capabilities of modern LLMs

Modern decoder-based LLMs have demonstrated impressive capabilities:

| Capability | Description | Example |
|------------|-------------|---------|
| Text generation | Creating coherent and contextually relevant text | Writing essays, stories, or emails |
| Summarization | Condensing long documents into shorter versions | Creating executive summaries of reports |
| Translation | Converting text between languages | Translating English to Spanish |
| Question answering | Providing answers to factual questions | "What is the capital of France?" |
| Code generation | Writing or completing code snippets | Creating a function based on a description |
| Reasoning | Working through problems step by step | Solving math problems or logical puzzles |
| Few-shot learning | Learning from a few examples in the prompt | Classifying text after seeing just 2-3 examples |

You can experiment with decoder-based LLMs directly in your browser via model repo pages on the Hub. Here's an example with the classic [GPT-2](https://huggingface.co/openai-community/gpt2) (OpenAI's finest open source model!):

<a 
  href="https://huggingface.co/openai-community/gpt2" target="_blank">
  View GPT-2 model on Hugging Face
</a>

## Sequence-to-sequence models[[sequence-to-sequence-models]]

<Youtube id="0_4KEb08xrE" />

Encoder-decoder models (also called *sequence-to-sequence models*) use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.

The pretraining of these models can take different forms, but it often involves reconstructing a sentence for which the input has been somehow corrupted (for instance by masking random words). The pretraining of the T5 model consists of replacing random spans of text (that can contain several words) with a single mask special token, and the task is then to predict the text that this mask token replaces.

Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering.

> [!TIP]
> As we saw in [How 🤗 Transformers solve tasks](https://huggingface.co/learn/llm-course/chapter1/5), encoder-decoder models like BART and T5 combine the strengths of both architectures. The encoder provides deep bidirectional understanding of the input, while the decoder generates appropriate output text. This makes them perfect for tasks that transform one sequence into another, like translation or summarization.

### Practical applications

Sequence-to-sequence models excel at tasks that require transforming one form of text into another while preserving meaning. Some practical applications include:

| Application | Description | Example Model |
|-------------|-------------|---------------|
| Machine translation | Converting text between languages | Marian, T5 |
| Text summarization | Creating concise summaries of longer texts | BART, T5 |
| Data-to-text generation | Converting structured data into natural language | T5 |
| Grammar correction | Fixing grammatical errors in text | T5 |
| Question answering | Generating answers based on context | BART, T5 |

Here's an interactive demo of a sequence-to-sequence model for translation:

<iframe
	src="https://course-demos-speech-to-speech-translation.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

Representatives of this family of models include:

- [BART](https://huggingface.co/docs/transformers/model_doc/bart)
- [mBART](https://huggingface.co/docs/transformers/model_doc/mbart)
- [Marian](https://huggingface.co/docs/transformers/model_doc/marian)
- [T5](https://huggingface.co/docs/transformers/model_doc/t5)

## Choosing the right architecture[[choosing-the-right-architecture]]

When working on a specific NLP task, how do you decide which architecture to use? Here's a quick guide:

| Task | Suggested Architecture | Examples |
|------|------------------------|----------|
| Text classification (sentiment, topic) | Encoder | BERT, RoBERTa |
| Text generation (creative writing) | Decoder | GPT, LLaMA |
| Translation | Encoder-Decoder | T5, BART |
| Summarization | Encoder-Decoder | BART, T5 |
| Named entity recognition | Encoder | BERT, RoBERTa |
| Question answering (extractive) | Encoder | BERT, RoBERTa |
| Question answering (generative) | Encoder-Decoder or Decoder | T5, GPT |
| Conversational AI | Decoder | GPT, LLaMA |

> [!TIP]
> When in doubt about which model to use, consider:  
>
> 1. What kind of understanding does your task need? (Bidirectional or unidirectional)  
> 2. Are you generating new text or analyzing existing text?  
> 3. Do you need to transform one sequence into another?  
>
> The answers to these questions will guide you toward the right architecture. 

## The evolution of LLMs

Large Language Models have evolved rapidly in recent years, with each generation bringing significant improvements in capabilities. 

## Attention mechanisms[[attention-mechanisms]]

Most transformer models use full attention in the sense that the attention matrix is square. It can be a big
computational bottleneck when you have long texts. Longformer and reformer are models that try to be more efficient and
use a sparse version of the attention matrix to speed up training.

> [!TIP]
> Standard attention mechanisms have a computational complexity of O(n²), where n is the sequence length. This becomes problematic for very long sequences. The specialized attention mechanisms below help address this limitation.

### LSH attention

[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer) uses LSH attention. In the softmax(QK^t), only the biggest elements (in the softmax dimension) of the matrix QK^t are going to give useful contributions. So for each query q in Q, we can consider only
the keys k in K that are close to q. A hash function is used to determine if q and k are close. The attention mask is
modified to mask the current token (except at the first position), because it will give a query and a key equal (so
very similar to each other). Since the hash can be a bit random, several hash functions are used in practice
(determined by a n_rounds parameter) and then are averaged together.

### Local attention

[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer) uses local attention: often, the local context (e.g., what are the two tokens to the  left and right?) is enough to take action for a given token. Also, by stacking attention layers that have a small
window, the last layer will have a receptive field of more than just the tokens in the window, allowing them to build a
representation of the whole sentence.

Some preselected input tokens are also given global attention: for those few tokens, the attention matrix can access
all tokens and this process is symmetric: all other tokens have access to those specific tokens (on top of the ones in
their local window). This is shown in Figure 2d of the paper, see below for a sample attention mask:

<div class="flex justify-center">
    <img scale="50 %" align="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png"/>
</div>

Using those attention matrices with less parameters then allows the model to have inputs having a bigger sequence
length.

### Axial positional encodings

[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer) uses axial positional encodings: in traditional transformer models, the positional encoding
E is a matrix of size \\(l\\) by \\(d\\), \\(l\\) being the sequence length and \\(d\\) the dimension of the
hidden state. If you have very long texts, this matrix can be huge and take way too much space on the GPU. To alleviate
that, axial positional encodings consist of factorizing that big matrix E in two smaller matrices E1 and E2, with
dimensions \\(l_{1} \times d_{1}\\) and \\(l_{2} \times d_{2}\\), such that \\(l_{1} \times l_{2} = l\\) and
\\(d_{1} + d_{2} = d\\) (with the product for the lengths, this ends up being way smaller). The embedding for time
step \\(j\\) in E is obtained by concatenating the embeddings for timestep \\(j \% l1\\) in E1 and \\(j // l1\\)
in E2.

## Conclusion[[conclusion]]

In this section, we've explored the three main Transformer architectures and some specialized attention mechanisms. Understanding these architectural differences is crucial for selecting the right model for your specific NLP task.

As we move forward in the course, you'll get hands-on experience with these different architectures and learn how to fine-tune them for your specific needs. In the next section, we'll look at some of the limitations and biases present in these models that you should be aware of when deploying them.


---

<!-- Section 1.7 -->

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# Ungraded quiz[[ungraded-quiz]]

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

So far, this chapter has covered a lot of ground! Don't worry if you didn't grasp all the details, but it's to reflect on what you've learned so far with a quiz. 

This quiz is ungraded, so you can try it as many times as you want. If you struggle with some questions, follow the tips and revisit the material. You'll be quizzed on this material again in the certification exam.

### 1. Explore the Hub and look for the `roberta-large-mnli` checkpoint. What task does it perform?


<Question
	choices={[
		{
			text: "Summarization",
			explain: "Look again on the <a href=\"https://huggingface.co/roberta-large-mnli\">roberta-large-mnli page</a>."
		},
		{
			text: "Text classification",
			explain: "More precisely, it classifies if two sentences are logically linked across three labels (contradiction, neutral, entailment) — a task also called <em>natural language inference</em>.",
			correct: true
		},
		{
			text: "Text generation",
			explain: "Look again on the <a href=\"https://huggingface.co/roberta-large-mnli\">roberta-large-mnli page</a>."
		}
	]}
/>

### 2. What will the following code return?

```py
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

<Question
	choices={[
		{
			text: "It will return classification scores for this sentence, with labels \"positive\" or \"negative\".",
			explain: "This is incorrect — this would be a <code>sentiment-analysis</code> pipeline."
		},
		{
			text: "It will return a generated text completing this sentence.",
			explain: "This is incorrect — it would be a <code>text-generation</code> pipeline.",
		},
		{
			text: "It will return the words representing persons, organizations or locations.",
			explain: "Furthermore, with <code>aggregation_strategy=&quot;simple&quot;</code>, it will group together the words belonging to the same entity, like \"Hugging Face\".",
			correct: true
		}
	]}
/>

### 3. What should replace ... in this code sample?

```py
from transformers import pipeline

filler = pipeline("fill-mask", model="bert-base-cased")
result = filler("...")
```

<Question
	choices={[
		{
			text: "This &#60;mask> has been waiting for you.",
			explain: "This is incorrect. Check out the <code>bert-base-cased</code> model card and try to spot your mistake."
		},
		{
			text: "This [MASK] has been waiting for you.",
			explain: "This model's mask token is [MASK].",
			correct: true
		},
		{
			text: "This man has been waiting for you.",
			explain: "This is incorrect. This pipeline fills in masked words, so it needs a mask token somewhere."
		}
	]}
/>

### 4. Why will this code fail?

```py
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library")
```

<Question
	choices={[
		{
			text: "This pipeline requires that labels be given to classify this text.",
			explain: "Right — the correct code needs to include <code>candidate_labels=[...]</code>.",
			correct: true
		},
		{
			text: "This pipeline requires several sentences, not just one.",
			explain: "This is incorrect, though when properly used, this pipeline can take a list of sentences to process (like all other pipelines)."
		},
		{
			text: "The 🤗 Transformers library is broken, as usual.",
			explain: "We won't dignify this answer with a comment!"
		},
		{
			text: "This pipeline requires longer inputs; this one is too short.",
			explain: "This is incorrect. Note that a very long text will be truncated when processed by this pipeline."
		}
	]}
/>

### 5. What does "transfer learning" mean?

<Question
	choices={[
		{
			text: "Transferring the knowledge of a pretrained model to a new model by training it on the same dataset.",
			explain: "No, that would be two versions of the same model."
		},
		{
			text: "Transferring the knowledge of a pretrained model to a new model by initializing the second model with the first model's weights.",
			explain: "When the second model is trained on a new task, it *transfers* the knowledge of the first model.",
			correct: true
		},
		{
			text: "Transferring the knowledge of a pretrained model to a new model by building the second model with the same architecture as the first model.",
			explain: "The architecture is just the way the model is built; there is no knowledge shared or transferred in this case."
		}
	]}
/>

### 6. True or false? A language model usually does not need labels for its pretraining.

<Question
	choices={[
		{
			text: "True",
			explain: "The pretraining is usually <em>self-supervised</em>, which means the labels are created automatically from the inputs (like predicting the next word or filling in some masked words).",
			correct: true
		},
		{
			text: "False",
			explain: "This is not the correct answer."
		}
	]}
/>

### 7. Select the sentence that best describes the terms "model", "architecture", and "weights".

<Question
	choices={[
		{
			text: "If a model is a building, its architecture is the blueprint and the weights are the people living inside.",
			explain: "Following this metaphor, the weights would be the bricks and other materials used to construct the building."
		},
		{
			text: "An architecture is a map to build a model and its weights are the cities represented on the map.",
			explain: "The problem with this metaphor is that a map usually represents one existing reality (there is only one city in France named Paris). For a given architecture, multiple weights are possible."
		},
		{
			text: "An architecture is a succession of mathematical functions to build a model and its weights are those functions parameters.",
			explain: "The same set of mathematical functions (architecture) can be used to build different models by using different parameters (weights).",
			correct: true
		}
	]}
/>


### 8. Which of these types of models would you use for completing prompts with generated text?

<Question
	choices={[
		{
			text: "An encoder model",
			explain: "An encoder model generates a representation of the whole sentence that is better suited for tasks like classification."
		},
		{
			text: "A decoder model",
			explain: "Decoder models are perfectly suited for text generation from a prompt.",
			correct: true
		},
		{
			text: "A sequence-to-sequence model",
			explain: "Sequence-to-sequence models are better suited for tasks where you want to generate sentences in relation to the input sentences, not a given prompt."
		}
	]}
/>

### 9. Which of those types of models would you use for summarizing texts?

<Question
	choices={[
		{
			text: "An encoder model",
			explain: "An encoder model generates a representation of the whole sentence that is better suited for tasks like classification."
		},
		{
			text: "A decoder model",
			explain: "Decoder models are good for generating output text (like summaries), but they don't have the ability to exploit a context like the whole text to summarize."
		},
		{
			text: "A sequence-to-sequence model",
			explain: "Sequence-to-sequence models are perfectly suited for a summarization task.",
			correct: true
		}
	]}
/>

### 10. Which of these types of models would you use for classifying text inputs according to certain labels?

<Question
	choices={[
		{
			text: "An encoder model",
			explain: "An encoder model generates a representation of the whole sentence which is perfectly suited for a task like classification.",
			correct: true
		},
		{
			text: "A decoder model",
			explain: "Decoder models are good for generating output texts, not extracting a label out of a sentence."
		},
		{
			text: "A sequence-to-sequence model",
			explain: "Sequence-to-sequence models are better suited for tasks where you want to generate text based on an input sentence, not a label.",
		}
	]}
/>

### 11. What possible source can the bias observed in a model have?

<Question
	choices={[
		{
			text: "The model is a fine-tuned version of a pretrained model and it picked up its bias from it.",
			explain: "When applying Transfer Learning, the bias in the pretrained model used persists in the fine-tuned model.",
			correct: true
		},
		{
			text: "The data the model was trained on is biased.",
			explain: "This is the most obvious source of bias, but not the only one.",
			correct: true
		},
		{
			text: "The metric the model was optimizing for is biased.",
			explain: "A less obvious source of bias is the way the model is trained. Your model will blindly optimize for whatever metric you chose, without any second thoughts.",
			correct: true
		}
	]}
/>


---

<!-- Section 1.8 -->

# Deep dive into Text Generation Inference with LLMs[[inference-with-llms]]

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

<Youtube id="Xp2w1_LKZN4" />

So far, we've explored the transformer architecture in relation to a range of discrete tasks, like text classification or summarization. However, Large Language Models are most used for text generation, and this is what we'll explore in this chapter.

In this page, we'll explore the core concepts behind LLM inference, providing a comprehensive understanding of how these models generate text and the key components involved in the inference process.

## Understanding the Basics

Let's start with the fundamentals. Inference is the process of using a trained LLM to generate human-like text from a given input prompt. Language models use their knowledge from training to formulate responses one word at a time. The model leverages learned probabilities from billions of parameters to predict and generate the next token in a sequence. This sequential generation is what allows LLMs to produce coherent and contextually relevant text. 

## The Role of Attention

The attention mechanism is what gives LLMs their ability to understand context and generate coherent responses. When predicting the next word, not every word in a sentence carries equal weight - for example, in the sentence *"The capital of France is ..."*, the words "France" and "capital" are crucial for determining that "Paris" should come next. This ability to focus on relevant information is what we call attention.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AttentionSceneFinal.gif" alt="Visual Gif of Attention" width="60%">

This process of identifying the most relevant words to predict the next token has proven to be incredibly effective. Although the basic principle of training LLMs—predicting the next token—has remained generally consistent since BERT and GPT-2, there have been significant advancements in scaling neural networks and making the attention mechanism work for longer and longer sequences, at lower and lower costs.

> [!TIP]
> In short, the attention mechanism is the key to LLMs being able to generate text that is both coherent and context-aware. It sets modern LLMs apart from previous generations of language models.

### Context Length and Attention Span

Now that we understand attention, let's explore how much context an LLM can actually handle. This brings us to context length, or the model's 'attention span'.

The context length refers to the maximum number of tokens (words or parts of words) that the LLM can process at once. Think of it as the size of the model's working memory.

These capabilities are limited by several practical factors:
- The model's architecture and size
- Available computational resources
- The complexity of the input and desired output

In an ideal world, we could feed unlimited context to the model, but hardware constraints and computational costs make this impractical. This is why different models are designed with different context lengths to balance capability with efficiency.

> [!TIP]
> The context length is the maximum number of tokens the model can consider at once when generating a response.

### The Art of Prompting

When we pass information to LLMs, we structure our input in a way that guides the generation of the LLM toward the desired output. This is called _prompting_.

Understanding how LLMs process information helps us craft better prompts. Since the model's primary task is to predict the next token by analyzing the importance of each input token, the wording of your input sequence becomes crucial.

> [!TIP]
> Careful design of the prompt makes it easier **to guide the generation of the LLM toward the desired output**.

## The Two-Phase Inference Process

Now that we understand the basic components, let's dive into how LLMs actually generate text. The process can be broken down into two main phases: prefill and decode. These phases work together like an assembly line, each playing a crucial role in producing coherent text.

### The Prefill Phase

The prefill phase is like the preparation stage in cooking - it's where all the initial ingredients are processed and made ready. This phase involves three key steps:

1. **Tokenization**: Converting the input text into tokens (think of these as the basic building blocks the model understands)
2. **Embedding Conversion**: Transforming these tokens into numerical representations that capture their meaning
3. **Initial Processing**: Running these embeddings through the model's neural networks to create a rich understanding of the context

This phase is computationally intensive because it needs to process all input tokens at once. Think of it as reading and understanding an entire paragraph before starting to write a response.

You can experiment with different tokenizers in the interactive playground below:

<iframe
	src="https://agents-course-the-tokenizer-playground.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

### The Decode Phase

After the prefill phase has processed the input, we move to the decode phase - this is where the actual text generation happens. The model generates one token at a time in what we call an autoregressive process (where each new token depends on all previous tokens).

The decode phase involves several key steps that happen for each new token:
1. **Attention Computation**: Looking back at all previous tokens to understand context
2. **Probability Calculation**: Determining the likelihood of each possible next token
3. **Token Selection**: Choosing the next token based on these probabilities
4. **Continuation Check**: Deciding whether to continue or stop generation

This phase is memory-intensive because the model needs to keep track of all previously generated tokens and their relationships. 

## Sampling Strategies

Now that we understand how the model generates text, let's explore the various ways we can control this generation process. Just like a writer might choose between being more creative or more precise, we can adjust how the model makes its token selections.

You can interact with the basic decoding process yourself with SmolLM2 in this Space (remember, it decodes until reaching an **EOS** token which is  **<|im_end|>** for this model):

<iframe
	src="https://agents-course-decoding-visualizer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

### Understanding Token Selection: From Probabilities to Token Choices

When the model needs to choose the next token, it starts with raw probabilities (called logits) for every word in its vocabulary. But how do we turn these probabilities into actual choices? Let's break down the process:

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/1.png)  

1. **Raw Logits**: Think of these as the model's initial gut feelings about each possible next word
2. **Temperature Control**: Like a creativity dial - higher settings (>1.0) make choices more random and creative, lower settings (<1.0) make them more focused and deterministic
3. **Top-p (Nucleus) Sampling**: Instead of considering all possible words, we only look at the most likely ones that add up to our chosen probability threshold (e.g., top 90%)
4. **Top-k Filtering**: An alternative approach where we only consider the k most likely next words

### Managing Repetition: Keeping Output Fresh

One common challenge with LLMs is their tendency to repeat themselves - much like a speaker who keeps returning to the same points. To address this, we use two types of penalties:

1. **Presence Penalty**: A fixed penalty applied to any token that has appeared before, regardless of how often. This helps prevent the model from reusing the same words.
2. **Frequency Penalty**: A scaling penalty that increases based on how often a token has been used. The more a word appears, the less likely it is to be chosen again.

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/2.png)  

These penalties are applied early in the token selection process, adjusting the raw probabilities before other sampling strategies are applied. Think of them as gentle nudges encouraging the model to explore new vocabulary.

### Controlling Generation Length: Setting Boundaries

Just as a good story needs proper pacing and length, we need ways to control how much text our LLM generates. This is crucial for practical applications - whether we're generating a tweet-length response or a full blog post.

We can control generation length in several ways:
1. **Token Limits**: Setting minimum and maximum token counts
2. **Stop Sequences**: Defining specific patterns that signal the end of generation
3. **End-of-Sequence Detection**: Letting the model naturally conclude its response

For example, if we want to generate a single paragraph, we might set a maximum of 100 tokens and use "\n\n" as a stop sequence. This ensures our output stays focused and appropriately sized for its purpose.

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/3.png)  

### Beam Search: Looking Ahead for Better Coherence

While the strategies we've discussed so far make decisions one token at a time, beam search takes a more holistic approach. Instead of committing to a single choice at each step, it explores multiple possible paths simultaneously - like a chess player thinking several moves ahead.

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/4.png)  

Here's how it works:
1. At each step, maintain multiple candidate sequences (typically 5-10)
2. For each candidate, compute probabilities for the next token
3. Keep only the most promising combinations of sequences and next tokens
4. Continue this process until reaching the desired length or stop condition
5. Select the sequence with the highest overall probability

You can explore beam search visually here:

<iframe
	src="https://agents-course-beam-search-visualizer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

This approach often produces more coherent and grammatically correct text, though it requires more computational resources than simpler methods.

## Practical Challenges and Optimization

As we wrap up our exploration of LLM inference, let's look at the practical challenges you'll face when deploying these models, and how to measure and optimize their performance.

### Key Performance Metrics

When working with LLMs, four critical metrics will shape your implementation decisions:

1. **Time to First Token (TTFT)**: How quickly can you get the first response? This is crucial for user experience and is primarily affected by the prefill phase.
2. **Time Per Output Token (TPOT)**: How fast can you generate subsequent tokens? This determines the overall generation speed.
3. **Throughput**: How many requests can you handle simultaneously? This affects scaling and cost efficiency.
4. **VRAM Usage**: How much GPU memory do you need? This often becomes the primary constraint in real-world applications.

### The Context Length Challenge

One of the most significant challenges in LLM inference is managing context length effectively. Longer contexts provide more information but come with substantial costs:

- **Memory Usage**: Grows quadratically with context length
- **Processing Speed**: Decreases linearly with longer contexts
- **Resource Allocation**: Requires careful balancing of VRAM usage

Recent models like [Qwen2.5-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M) offer impressive 1M token context windows, but this comes at the cost of significantly slower inference times. The key is finding the right balance for your specific use case.  


<div style="max-width: 800px; margin: 20px auto; padding: 20px; 
font-family: system-ui;">
    <div style="border: 2px solid #ddd; border-radius: 8px; 
    padding: 20px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; 
        margin-bottom: 15px;">
            <div style="flex: 1; text-align: center; padding: 
            10px; background: #f0f0f0; border-radius: 4px;">
                Input Text (Raw)
            </div>
            <div style="margin: 0 10px;">→</div>
            <div style="flex: 1; text-align: center; padding: 
            10px; background: #e1f5fe; border-radius: 4px;">
                Tokenized Input
            </div>
        </div>
        <div style="display: flex; margin-bottom: 15px;">
            <div style="flex: 1; border: 1px solid #ccc; 
            padding: 10px; margin: 5px; background: #e8f5e9; 
            border-radius: 4px; text-align: center;">
                Context Window<br/>(e.g., 4K tokens)
                <div style="display: flex; margin-top: 10px;">
                    <div style="flex: 1; background: #81c784; 
                    margin: 2px; height: 20px; border-radius: 
                    2px;"></div>
                    <div style="flex: 1; background: #81c784; 
                    margin: 2px; height: 20px; border-radius: 
                    2px;"></div>
                    <div style="flex: 1; background: #81c784; 
                    margin: 2px; height: 20px; border-radius: 
                    2px;"></div>
                    <div style="flex: 1; background: #81c784; 
                    margin: 2px; height: 20px; border-radius: 
                    2px;"></div>
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: 
        space-between; text-align: center; font-size: 0.9em; 
        color: #666;">
            <div style="flex: 1;">
                <div style="border: 1px solid #ffcc80; padding: 
                8px; margin: 5px; background: #fff3e0; 
                border-radius: 4px;">
                    Memory Usage<br/>∝ Length²
                </div>
            </div>
            <div style="flex: 1;">
                <div style="border: 1px solid #90caf9; padding: 
                8px; margin: 5px; background: #e3f2fd; 
                border-radius: 4px;">
                    Processing Time<br/>∝ Length
                </div>
            </div>
        </div>
    </div>
</div>

### The KV Cache Optimization

To address these challenges, one of the most powerful optimizations is KV (Key-Value) caching. This technique significantly improves inference speed by storing and reusing intermediate calculations. This optimization:
- Reduces repeated calculations
- Improves generation speed
- Makes long-context generation practical

The trade-off is additional memory usage, but the performance benefits usually far outweigh this cost.

## Conclusion

Understanding LLM inference is crucial for effectively deploying and optimizing these powerful models. We've covered the key components:

- The fundamental role of attention and context
- The two-phase inference process
- Various sampling strategies for controlling generation
- Practical challenges and optimizations

By mastering these concepts, you'll be better equipped to build applications that leverage LLMs effectively and efficiently.

Remember that the field of LLM inference is rapidly evolving, with new techniques and optimizations emerging regularly. Stay curious and keep experimenting with different approaches to find what works best for your specific use cases.


---

<!-- Section 1.9 -->

# Bias and limitations[[bias-and-limitations]]

<CourseFloatingBanner chapter={1}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter1/section8.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter1/section8.ipynb"},
]} />

If your intent is to use a pretrained model or a fine-tuned version in production, please be aware that, while these models are powerful tools, they come with limitations. The biggest of these is that, to enable pretraining on large amounts of data, researchers often scrape all the content they can find, taking the best as well as the worst of what is available on the internet. 

To give a quick illustration, let's go back to the example of a `fill-mask` pipeline with the BERT model:

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
```

```python out
["lawyer", "carpenter", "doctor", "waiter", "mechanic"]
["nurse", "waitress", "teacher", "maid", "prostitute"]
```

When asked to fill in the missing word in these two sentences, the model gives only one gender-free answer (waiter/waitress). The others are work occupations usually associated with one specific gender -- and yes, prostitute ended up in the top 5 possibilities the model associates with "woman" and "work." This happens even though BERT is one of the rare Transformer models not built by scraping data from all over the internet, but rather using apparently neutral data (it's trained on the [English Wikipedia](https://huggingface.co/datasets/wikipedia) and [BookCorpus](https://huggingface.co/datasets/bookcorpus) datasets). 

When you use these tools, you therefore need to keep in the back of your mind that the original model you are using could very easily generate sexist, racist, or homophobic content. Fine-tuning the model on your data won't make this intrinsic bias disappear.


---

<!-- Section 1.10 -->

# Summary[[summary]]

<CourseFloatingBanner
    chapter={1}
    classNames="absolute z-10 right-0 top-0"
/>

In this chapter, you've been introduced to the fundamentals of Transformer models, Large Language Models (LLMs), and how they're revolutionizing AI and beyond.

## Key concepts covered

### Natural Language Processing and LLMs

We explored what NLP is and how Large Language Models have transformed the field. You learned that:
- NLP encompasses a wide range of tasks from classification to generation
- LLMs are powerful models trained on massive amounts of text data
- These models can perform multiple tasks within a single architecture
- Despite their capabilities, LLMs have limitations including hallucinations and bias

### Transformer capabilities

You saw how the `pipeline()` function from 🤗 Transformers makes it easy to use pre-trained models for various tasks:
- Text classification, token classification, and question answering
- Text generation and summarization
- Translation and other sequence-to-sequence tasks
- Speech recognition and image classification

### Transformer architecture

We discussed how Transformer models work at a high level, including:
- The importance of the attention mechanism
- How transfer learning enables models to adapt to specific tasks
- The three main architectural variants: encoder-only, decoder-only, and encoder-decoder

### Model architectures and their applications
A key aspect of this chapter was understanding which architecture to use for different tasks:

| Model           | Examples                                   | Tasks                                                                            |
|-----------------|--------------------------------------------|----------------------------------------------------------------------------------|
| Encoder-only    | BERT, DistilBERT, ModernBERT               | Sentence classification, named entity recognition, extractive question answering |
| Decoder-only    | GPT, LLaMA, Gemma, SmolLM                  | Text generation, conversational AI, creative writing                             |
| Encoder-decoder | BART, T5, Marian, mBART                    | Summarization, translation, generative question answering                        |

### Modern LLM developments
You also learned about recent developments in the field:
- How LLMs have grown in size and capability over time
- The concept of scaling laws and how they guide model development
- Specialized attention mechanisms that help models process longer sequences
- The two-phase training approach of pretraining and instruction tuning

### Practical applications
Throughout the chapter, you've seen how these models can be applied to real-world problems:
- Using the Hugging Face Hub to find and use pre-trained models
- Leveraging the Inference API to test models directly in your browser
- Understanding which models are best suited for specific tasks

## Looking ahead

Now that you have a solid understanding of what Transformer models are and how they work at a high level, you're ready to dive deeper into how to use them effectively. In the next chapters, you'll learn how to:

- Use the Transformers library to load and fine-tune models
- Process different types of data for model input
- Adapt pre-trained models to your specific tasks
- Deploy models for practical applications

The foundation you've built in this chapter will serve you well as you explore more advanced topics and techniques in the coming sections.


---

<!-- Section 1.11 -->

# Exam Time!

It's time to put your knowledge to the test! We've prepared a short quiz for you to test your understanding of the concepts covered in this chapter.

To take the quiz, you will need to follow these steps:

1. Sign in to your Hugging Face account.
2. Answer the questions in the quiz.
3. Submit your answers.


## Multiple Choice Quiz

In this quiz, you will be asked to select the correct answer from a list of options. We'll test you on the fundamentals of supervised finetuning.

<iframe
	src="https://huggingface-course-chapter-1-exam.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

