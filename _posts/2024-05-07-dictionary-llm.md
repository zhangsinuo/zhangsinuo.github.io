---
layout: distill
title: 
  One-Hour Survey of Large Language Models —— Quick Mastery of New Techniques, Concepts, and Case Studies
description: 
  The rapid advancements in large language models (LLMs) have given rise to numerous technical terms, which can be overwhelming for researchers. Therefore, there is an urgent need to sort and summarize these terms for the LLM technology research community. To cater to this need, we proposed a blog named  &quot;Dictionary LLM&quot;, which organizes and explains the existing terminologies of LLMs, providing a convenient reference for researchers. This dictionary helps them quickly understand the basic concepts of LLMs and delve deeper into the technology related to large models. The &quot;Dictionary LLM&quot; is essential for keeping up with the rapid developments in this dynamic field, ensuring that researchers and practitioners have access to the latest terminology and concepts.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
   - name: An

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-dictionary-llm.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Pre-training Technologies
    subsections:
    - name: Encoder-Decoder Architecture
    - name: Causal Decoder Architecture
    - name: Prefix Decoder Architecture

  - name: Fine-tuning Methods
    subsections:
    - name: Supervised Fine-tuning
    - name: Reinforcement Learning From Human Feedback
    - name: Instruction Tuning
    - name: Alignment
    - name: Adapter Tuning
    - name: Prefix Tuning
    - name: Prompt Tuning
    - name: Low-Rank Adaptation
  - name: Prompt Engineering
    subsections:
    - name: Emergent Abilities
    - name: Scaling Law
    - name: In-context Learning
    - name: Context Window
    - name: Step-by-step reasoning
    - name: Chain-of-thought Prompt
  - name: References
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## Introduction

A large language model (LLM) is a type of language model notable for its ability to achieve general-purpose language understanding and generation with billions of parameters that require significant computational resources and vast data for training. These artificial neural networks, mainly transformers, are pre-trained using self-supervised and semi-supervised learning techniques. The boom in the field of LLM has attracted many researchers and practitioners to devote themselves to related research. Notable examples include OpenAI's GPT models (e.g., GPT-3.5 and GPT-4, used in ChatGPT), Google's PaLM (used in Bard), and Meta's LLaMa, as well as BLOOM, Ernie 3.0 Titan, and Anthropic's Claude 2, thus promoting the advancement of technology in this field. 

As technology advances rapidly, many technical terms have emerged, some of which may not be easily understood. For example, 'in-context learning' refers to the model's ability to understand and respond based on the provided context, while 'instruction tuning' involves refining the model to respond to specific instructions more effectively. This proliferation of terms can make it challenging for researchers and users of LLMs to stay abreast of the latest advancements, creating unfavorable conditions for developing LLM technology. Therefore, sorting out and summarizing these proprietary terms has become an urgent demand for the LLM technology research community. To address this issue, this blog has launched "Dictionary LLM", which aims to organize and explain the existing terminologies of LLMs and provides a convenient reference for related researchers so that they can quickly understand the basic concepts of LLMs and then deeply catch the technology related to large models. This blog and the 'Dictionary LLM' are particularly useful for researchers, practitioners, and anyone interested in the field of LLMs, offering a comprehensive resource for both newcomers and experienced professionals.

In the following parts, Section 2 will explore pre-training technologies, detailing the training datasets used, the architectural nuances of various models, and the principles of model alignment with human values and expectations. Section 3 will delve into the nuances of fine-tuning methods, focusing on the art and science of prompt engineering and its impact on model performance. Finally, Section 4 will discuss the varied applications and challenges in LLM research, addressing critical issues like model-generated hallucinations and their implications.

## Pre-training Technologies

Pre-training is the first stage in the training process of machine learning models. It involves training the model on a large dataset to learn general features or representations of the data. These general features often include linguistic structures, syntax, common phrases, and contextual relationships, which form the foundational understanding of the model. Once pre-training is complete, fine-tuning follows, where the model is further trained on a smaller, task-specific dataset to specialize it for particular tasks. The pre-training phase helps the model leverage knowledge from a vast amount of data, making it more proficient in handling specific tasks with less data during the fine-tuning phase. This large dataset usually comprises diverse text sources, ranging from books and articles to websites, encompassing a wide array of topics and styles. This approach is commonly used in training large language models and deep learning models. Here, we briefly introduce three typical pre-training architectures for LLMs and an important technology, alignment.

<div class="row mt-3">
  {% include figure.html path="assets/img/2024-05-07-dictionary-llm/1.png" class="img-fluid" %}
</div>
<div class="caption">
    A comparison of the attention patterns in three mainstream architectures. Here, the yellow, green, blue, and grey rectangles indicate the encoder-encoder attention, encoder-decoder attention, decoder-decoder attention, and masked attention, respectively.
</div>

### Encoder-Decoder Architecture

**Definition:** The Encoder-Decoder Architecture is a fundamental concept in machine translation and sequence-to-sequence tasks. It is composed of two parts:

1. Encoder: This component processes the input sequence and encodes it into a context vector of fixed size, capturing all the relevant information of the sequence.

2. Decoder: This component takes the context vector generated by the encoder and produces an output sequence step-by-step. It often translates a sequence into another language or generates responses in a dialogue system.

This architecture is useful in handling sequences of varying lengths and is crucial in many Natural Language Processing (NLP) and time-series prediction tasks. So far, only a few LLMs are built based on the encoder-decoder architecture, e.g., Flan-T5 \cite{vaswani2017attention}.

**Case Studies:** Take English to French Translation as an example.

1. Encoder: Processes an English sentence ("The weather is nice today") by converting words into vectors and generating a context vector encapsulating the sentence's meaning.

2. Decoder: Uses the context vector to sequentially generate a translated French sentence ("Le temps est agréable aujourd'hui"), considering the context and previously generated words.

3. Key Feature: Attention mechanism, allowing the decoder to focus on relevant parts of the input for each translated word.

4. Outcome: The architecture enables effective translation by handling varying lengths and complexities in language sequences.

**Related Work:** More information can be seen in Vaswani et al. [2017] <d-cite key="vaswani2017attention"></d-cite>. 

The Encoder-Decoder Architecture <d-cite key="vaswani2017attention"></d-cite> is a fundamental concept in machine translation and sequence-to-sequence tasks. It is composed of two parts:
1. Encoder: This component processes the input sequence and encodes it into a context vector of fixed size, capturing all the relevant information of the sequence.
2. Decoder: This component takes the context vector generated by the encoder and produces an output sequence step-by-step. It often translates a sequence into another language or generates responses in a dialogue system.
This architecture is useful in handling sequences of varying lengths and is crucial in many Natural Language Processing (NLP) and time-series prediction tasks. So far, only a few LLMs are built based on the encoder-decoder architecture, e.g., Flan-T5  <d-cite key="chung2022scaling"></d-cite>.

### Causal Decoder Architecture

**Definition:** The causal decoder architecture incorporates the unidirectional attention mask to guarantee that each input token can only attend to the past tokens and itself. The input and output tokens are processed similarly through the decoder. This feature is particularly useful for language modeling, where each word's prediction depends solely on the previous words. This architecture is beneficial in maintaining the causal integrity of sequences, making it an ideal choice for tasks that require understanding or generating sequences in a causal or chronological order.  As representative language models of this architecture, the GPT-series models are developed based on the causal-decoder architecture. 

**Case Studies:** Take text generation with the GPT Series as an example.

1. Architecture: The GPT models use a causal-decoder structure where each token in the sequence can only attend to previous tokens and itself. This is achieved through a unidirectional attention mask.

2. Process: When generating text, the model receives a prompt (e.g., "The sun sets over the"). It processes each token in the prompt, predicting the next word based on the previous context. For instance, after "the", it might predict "horizon".

3. Application: This architecture is ideal for tasks like story generation, where the narrative must follow a logical and chronological sequence.

4. Outcome: GPT models, thanks to the causal-decoder architecture, can generate coherent and contextually relevant text that follows a logical temporal order, making them powerful tools for various language generation tasks.

**textbf{Related Work:** More information can be seen in Brown et al. [2020] \cite{brown2020language}.

### Prefix Decoder Architecture

**Definition:** The Prefix Decoder Architecture is a specific design used in language model architectures \cite{dong2019unified}. It is also known as the non-causal decoder, which revises the masking mechanism of causal decoders. This enables it to perform bidirectional attention over the prefix tokens while only using unidirectional attention on generated tokens. This allows prefix decoders to bidirectionally encode the prefix sequence and autoregressively predict the output tokens individually. During encoding and decoding, the same parameters are shared. This makes the prefix decoder similar to the encoder-decoder architecture.

**Case Studies:** Unified Language Model Using Prefix Decoder.

1. Architecture: This design allows the model to bidirectionally process a given prefix (pre-existing sequence of tokens) while generating future tokens in an autoregressive (one-by-one) manner.

2. Process: Suppose the model is given a prefix like "In a world where technology". It first encodes this prefix bidirectionally, understanding the context from both sides. Then, for generating the subsequent text, it switches to a unidirectional approach, predicting one token at a time based on the prefix and previously generated tokens.

3. Application: This architecture is beneficial in tasks like text completion or question answering, where understanding the full context of the prefix is crucial for generating relevant and coherent continuations.

4. Outcome: Models using the Prefix Decoder architecture effectively leverage both bidirectional and unidirectional attention mechanisms, making them versatile for a range of language understanding and generation tasks. They combine aspects of encoder-decoder architectures with the autoregressive generation capabilities of causal decoders.

**Related Work:** More information can be seen in Zhang et al. [2022] \cite{zhang2022examining}.

### Alignment

**Definition:** Alignment is a crucial aspect of pre-training and involves ensuring that the model's outputs are not only accurate but also ethically aligned with human values, reducing the risks of biases and harmful outputs. Ethical alignment involves addressing issues like fairness, privacy, transparency, and accountability. It ensures that models do not perpetuate biases or stereotypes and respect user privacy and data security. Alignment tax (or safety tax) refers to the extra costs involved in aligning an AI system with human ethics and morality as opposed to building an unaligned system. These costs may arise from extensive data curation to eliminate biases, the development of complex algorithms that can discern and adhere to ethical norms, and ongoing monitoring and evaluation to ensure continued alignment. Alignment is key to building trust in AI systems, as it ensures that users can anticipate how a model might behave in different scenarios and that the model's reasoning and outputs are interpretable and justifiable. 

**Case Studies:** Implementing Ethical Standards in an AI Recruitment Tool

1. Objective: To develop an AI recruitment tool that not only assesses candidates effectively but also aligns with ethical standards, ensuring fairness and diversity.

2. Challenges and Costs:

* Data Curation: Extensive effort to curate a diverse and unbiased dataset representing a wide range of demographics.
* Algorithm Development: Creating complex algorithms capable of making fair, unbiased decisions and adhering to ethical norms.
* Ongoing Monitoring: Continuous evaluation to maintain alignment, involving regular updates and checks.

3. Performance Trade-offs:

* Processing Time: Slower decision-making due to the complexity of ethical algorithms.
* Accuracy: Potentially lower accuracy in specific contexts to avoid ethical risks, like discriminating against certain groups.

4. Outcome: The AI recruitment tool successfully facilitates fair and diverse hiring practices. Despite slower processing and potential accuracy trade-offs, the tool gains trust and acceptance due to its ethical and fair decision-making process.

**Related Works:** More information can be seen in Christiano et al. [2017] \cite{christiano2017deep} and in Askell et al. [2021] \cite{askell2021general}.

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>



## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print(s)
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
