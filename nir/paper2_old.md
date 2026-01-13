# Morphological Disambiguation for the Yakut Language

For interpretatibility: 007_02_0016 the imperative with beje


## Abstract

Morphological disambiguation is the task of selecting the correct morphological analysis for words in context from among multiple possible interpretations. This paper presents an encoder-decoder approach to morphological disambiguation for Yakut (Sakha), a Turkic language characterized by complex agglutinative morphology. While morphological disambiguators typically select the most plausible analysis from a morphological analyzer's output, the rich inflectional system of Yakut presents particular challenges due to data sparsity. Our work specifically addresses these challenges by developing an effective disambiguation model trained on limited annotated resources, demonstrating that competitive performance can be achieved despite the constrained training data.


### Keywords
morphological disambiguation, discriminative model

## 1. Introduction

Morphological analysis has long been a cornerstone of NLP, providing linguistically grounded decompositions of words into stems and affixes based on grammatical rules. Modern large language models (LLMs), in contrast, rely on statistical subword tokenization (e.g., Byte-Pair Encoding) that approximates morphemes through frequency patterns rather than linguistic principles. While effective for many tasks, this statistical approach lacks explicit representation of morphological structure.

Traditionally, rule-based morphological transducers have been used to formalize linguistic knowledge through finite-state networks that explicitly encode [3]:
- Morpheme Inventories: lexical roots and affixes (are cataloged as lexemes with grammatical properties).
- Morphotactics: Valid morpheme sequences are constrained by finite-state rules.
- Phonological Rules: Alternations (e.g., vowel harmony in Turkic languages) are implemented as context-dependent rewrite rules (Koskenniemi, 1983).

In analysis mode, a morphological transducer breaks down surface word forms into their morphemic components and assigning grammatical interpretations. 

This mapping between surface and analysis forms can be effectively learn by a neural network using supervised learning on morphological annotated data [REFERENCE].



The bootleneck is the lack of morphological annotated data for the Yakut language.


Map a word form to its corresponding linguistic representation.
For agglutinative languages, this is identifying the lexical root and a series of concatenated affixes.

challenging to learn this representations from morphological annotated data.

Statistical models for agglutinative languages have to deal with the problem of data spacity caused by the complex morphology and the resulting number of distinct word forms.

The productive derivational morphology also results in larger tag sets, which in turn, demands larger corpora to be observed in.

Agglutinative languages, such as Yakut, exhibits a rich system of derivational suffixes, allowing a single root to generate numerous lexical variations. This morphological productivity introduces challenges in applying supervised machine learning methods, as individual lexical items may have sparse observations due to the vast number of possible derived forms.



However, there is often more that one possible interpretation and the correct choice depends on syntactic and semantic context.

 Rule-based transducers exhaustively generate such analyses but fail to rank them contextually.


### 1.2 Disambiguation


Morphological disambiguation is the task of selecting the sequence of morphological parses corresponding to a sequence of words, from the set of possible parses for those words [3].

Morphological disambiguation is typically harder for agglutinative languages for the following reasons:

- High Morpheme Density: A single word can encode multiple grammatical features, requiring disambiguation of overlapping or nested morpheme boundaries.

- Ambiguity in Segmentation: Morphemes may have homophonous forms (e.g., a suffix might mark a different morphological feature depending on context).

- Sparse Data: Rare morpheme combinations may not appear in training data, complicating statistical disambiguation.

Machine learning models (e.g., neural networks, CRFs) as they rely on frequency statistics.

The training corpus does not contain enough examples of certain word forms or morpheme combinations to reliably estimate their probabilities.


## 2. Previous Work

In prior work [PUBLICATION], we introduced a morphological pipeline for Yakut, decomposing words into their constituent roots and affixes. However, morphological analyzers often produce multiple valid interpretations for a given word form, necessitating disambiguation to select the most contextually appropriate analysis.

This paper addresses the task of morphological disambiguation for Yakut by resolving ambiguities in the output of a morphological transducer. We assume that the transducer generates correct but potentially ambiguous analyses, and our goal is to select the most probable sequence of morphological interpretations within a given sentence context.


We propose to handle this by breaking up the morhosyntactic tags into inflectional
groups, each of which contains the inflectional features for each (intermediate) derived form. 
the probability of each morhosyntactic tag by considering statistics over the individual inflectional groups and surface roots in trigram models.
Turkish presents an interesting problem for statistical models since the potential
tag set size (i.e., the number of possible morphological parses) is very large because
of the productive derivational morphology. Our approach handles this by breaking
up the morphosyntactic tags into inflectional groups, each of which contains the
inflectional features for each (intermediate) derived form. We developed and tested
four statistical models which score the probability of each morhosyntactic tag by
considering statistics over the individual inflectional groups and surface roots in
trigram models [1]



No work on Yakut morphology.

Hakkani-Tür [REFERENCE] present an overview of the morphological disambiguation problem for Turkish and the efforts that have been taken in the last three decades to address this topic. The taken apporaches span from ruled-based, to statistical and machine learning methods.

Rule-based approaches typically involve writing a large number of hand-crafted linguistic constraints to eliminate impossible tags or morphological parses for a given word in a given context. Since hand-crafting disambiguation rules is a very difficult task that requires deep knowledge of the language in question, we have focused only in statistical and machine learning alternatives.



Back in 2002 Hakkani-Tür [2002] proposed a way to disambiguate the analyses.
In order to aleviate the tag spasity problem, Hakkani-Tür [2002] followed a linguistically motivated approach to employ inflectional groups as basic units in modeling of tag sequences.

Their approach essentially involves breaking up the full morphological analysis across derivational boundaries and treating the components as subtags, and then determining the correct sequence of tags via statistical techniques.

The constraint for adopting supervised machine learning approaches is the lack of morphologically annotated sentences for Yakut language publicly available.

This limitation was partially addressed in [[SELF-REFERENCE]](#references)


## 3. Our Approach


We tackle the problem of selecting the correct morphological analysis of a word form in context by training a neural network. We frame this as a sequence-to-sequence task using an encoder-decoder architecture:
- The encoder processes the input sequence, which contains the target word form, and produces a contextual representation of the sentence.
- The decoder then generates the morphological analysis step by step, using information from the encoder.

To improve performance, we implement the Bahdanau attention mechanism, which allows the decoder to dynamically focus on different parts of the input sequence at each time step.
This helps overcome the bottleneck problem of traditional sequence-to-sequence models, which rely solely on the encoder’s final hidden state.

Once trained, rather than generating recursively the next token given the context, the neural network is used to rank the outputs produced by a rule-based morphological transducer, enabling the selection of the most probable analysis.

## 4. Tokenization

We use SentencePiece as the subword tokenization algorithm for both the encoder and decoder. However, instead of allowing the algorithm to learn subword units entirely from the training data, we partially hardcode specific units corresponding to syllables and affix tags.
This decision aims to achive better interpretability and generalization.

### 4.1 Interpretability

Syllables consist of a vowel nucleus (e.g., [a]), which may be preceded by consonants (onset, e.g., [t] in ta) and/or followed by consonants (coda, e.g., [p] in at).

Compared to subword units, which can vary greatly in length, syllables are more homogeneous segments. By aligning chunks of similar lengths, we achieve a clearer visual representation when plotting the attention mechanism. Since most affixes are monosyllabic [SOURCE], this benefits the mapping. Additionally, Yakut word forms have a relatively simple syllabic structure, allowing for unambiguous segmentation.

### 4.2 Generalization

A fixed syllabic structure (rather than learned units) may generalize better across different data distributions.

While we could approximate syllables using SentencePiece by training on a sufficiently large corpus and tuning vocabulary sizes, this introduces two additional parameters (vocabulary size and corpus size) that require experimentation.
By hardcoding syllables in both the encoder and decoder inputs, we gain more control over the tokenization process.

### 4.3 Implementation Details

For the encoder, we use a orthografical transducer to extract the syllables from the Leipzig corpora [10].
The transducer segmentates well-formed Yakut word forms according to the principles outlined in [SOURCE].
To compatibilize it with SentencePiece algorithm, syllables are differentiated as initial (preceded by the symbol _) and following (no special marks).
The set of initial and following syllables defines the minimum vocabulary size.
These syllables are hardcoded into SentencePiece, which is then run on the research corpus with a vocabulary size equal to the number of extracted syllables + 1,000 (to accommodate symbols and word forms that deviate from Yakut orthography, like loan words).

For the decoder, since syllables here correspond only to lexical roots, the orthographical transducer is applied to a list of such roots.
We again make the distinction between initial and following syllables.
The set of extracted syllables, along with affix tags, are hardcoded into SentencePiece, defining the decoder’s minimum vocabulary size.
The decoder is also trained on the research corpus to learn representations not present in Yakut lexical roots.





JUNK

If the specified vocabulary size is small, it is expected that SentencePiece is going to approximate them.
However, since the Yakut language have a syllable structure, instead of trusting the algorithm to get the syllables, we hardcode them.

The syllables that can be observed in a Yakut word form can be reconstructed from this table [[Ubryatova]](#references)





## 5. Types of Morphological Ambiguity

Morphological ambiguity stems from diverse sources. The author distinguishes between part-of-speech ambiguity, segmentation ambiguity, morphographemic processes, and homograph morphemes. All these types are observed in the Yakut language, each posing different degrees of difficulty to solve.

For example, part-of-speech ambiguity can have its origin on the lexicon, when a lexical root is associated with more than one primary stem. For instance, the root *санаа* can both be interpreted as a verbal stem (the second person imperative from of “to think”) or as a nominal stem (the singular nominative case of “thought”). Affixation often helps to solve this type of lexical ambiguity, since stem types introduce restrictions on the affix than can be attached to.

Furthermore lexicalized derivations inject additional ambiguities when the lexicalized word has gained a totally different meaning and life of its own [TUR].

Some monosyllabic adjectives coincide with verbs both in form and in content. There are bases that are identical in sound composition, but completely different in meaning. They are grammatical homonyms [§255, 161].

Segmentation ambiguity is very pervasive, since word forms often can be segmented in multiple different ways. This phenomenon is widespread in Yakut for disyllabic verbal roots that have a “fosilized” productive affix in the last syllable.

Read §500 about dead roots in reflexive verbs.

For example, the form *кэбис*, can be analysed as a lexical root with the sense of (to throw, toss, pitch, cast) or as *кэп* (push away, shove someone/something away) + the reciprocal affix *ис*.

Similar in meaning.

Segmentation ambiguity also occurs with the combination of some affixes. For example, it is oberved sistematically among the possessive cases in relation to the plural. For instance, the form доруобуйаларыгар can be segmented both as:

- доруобуйа+N+POSS.3PL+DAT
- доруобуйа+N+PL+POSS.3SG+DAT

The root *доруобуйа* (health) can be interpreted as singular or plural.


In other cases, ambiguity can introduced by morphographemic processes. For example, the verbal roots *көп* (to rise, to float to the surface) and *көт* (to fly) have the same past participle form көппүт. By regressive assimilation, an affix starting with a labial consonant like, the past participle-быт, forces the final dental voiceless stop (т) in the stem көт to changes to a bilabial voiced stop (п). Because көт and көп are verbs, көппүт can only be disambiguated semantically.

Morphographemic processes render different interpretations of a surface form. For example, барыах is analyzed as a verbal root *бар* ( to go, leave, depart) with the future participle -ыах:
бар+V+PART_FUT
However, since affixes starting with vowels or diphthongs cause the deletion of long consonants in stem endings, the lexical root can also be interpreted as the verbal root *бараа* (to spend):

бараа+V+PART_FUT

Yakut has homograph morphemes. For example, the predicate and possessive affixes in Yakut have the same form for the first and second person of the plural (*-быт, -гыт* and their allomorphs) [Ubryatova, 311]. Another example is observed for *-нан* and its allomorphs, that represents the instrument both in it plain and in it possessive variants. When homograph  affixes have similar grammatical function, syntax is not sufficient to disambiguate them.




## 6. Experiment 


This leads to:
- Better handling of long sequences – The model does not have to encode all information into a single fixed-length vector.
- Improved alignment between input and output tokens – Useful for translation and other sequence tasks where words correspond non-linearly.
- Enhanced interpretability – The attention weights indicate which input tokens influence each output step [Experiment].


We experiment with an LSTM and with a GRU.


## 7. Conclusions and Further Work

The disambiguation algorithm is ineficient.
Slow to process data.
However, the it can be use to automatically produce a lot of high-quality annotated data, that can be used to train an end-to-end neural model.




JUNK

Lack of annotated data.

Difficult for testing.

However, lexical roots would necessitate for bigger corpora, even when applying subword units.
Assets the effect on disambiguation of both models, being the inflectional group the one with the biggest contribution.



More importantly, as the sentences are examples of the topic that is being discused, the explanations are very helpfull to discard non-valid parses in the given context. Another advantage of using this source, is that the examples are extracted from real texts.
To avoid reserving a part of the training data for model validation, we resource to an external validation set. For this matter we employ as source the sentences from the Samoutchitel for Yakut language. This is a teach-yourself book for Yakut language by Kharatinov, that contains 80 lessons that cover a part of Yakut grammar. Each 10th lesson is a review of the previous ten, so we extracted and annotated the example sentences from lessons: 10, 20, 30, 40, 50, 60, 70 and 80. The annotation process is identical to the one used for the training sample: we run the transducer over the sentences and then we proceed to manually disambiguate the analyses.
The sentences of this dataset are not sourced on real data, but constructed for pedagogical purposes. With the progression of the lessons the sentences increase in complexity. This property could be useful to assess the performace of the dismbiguation model with different types of constructions.




## References

[1] Koskenniemi
[2] Ubryatova 
[3] Transducers
[4] Hakkani-Tür D., Oflazer K., Tür G. Statistical Morphological Disambiguation for Agglutinative Languages, 2002.

[10] Leipzig corpora.





# JUNK

Since Yakut language has no prefixes, the Yakut word form can be described as a lexical root, followed by zero or more suffixes.
Morphological analysis can be used for:
identifying different forms of words that share the same root;
aggregating multiple word forms into a single lemma;
extracting morphological features.


The Hakkani-Tür distinguish two main challenges for morphological disambiguation in Turkish are the following:
1) The number of possible distinct word forms and hence the complex morphological parses to capture all possible morphological distinctions is very large; in the tens of thousands.
2) The number of possible morphological interpretations per word, especially when very high-frequency words are excluded, is relatively large These are typically caused by root part-of-speech ambiguity, segmentation ambiguity, morphographemic processes, homograph morphemes, but can occasionally be caused by the original source of a (borrowed) word. 



In agglutinative languages words are formed by linearly concatenating morphemes to a root, often creating long, multi-morpheme words with nuanced meanings. 

Each morpheme carries grammatical information (case, tense, possession, etc.), leading to exponentially more possible word forms compared to isolating (e.g., Mandarin) or fusional languages (e.g., Spanish).



While morphological analysis was essential in classical NLP pipelines to handle word-form variability and sparsity, modern LLMs implicitly capture these regularities through subword tokenization and pretraining at scale. However, explicit morphology remains relevant for low-resource languages, model interpretability, and certain generation tasks.

Agglutinative languages, such as Yakut, exhibits a rich system of derivational suffixes, allowing a single root to generate numerous lexical variations. This morphological productivity introduces challenges in applying statistical approaches, as individual lexical items may have sparse observations due to the vast number of possible derived forms.


The previous examples of segmentation ambiguity is natural to the language, however, there is a type of segmentation ambiguity that is caused by including in the lexicon word forms that can be further analysed. Ideally, the lexicon should contain only lexical roots without any productive affixes. As a general rule, if a word form can be derived from another, the derived form should not be included in the lexicon. However, this distinction is not always clear-cut and it is related to semantics and the different senses of a word form.


Morphological analysis is a fundamental component of natural language processing (NLP), enabling systems to capture the structural regularities of language, handle diverse word forms, and mitigate data sparsity in linguistic datasets. Yakut (Sakha), a Turkic language with highly agglutinative morphology, exhibits a rich system of derivational suffixes, allowing a single root to generate numerous lexical variations. While this morphological productivity is linguistically fascinating, it also introduces challenges in data acquisition, as individual lexical items may have sparse observations due to the vast number of possible derived forms.



## 4. Tokenization

We use SentencePiece as the subword tokenization algorithm for both the encoder and decoder.
However, instead of allowing the algorithm to learn subword units entirely from the training data, we partially hardcode specific units corresponding to syllables and affix tags.
This decision aims to achieve better interpretability and generalization.

### 4.1 Interpretability

Syllables consist of a vowel nucleus (e.g., *[а]*), which may be preceded by consonants (onset, e.g., *[т]* in **та**) and/or followed by consonants (coda, e.g., *[т]* in **ат**).

Compared to subword units, which can vary greatly in length, syllables are more homogeneous segments.
This more consistent length facilitates the alignment of linguistic units with the attention weights, leading to more interpretable patterns.
Additionally, Yakut word forms have a relatively simple syllabic structure, allowing for unambiguous segmentation.


### 4.2 Generalization

A fixed syllabic structure (rather than learned units) may generalize better across different data distributions.

While we could approximate syllables using SentencePiece by training on a sufficiently large corpus and tuning vocabulary sizes, this introduces two additional parameters (vocabulary size and corpus size) that require experimentation.
By hardcoding syllables in both the encoder and decoder inputs, we gain more control over the tokenization process.


### 4.3 Implementation Details

For the encoder, we use an orthographic transducer to extract the syllables from the Leipzig corpora [10].
The transducer segments well-formed Yakut word forms according to the principles outlined in [SOURCE].
To compatibilize it with the SentencePiece algorithm, syllables are differentiated as initial (preceded by the symbol _) and following (no special marks).

The set of initial and following syllables defines the minimum vocabulary size.
These syllables are hardcoded into SentencePiece, which is then run on the research corpus with a vocabulary size equal to the number of extracted syllables + 1,000 (to accommodate symbols and word forms that deviate from Yakut orthography, like loan words).

For the decoder, since syllables here correspond only to lexical roots, the orthographic transducer is applied to a list of such roots.
We again make the distinction between initial and following syllables.
The set of extracted syllables, along with affix tags, are hardcoded into SentencePiece, defining the decoder’s minimum vocabulary size.
The decoder is also trained on the research corpus to learn representations not present in Yakut lexical roots.

