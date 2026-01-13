# A Morphologically Labeled Dataset for Yakut: Design, Compilation, and Applications

Annotated data is essential for training supervised machine learning models and for evaluation purposes. However, such data is often unavailable, requires specialized linguistic expertise to produce, and can be costly or time-consuming to obtain.
This paper introduces a dataset annotated with morphological labels for the Yakut language — a resource that, to our knowledge, did not exist prior to 2025 based on extensive literature review.
The dataset currently consists of over 12,000 sentences and multi-word expressions, drawn from various linguistic studies on Yakut morphology and standardized into a unified annotation schema.
As this is an ongoing project, the dataset is expected to grow over time.
The dataset is publicly available in a dedicated repository.

Keywords: Yakut morphology, annotated dataset, agglutinative typology, Turkic language.



## 1. Introduction

Data sparseness is a well-known challenge for agglutinative languages like Yakut due to their highly productive morphology.
Words are formed by attaching multiple suffixes to a root, resulting in an extensive array of possible word forms.
This process leads to a combinatorial explosion of inflected forms, many of which occur infrequently—or not at all—in typical datasets, creating a long tail of low-frequency tokens [[1]](#references).

Subword tokenization algorithms, such as those proposed in [[2]](#references), [[3]](#references), and [[4]](#references), have helped mitigate the data sparsity problem by segmenting less frequent word forms into smaller, more reusable chunks.
Notably, subword tokenization algorithms have shown the ability to approximate morphemes [[5]](#references).
Subword tokenization has proven effective in reducing data sparsity and vocabulary size, offering a viable alternative to traditional methods like lemmatization, which have long been part of the NLP pipeline.

Despite its practical utility, subword tokenization presents two major caveats in relation to the morphological representation of language:

1) The segmentation into subwords depends on the corpus used to train the algorithm and the specified vocabulary size—two parameters that introduce significant bias.
2) The identified subwords or morphemes are not labeled, meaning they do not provide a linguistic representation of the word form, unlike the structured analyses found in specialized linguistic literature.

Morphological analysis aims to map a surface form to its corresponding linguistic representation [[6]](#references).
Neural networks, as universal function approximators, are well-suited for learning such mappings.
However, because linguistic representations are grounded in theoretical frameworks, annotated data is essential for training these models in a supervised fashion.

Rule-based morphological transducers offer an alternative approach for mapping surface forms to their analyses.
Since the rules are explicitly formulated, annotated data is only required for testing purposes.
This rule-based approach is a natural starting point for encoding the word-formation processes studied by Yakut language specialists, as it offers direct control over the output.

Nevertheless, this method has several drawbacks:

1) Formulating rules for concatenation and transformation is complex and time-consuming.
Additionally, certain aspects of Yakut morphology may be insufficiently studied, making them difficult to encode.
2) The mapping between surface and analysis forms is not always one-to-one; a given word form may have multiple valid interpretations, requiring contextual disambiguation.

However, rule-based and neural network approaches are not mutually exclusive.
In low-resource scenarios, a rule-based system can generate initial annotations.
As more labeled data becomes available, a transition to a fully neural approach becomes viable.
Hybrid systems can also serve as intermediate solutions—for example, using a rule-based transducer to generate morphological analyses and a neural network to disambiguate them.

In all cases, the compilation of morphologically annotated data remains a crucial task for both training and evaluation.
Fortunately, existing work on Yakut morphology provides valuable source material for constructing such a dataset.
These studies often present Yakut morphology using example sentences with glosses, explanations, and topic-based organization.
This structure not only facilitates annotation by non-native speakers but also provides essential linguistic context that is often absent in automatically compiled datasets.
Such contextual information is particularly important for ensuring the grammatical representativeness of the corpus [[7]](#references), as the annotated sentences illustrate specific morphological features in use.

The present work involves compiling sentences related to various morphological topics in Yakut from specialized sources, annotating them, and presenting them in a structured format.
Annotations are performed semi-automatically: a morphological transducer [[8]](#references) generates possible analyses for each word form, and human supervision resolves ambiguous interpretations.
Aside from removing ambiguous analyses, no further editing is performed; thus, the morphological segmentation and tags reflect the raw output of the transducer.

The remainder of this paper is organized as follows: Section 2 provides a general overview of morphological annotations for Turkic languages and the existing resources for Yakut.
Section 3 outlines the main features of Yakut morphology and the principles followed when labeling lexical roots and affixes.
Section 4 describes the contents of the annotated dataset, while Section 5 discusses potential sources of error.
Finally, Section 6 presents the conclusions.


## 2. Previous Work

While descriptive linguistic studies on Yakut exist (e.g., [[9]](#references)), computational resources such as annotated corpora and morphological analyzers remain extremely limited.

To the best of our knowledge, no morphologically annotated dataset exists for the Yakut language, apart from a treebank created by Merzhevich T. and Ferraz Gerardi F. [[10]](#references), which contains 301 annotated sentences within the Universal Dependencies (UD) framework.

The UD project provides a standardized framework for grammatical annotation, covering both morphological features and syntactic dependencies across a wide range of languages [[11]](#references).
It adopts a lexicalist approach to syntax, where dependency relations are defined between syntactic words rather than phonological or orthographic units [[idem]](#references).
However, UD guidelines do not segment words into morphemes; instead, morphological features are encoded as properties of entire words.
This poses challenges for representing agglutinative languages, which frequently express complex meanings within a single word form—meanings that would require multiple words in less synthetic languages [[12:1]](#references).

Due to the scarcity of research and resources on the Yakut language in the field of computational morphology, this study draws primarily on methods and findings developed for Turkish, the most extensively studied Turkic language in NLP [[13]](#references). 

Efforts have been made to achieve greater uniformity in the study of Turkic languages and the typology of agglutinative languages [[14]](#references).
One such effort is the Turkic Morpheme Portal, a toolkit that provides linguistic databases describing morphological, syntactic, and semantic features of Turkic languages. This resource is built upon a structural-parametric morpheme model [[15]](#references).

As mentioned in the introduction, the compilation and annotation of Yakut sentences draw on several foundational works on Yakut morphology, which are discussed in detail in Section 4.


## 3. Yakut Morphology

Agglutinative languages like Yakut primarily form words by attaching a sequence of affixes to a lexical root.
The overall interpretation of a word arises from the combined contributions of these affixes, each of which modifies or extends the meaning of the root.
For instance, the word form *суруйтарбахтаамаайабын* [[16:48]](#references) can be segmented into a root and five affixes:

- *суруй* (to write) — lexical root
- *-тар* (causative)
- *-бахтаа* (diminutive)
- *-м* (negative)
- *-аайа* (potential)
- *-бын* (first person singular)

Throughout the process of affixation, the lexical root retains its core meaning, while each affix contributes a distinct grammatical or semantic nuance.
The stem refers to the base to which affixes are attached; it may consist of the root alone or the root plus one or more affixes.
In Yakut, affixes are always concatenated to the right of the stem.

Yakut word forms can be broadly classified into three morphological types: nominal, verbal, and immutable [[9:37]](#references):
- Nominal: undergo declension.
- Verbal: undergo conjugation.
- Immutable: normally do not undergo declension or conjugation.

As noted by Ubryatova et al., while this classification is approximate, it is generally consistent and stable [[idem]](#references).

The nominal category includes nouns, adjectives, numerals, and pronouns. These forms can be inflected to express grammatical categories such as case, number, and possession.

The verbal category encompasses verbs, participles, and gerunds. These are inflected for tense, aspect, mood, person, and other verbal features.

Immutable words include adverbs, interjections, onomatopoeias, and function words such as particles and modals.
These forms are not subject to inflection and typically remain unchanged across grammatical contexts.

Derivational affixes can change the morphological type of a stem. For example, the affix *-лаа* and its allomorphs are used to derive verbal forms from a variety of stem types [[9]](#references).

A word form may also serve different syntactic functions. In some cases, this shift is marked morphologically—for instance, pluralized adjective stems may function as nouns. However, functional shifts can also occur without any morphological change to the stem.

Although Yakut is primarily a synthetic language, certain word formation mechanisms are analytical.
These analytical processes are not annotated in the dataset, as the morphological transducer used for annotation operates at the level of orthographic words and does not analyze multi-word constructions.

The following three subsections describe the identification and labeling of lexical roots and affixes as implemented in the morphological transducer. This means that any modification related to the analysis of a word form is not directly applied in the dataset, but first on the morphological transducer that produces the annotations.


### 3.1 Lexical Roots

The lexical root is the most basic, indivisible part of a word that carries its core lexical meaning.
Lexical roots must satisfy two general conditions:
1) They must function as stems without any productive affixation.
2) They must be stand-alone words currently used in the modern language.

The first condition implies that the lexical root cannot be broken down into smaller meaningful units.
The second means that the root must appear as an entry in dictionaries, rather than being merely reconstructed through historical analysis.

Lexical roots are categorized into one of three major aforementioned types: nominal, verbal, or immutable. This classification is often problematic for several reasons.

One such issue is the distinction between nouns and adjectives within nominal roots.
For instance, the root *баай* may be interpreted as either "rich" (adjective) or "wealth" (noun).
In dictionaries such as Slepsov’s [[17]](#references), these interpretations are typically grouped under a single entry.
Whether *баай* behaves more like a noun or an adjective depends on the context.

The morphological transducer used to annotate the dataset classifies such nominal roots as either nouns or adjectives to avoid duplication and reduce ambiguity in the output.

By default, noun-adjective roots are categorized as nouns, unless explicitly classified as adjectives in authoritative Yakut grammars.
For example, the root *баай* is classified as a monosyllabic adjective in [[18:59]](#references), and is therefore labeled as such in the dataset, regardless of its syntactic function in a sentence.
This principle is not without issues, however, since sources may differ in how they classify a given entry.

A distinct case involves noun-adjective homonyms.
For example, *саҥа* can mean both "voice" (noun) [[17:235]](#references) and "new" (adjective) [[17:237]](#references).
Since these are clearly distinct dictionary entries, both forms are included in the transducer’s lexicon.
Because both variants often accept the same affixes, the transducer’s output requires contextual disambiguation.

Some roots may belong to more than one category.
For instance, *санаа* means both "thought" (noun) and "to think" (verb).
Despite their semantic closeness, such roots are listed separately in the transducer’s lexicon, as they follow different morphological rules.
Disambiguation might occur naturally by the affixes they take.

In the case of immutable words, researchers often distinguish between those of nominal and verbal origin.

Some immutable words are no longer used in their original forms and have become lexicalized, forming independent categories.
These include proper adverbs [[19:12-15]](#references), particles [[16:69-184]](#references), modal words [[20:69-139]](#references), and conjunctions [[19:80]](#references).
As such, they are considered lexical roots in their own right and are accompanied with their respective type.

The sources cited above provide many examples of such lexicalized forms, though the lists are necessarily incomplete due to the ongoing and diachronic nature of lexicalization.
As Petrov notes, many word forms exhibit different stages of lexicalization [[16:94]](#references).

To classify lexical roots, Ubryatova et al. [[9]](#references) rely on morphological, semantic, and syntactic criteria.
Following this guideline, we first observe distinctive morphological features—such as the derivations a root can take, the presence of fossilized affixes, and so on.
We also draw on existing works in Yakut lexicography, as mentioned before.
When these two criteria are inconclusive, syntactic features may be taken into account—for example, how the root combines with other words.


### 3.2 Affixes

An affix is the simplest morphological element (morpheme) added to a stem to modify or extend its meaning.
It cannot be used independently of the base word in the language [[9:34]](#references).

Affixes are typically categorized as derivational or inflectional, based on their function.
However, as noted by Ubryatova et al., "complete unanimity has not yet been achieved in the linguistic literature on Turkic languages regarding the boundary between word formation and word inflection" [[9:35]](#references).

Affixes identified solely through historical or diachronic analysis—those which no longer function productively—are considered fossilized, and are thus regarded as part of the lexical root.
For an affix to be considered productive, it must actively contribute to the formation of new word forms in the modern language.

Nonetheless, in Yakut, affixes—especially those involved in word formation—exhibit varying degrees of productivity.

In this work, we restrict the morphological transducer to identifying only those affixes described by Ubryatova et al. that are relevant to the productive system of the modern language [[9]](#references).



### 3.3 Morphological Labels

A morphological label represents the mapping of an allomorph to an abstract linguistic representation.
For example, the affix conveying plurality in Yakut has sixteen allomorphs:

*{-лар, -лэр, -лор, -лөр, -тар, -тэр, -тор, -төр, -дар, -дэр, -дор, -дөр, -нар, -нэр, -нор, -нөр}*

All of these are mapped to a single tag, such as *+PL*., representing the plural feature.

Morphological analyses are presented as sequences of tags denoting marked morphological features, appended to the lexical root in the format:

root+tag₁+tag₂+...+tagₙ

Morphological features corresponding to zero-affixation (e.g., the singular number in nouns) are often also assigned tags.
However, to maintain economy in representation, the morphological transducer used for the annotation omits such unmarked features. These can be restored later if required.

To represent the internal structure of word forms, specialized literature [12]](#references), [[21]](#references), [[22]](#references) often proposes splitting orthographic words into inflectional groups (IGs).

In this approach, roots and derivational elements are treated as separate IGs, delimited by derivational boundaries (DB), using the format:

root+IG₁ˆDB+IG₂…+ˆDB+IGₙ

Here, **root** represents the lexical root, **IGᵢ** represents inflectional morphemes, and **ˆDB** marks a derivational boundary.

For instance, Eryiğit et al. [[22:139]](#references) segment the Turkish word *arabanızdaydı* ("it was in your car") as:

*arabanızdaˆDB ydı*

Each IG is annotated with its own part of speech and inflectional features:

*araba+Noun+A3sg+P2pl+LocˆDB+Verb+Zero+Past+A3sg*

The first IG, *arabanızda* ("in your car"), is a noun marked for locative case and second person plural possession. The second IG, *ydı*, is a verb derived from the noun, marked for past tense and third person singular agreement [[ibidem]](#references).

This representation allows greater flexibility in syntactic annotation, as dependency relations can be established between individual inflectional groups.

However, as Çöltekin [[12:1]](#references) points out, the implementation of IGs remains controversial, with differing criteria for determining which affixes introduce derivational boundaries.

This observation is consistent with the view expressed by Ubryatova et al. in the previous section. Similarly, Sproat notes: "The determination of whether an affix is part of inflectional or derivational morphology is a subject of debate," and such decisions often reflect practical rather than theoretical considerations [[6:19-43]](#references).

In this study, we forgo the use of inflectional groupings to maximize flexibility in morphological representation. Should the need arise, IGs can still be generated programmatically.

The tag set used by the morphological transducer tries to follow the labels proposed in the Turkish Morpheme Portal [[15]](#references), aiming to adopt an existing standard rather than introduce a new one/

However, the tag set remains purely conventional and may be mapped to alternative notations as needed.

To distinguish affixes that serve the same grammatical function (e.g., multiple nominalization affixes), a numerical subscript (e.g., +NOM#1, +NOM#2) is added to the general tag.
This allows morphologically distinct affixes to be grouped under the same functional category while preserving their individual identities.


## 4. The Annotated Corpus

Ubryatova’s edited volume [[9]](#references), comprising 495 pages, remains the most comprehensive resource on Yakut morphology. A second volume [[23]](#references), focused on syntax, was not used as a source in compiling the dataset.

The first part of the volume addresses linguistic theory (pp. 1–36) and Yakut phonetics (pp. 39–107), while the remainder is dedicated to the description of Yakut morphology, structured across the following sections:

- Nouns (pp. 109–155)
- Adjectives (pp. 156–173)
- Numerals (pp. 174–186)
- Pronouns (pp. 187–212)
- Verbs (pp. 213–226)
- Participles (pp. 227–239)
- Gerunds (pp. 240–252)
- Verbal voice (pp. 253–273)
- Verbal aspect (pp. 274–304)
- Verbal moods (pp. 305–342)
- Adverbs (pp. 343–366)
- Onomatopoeias (pp. 366–371)
- Figurative verbs (pp. 375–383)
- Interjections (pp. 384–396)
- Auxiliary verbs (pp. 401–406)
- Auxiliary nouns (pp. 407–414)
- Postpositions (pp. 415–430)
- Particles (pp. 431–448)
- Modal words (pp. 449–466)
- Conjunctions (pp. 467–477)

This volume serves as the principal source for the compiled dataset, both in terms of the number of extracted sentences and the breadth of grammatical categories it covers.

The editorial team of the volume included Korkina E.I., Kharitonov L.N., and Petrov N.E., each of whom contributed individual studies focusing on specific aspects of Yakut morphology: Korkina on verbal moods and conjugation [[24]](#references), and participles [[25]](#references); Kharitonov on verbal voice [[26]](#references) and uninflected word classes [[19]](#references); and Petrov on particles [[16]](#references) and modal words  [[20]](#references). These individual contributions, together with the main reference work [[9]](#references), form the core foundation of the annotated dataset.


The oldest material, dating back to 1947, comes from Kharitonov's work [[26]](#references).Despite its age, it remains consistent with the current orthographic conventions.

At its current stage of development, the dataset contains over 10,000 annotated sequences.


## 5. Incorrect  Annotations

The semi-automatic two-step annotation process introduces the possibility of errors during both the automatic and manual stages:

1) Automatic errors: The morphological transducer fails to generate the correct analysis for a given word form, leaving no accurate interpretation to choose from.
2) Manual errors: Although the correct analysis is generated, an incorrect interpretation is selected during the disambiguation phase due to contextual misjudgment.

In some instances, multiple interpretations may be equally plausible. For example, the instrumental case and the instrumental possessive case share the same allomorph [[9-146]](#references), making it difficult in certain contexts to determine whether a word form conveys possessiveness.

Below are common types of mislabeling that may occur in the annotated corpus.

### 5.1 Misclassification of the Lexical Root

The lexical root may be correctly identified but assigned to an incorrect lexical category. This misclassification is best understood in terms of the core meaning of the root rather than its syntactic role within the sentence. For example:
- Nouns may function as determiners or adverbs.
- Adjectives and numerals can become substantivized (i.e., used as nouns).

Such ambiguities are particularly common in nominal categories, where it is often difficult to determine whether a lexical root corresponds to a noun, adjective, or adverb, as mentioned in section 3.1.

### 5.2 Segmentation Issues

Segmentation errors in the corpus typically fall into two categories: under-segmentation and over-segmentation.

Under-segmentation occurs when a lexical root is identified but not fully analyzed, often due to the absence of the expected root in the transducer’s lexicon.

Over-segmentation is the opposite: the analysis extends too far, segmenting a stem down to a root that no longer conveys a recoverable meaning or is no longer attested in the modern language.

These errors are particularly common in verbal stems, which often include fossilized voice affixes [[26:74]](#references). This issue is closely related to the phenomenon of lexicalization. For instance, Ubryatova et al. [[9:364]](#references) note that participial forms such as *соруйан* ("on purpose", "deliberately") and *дьоҕойон* ("in vain") have developed into adverbs. The former has lost its semantic connection with the base *соруй*, while the latter derives from *дьоҕой*, a root no longer found in the contemporary language.

In many such cases, interpretation is open to debate. Morphologically complex forms may become opaque as speakers cease to perceive their internal structure as compositional.

When uncertainty arises, the annotation process tends to favor over-segmentation.


## 6. Conclusions and Further Work

We have presented a collection of sentences with morphological annotations for the Yakut language, using the orthographic word as the unit of analysis.

Each Yakut word form is represented by its lexical root and corresponding type, followed by a sequence of labels denoting the concatenated affixes.

Importantly, the dataset is compiled exclusively from authoritative works on Yakut morphology—a deliberate choice that ensures the annotations are grounded in linguistic interpretations provided by specialists.

Although relatively small in size, the dataset is expected to offer a robust representation of Yakut morphology, as its core is derived from a comprehensive Yakut grammar.
The supplementary sources used during compilation were selected to maintain balanced coverage of the key topics addressed in the core reference.

We plan to continue expanding the dataset by incorporating additional works on Yakut morphology.
In future iterations, the annotation scheme may also be extended to include lemmas and syntactic relations between lexical units, following the model of Universal Dependencies treebanks.

Furthermore, the dataset may be used as a reference point for training and evaluating morphological transducers, supporting ongoing work in the computational modeling of Yakut morphology.

## References

[1] Oflazer K., Saraçlar M. (2018). Turkish and its Challenges for Language and Speech Processing. In: Turkish Natural Language Processing. Oflazer K., Saraçlar M. (eds.). Springer, pp. 1-19.

[2] Sennrich R., Haddow, B., Birch, A. (2016). Neural machine translation of rare words with subword units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1715–1725.

[3] Schuster M., Kaisuke N. (2012). Japanese and Korean Voice Search. 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5149–5152.

[4] Taku K., Richardson J. (2018). SentencePiece: A Simple and Language-Independent Subword Tokenizer and Detokenizer for Neural Text Processing. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66–71.

[5] Bostrom K., Durrett G. (2020). Byte Pair Encoding is Suboptimal for Language Model Pretraining. Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4617–4624.

[6] Sproat R. (1992). Morphology and Computation. ACL-MIT Press series in natural-language processing. 1992 Massachusetts Institute of Technology.

[7] Zakharov V.P. (2016). Prolegomeny k korpusnoy lingvistike. Voprosy psikholingvistiki 2016. №2, pp. 150-161.

[8] Cortegoso Vissio N., Khokhlova M.A. (2024). Three-Stage Morphological Pipeline for Word Form Analysis in Yakut. Conference: 2024 IEEE 3rd International Conference on Problems of Informatics, Electronics and Radio Engineering (PIERE), pp. 1860-1864.

[9] Grammatika sovremennogo jakutskogo literaturnogo jazyka : [v 2 tomah]. (1982). Akad. nauk SSSR, Sib. otd- nie, Jakut. fil., In-t jaz. lit.i istorii; [redkol.: E. I. Ubrjatova (otv. red.), E. I. Korkina, L. N. Haritonov, N. E. Petrov]. - Moskva : Nauka, 1982-1995. T. 2: T. 1: (Bukva B) Fonetika i morfologija / [L. N. Haritonov, N. D. D'jachkovskij, S. A. Ivanov i dr.]. - Moskva: Nauka.

[10] Merzhevich T. Ferraz Gerardi F. (2022) Introducing YakuToolkit. Yakut Treebank and Morphological Analyzer. In Proceedings of the 1st Annual Meeting of the ELRA/ISCA Special Interest Group on Under-Resourced Languages, pages 185–188, Marseille, France. European Language Resources Association.

[11] Nivre J., Zeman D., Ginter F., Tyers F. (2017). Universal Dependencies. Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Tutorial Abstracts, Valencia, Spain. Association for Computational Linguistics.

[12] Çöltekin Ç. (When) Do We Need İnflectional Groups? (2006). Proceedings of The First International Conference on Turkic Computational Linguistics, pp. 38-43.

[13] Çöltekin Ç., Doğruöz A. S., Çetinoğlu Ö. (2023). Resources for Turkish natural language processing: A critical survey. Language resources and evaluation, Т. 57, №. 1, pp. 449-488.

[14] Gatiatullin А.R, Prokopyev N., Abdurakhmonova N. (2020). Turkic Morpheme Web Portal as a Platform for Turkology Research. pp. 1-5.

[15] Gatiatullin A.R., Suleymanov D., Prokopyev N., Khakimov B. (2020). About Turkic Morpheme Portal. CEUR Workshop Proceedings, vol. 2780.

[16] Petrov N.E. (1978). Chasticy v jakutskom jazyke. Jakutsk : Kn.izd-vo.

[17] Bol'shoj tolkovyj slovar' jakutskogo jazyka = Saha tylyn byһaaryylaah ulahan tyld'yta. (2004). Akademija nauk Respubliki Saha (Jakutija), Institut gumanitarnyh issledovanij ; pod obshhej redakciej P. A. Slepcova, Novosibirsk, Nauka.

[18] Ivanov S.A. (1983). K grammaticheskoj harakteristike odnoslozhnyh prilagatel'nyh v yakutskom yazyke. Issledovaniya po grammatike yakutskogo yazyka: Sbornik Nauchnyh Trudov / Akad. Nauk SSSR, Sib. otd-nie, Yakut. fil., In-t yaz., lit. i istorii. Yakutsk : YaF SO AN SSSR.

[19] Kharitonov L.N. (1943). Neizmenjaemye slova v jakutskom jazyke : posobie dlja uchitelej / L. N. Haritonov ; Nar. komissariat prosveshhenija JaASSR. - Jakutsk : Gosudarstvennoe izdatel'stvo JaASSR.

[20] Petrov N.E. (1984). Modal'nye slova v jakutskom jazyke. N. E. Petrov ; otvetstvennyj redaktor E. I. Korkina ; Akademija nauk SSSR, Sibirskoe otdelenie, Jakutskij filial, Institut jazyka, literatury i istorii, Novosibirsk, Nauka.

[21] Oflazer K., Say B. Hakkani-Tür Z.D., Tür G. (2003). Building a Turkish Treebank. Abeillé.

[22] Eryiğit G., Nivre J., Oflazer K. (2008). Dependency Parsing of Turkish. Computational Linguistics, № 34, pp. 357-389.

[23] Grammatika sovremennogo jakutskogo literaturnogo jazyka: [v 2 tomah]. (1995). Akad. nauk SSSR, Sib. otd- nie, Jakut. fil., In-t jaz. lit.i istorii ; [redkol.: E. I. Ubrjatova (otv. red.), E. I. Korkina, L. N. Haritonov, N. E. Petrov]. - Moskva : Nauka, T. 2: Sintaksis / [E. I. Ubrjatova, N. N. Efremov, N. N. Neustroev i dr. ; redkol.: M. I. Cheremisina (otv. red.) i dr.]. - Novosibirsk : Nauka.

[24] Korkina E.I. (1970). Naklonenija glagola v jakutskom jazyke. E. I. Korkina ; Akad. nauk SSSR, Jakut. fil. Sib. otd- nija, In-t jaz., lit. i istorii ; [otv. red. prof. L. N. Haritonov]. - Moskva : Nauka.

[25] Korkina E.I. (1985). Deeprichastija v jakutskom jazyke / E. I. Korkina; otv. red. k. filol. n. N. E. Petrov; Akad. nauk SSSR, Sib. otd-nie, Jakut. fil., In-t jaz., lit. i istorii. - Novosibirsk : Nauka, Sibirskoe otdelenie.

[26] Kharitonov L.N. (1963). Zalogovye formy glagola v jakutskom jazyke / L. N. Haritonov; Akad. nauk SSSR, Jakut. fil. Sib. otd-nija, In-t jaz., lit. i istorii. — Moskva; Leningrad: Izd-vo AN SSSR.