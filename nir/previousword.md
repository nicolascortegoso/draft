2. Previous Work

Hakkani-Tür et al. [5] provide a comprehensive overview of morphological disambiguation approaches for Turkish up to 2017. These approaches range from early rule-based methods, including hand-crafted and machine-learned rules, to statistical and machine learning techniques such as hidden Markov models (HMMs), perceptron-based discriminative methods, and conditional random fields (CRFs). To mitigate data sparsity in statistical models, several strategies have been proposed, including simplifying output representations by using only the main part-of-speech [6], focusing on the last inflectional group, or disambiguating roots and affixes separately [7].

Most of the studies surveyed by Hakkani-Tür et al. predate the introduction of recurrent neural network (RNN) encoder-decoder architectures, which we explore in this work. The encoder-decoder framework was introduced independently by Sutskever et al. [8] and Cho et al. [9] for machine translation. While Sutskever et al. focused on end-to-end sequence generation, Cho et al. also applied the model to score candidate translation pairs, integrating it into a statistical machine translation system. This use of encoder-decoder models for ranking or scoring rather than generation is directly relevant to our approach.

A major enhancement to the encoder-decoder architecture came with the introduction of the attention mechanism by Bahdanau et al. [10]. Attention allows the decoder to dynamically focus on different parts of the input sequence, alleviating the fixed-length bottleneck of vanilla RNN encoders. This mechanism has since become a standard component in nearly all sequence-to-sequence models, improving both interpretability and performance across tasks.

Encoder-decoder architectures — with or without attention — have been widely adopted in morphological tasks, most often for morphological generation, such as generating inflected word forms from lemmas [11,12] or mapping between morphological feature sets and surface forms [13]. However, these models typically generate output from scratch and require full supervision.

In contrast, we use an encoder-decoder model with attention to rank a list of pre-generated morphological analyses produced by a rule-based transducer. This hybrid setup combines the linguistic precision of rule-based analyzers with the adaptive scoring capabilities of neural models. Framing morphological disambiguation as a sequence ranking task allows us to exploit existing transducers while leveraging the strengths of neural sequence modeling, particularly in morphologically rich languages.


[8] Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. arXiv:1406.1078.

[9] Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings of the International Conference on Learning Representations (ICLR).

[10] Kann, K., & Schütze, H. (2016). MED: The LMU System for the SIGMORPHON 2016 Shared Task on Morphological Reinflection. In Proceedings of the 14th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology.

[11] Cotterell, R., & Heigold, G. (2017). Cross-lingual Character-level Neural Morphological Tagging. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP).

[12] Aharoni, R., & Goldberg, Y. (2017). Morphological Inflection Generation with Hard Monotonic Attention. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL).





## PREVIOUS WORK BEFORE

Hakkani-Tür et al. [5] provide a comprehensive overview of morphological disambiguation approaches for Turkish up to 2017. These approaches range from early rule-based methods, including hand-crafted and machine-learned rules, to statistical and machine learning techniques such as hidden Markov models (HMMs), perceptron-based discriminative methods, and conditional random fields (CRFs). To mitigate data sparsity in statistical models, several strategies have been proposed, including simplifying output representations by using only the main part-of-speech [6], focusing on the last inflectional group, or disambiguating roots and affixes separately [7].

Most of the studies surveyed by Hakkani-Tür et al. predate the use of recurrent neural network (RNN) encoder-decoder architectures, which we explore in this work. Introduced by Cho et al. [8] for machine translation and extended with attention mechanisms by Bahdanau et al. [9], the encoder-decoder framework has proven effective for a range of sequence modeling tasks. In the context of morphology, encoder-decoder models have been widely used for generating inflected forms from lemmas [10,11] or for mapping between morphological representations [12].

In contrast to generation-based approaches, our work employs an encoder-decoder model to rank a set of candidate morphological analyses produced by a rule-based morphological transducer. This framing treats disambiguation as a sequence ranking task, allowing us to leverage the high coverage and precision of hand-crafted rule-based analyzers while using neural models to resolve ambiguity. This hybrid approach is particularly well suited to morphologically rich languages, where purely statistical generation may struggle to capture rare or linguistically conditioned forms.

More recent work on morphological analysis has begun to explore the use of neural architectures, particularly sequence-to-sequence models based on recurrent neural networks (RNNs). The encoder-decoder framework, first introduced by Cho et al. [8] for statistical machine translation, and later extended with attention mechanisms by Bahdanau et al. [9], has shown strong results in sequence generation tasks, including morphological inflection [10,11] and full-form generation from lemmas [12]. These models typically aim to generate the correct morphological analysis or surface form from scratch, either in a low-resource or high-resource setting.

In contrast, our approach leverages the encoder-decoder architecture not to generate analyses from scratch, but to rank a pre-generated list of candidate morphological analyses produced by a rule-based morphological transducer. This setup is particularly useful in languages with complex morphology, where high-precision rule-based systems can generate a rich set of plausible analyses, but selecting the correct one requires robust disambiguation. By treating the problem as a sequence ranking task, we take advantage of the powerful representational capacity of neural models while retaining the linguistic coverage of hand-crafted transducers.




The sequence-to-sequence (seq2seq) paradigm, introduced by Sutskever et al. [8], transformed structured prediction tasks in natural language processing.
Their work demonstrated that encoder-decoder architectures could effectively map variable-length input sequences to output sequences, initially for machine translation. Bahdanau et al. [9] further advanced this approach by introducing an attention mechanism, enabling the decoder to focus on relevant parts of the input sequence during output generation.
Additionally, Cho et al. [10] proposed an RNN encoder-decoder model to learn phrase representations for statistical machine translation, demonstrating the effectiveness of such architectures in capturing contextual dependencies for scoring and ranking tasks. While these seq2seq models were designed for generative tasks, our approach adapts the encoder-decoder framework to rank pre-generated morphological analyses produced by a rule-based morphological transducer, rather than generating analyses from scratch.
