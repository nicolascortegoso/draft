Yes, your approach is clear. Let me summarize to ensure I understand correctly:

You're developing an encoder-decoder RNN to resolve morphological ambiguity, where an external rule-based morphological analyzer generates multiple possible morphological analyses for a word, and your model disambiguates by scoring these analyses based on sentence context. Key aspects:

Architecture: An encoder-decoder RNN with Bahdanau attention. The encoder captures the sentence's contextual representation, and the decoder scores candidate morphological analyses using this context.
Input: The sentence is tokenized into subword tokens to handle data sparsity. The target word (to be disambiguated) is marked with <TARGET> and </TARGET> tags to highlight its subtokens, aiding the decoder's focus.
Decoder: Operates at the subtoken level, scoring each candidate morphological analysis for the target word.
Attention: Bahdanau attention is used to improve performance and provide interpretability by showing which parts of the input sentence the decoder focuses on when scoring analyses.
The use of subword tokens, target tags, and Bahdanau attention seems well-motivated to handle sparse data, focus the model, and ensure interpretable decisions. A few points for clarification or consideration, if you’re open to feedback:

Input Representation: How are the candidate morphological analyses fed into the decoder? Are they encoded as sequences of features (e.g., tags for tense, number) or as pre-embedded representations?
Attention Interpretability: Since you emphasize interpretability, will you analyze attention weights to explain which contextual tokens influence specific disambiguation decisions?
Training Data: Are you assuming a dataset with gold-standard morphological analyses for supervision, or will you rely on the rule-based analyzer’s outputs during training?


## Model Architecture

Our model employs an encoder-decoder recurrent neural network (RNN) with Bahdanau attention to score candidate morphological analyses for a target word in a sentence, resolving morphological ambiguity. The encoder captures the contextual representation of the input sentence, while the decoder uses this context to assign scores to predefined candidate analyses provided by an external rule-based morphological analyzer. Below, we detail the encoder, decoder, and attention mechanism.

### Encoder

The encoder is a bidirectional Long Short-Term Memory (LSTM) network that processes the input sentence, tokenized into subword tokens to address data sparsity. For a sentence containing a target word to be disambiguated, we enclose the subtokens corresponding to the target word with special tokens `<TARGET>` and `</TARGET>`. For example, the sentence "The cats run" with the target word "cats" might be tokenized as `[The, ca, <TARGET>, ts, </TARGET>, run]`. Each subword token is mapped to a dense embedding vector.

Given an input sequence of subword embeddings \( x = [x_1, x_2, \dots, x_T] \), the encoder computes forward and backward hidden states using the bidirectional LSTM:
\[
\vec{h}_t^f = \text{LSTM}_f(x_t, \vec{h}_{t-1}^f), \quad \vec{h}_t^b = \text{LSTM}_b(x_t, \vec{h}_{t+1}^b)
\]
The final hidden state for each time step \( \vec{h}_t = [\vec{h}_t^f; \vec{h}_t^b] \) concatenates the forward and backward states, capturing the contextual representation of the sentence. The `<TARGET>` and `</TARGET>` tokens guide the encoder to emphasize the target word’s context.

### Decoder

The decoder, a unidirectional LSTM, scores each candidate morphological analysis, which is encoded as a sequence of morphological feature vectors (e.g., [POS: Noun, Number: Plural, Case: Nominative]). Each feature is represented as an embedded vector, forming a sequence \( [f_1, f_2, \dots, f_M] \) for each candidate analysis. The decoder operates at the subtoken level, processing the feature sequence to produce a score for the entire analysis.

At each decoding step \( m \), the decoder takes the current feature embedding \( f_m \), the previous hidden state \( \vec{s}_{m-1} \), and a context vector derived from the encoder’s hidden states via attention (described below). The decoder updates its hidden state as:
\[
\vec{s}_m = \text{LSTM}([f_m; c_m], \vec{s}_{m-1})
\]
where \( c_m \) is the context vector. After processing the feature sequence, the final decoder hidden state is passed through a feedforward layer with a sigmoid or softmax activation to produce a score for the candidate analysis. The model is trained to maximize the score of the gold-standard analysis.

### Bahdanau Attention Mechanism

To enhance performance and interpretability, we employ the Bahdanau attention mechanism [Bahdanau et al., 2014]. This mechanism allows the decoder to focus on relevant parts of the input sentence when scoring each morphological feature. For each decoder step \( m \), attention weights \( \alpha_{t,m} \) are computed for each encoder hidden state \( \vec{h}_t \):
\[
e_{t,m} = v_a^\top \tanh(W_a \vec{h}_t + U_a \vec{s}_m), \quad \alpha_{t,m} = \frac{\exp(e_{t,m})}{\sum_{t'=1}^T \exp(e_{t',m})}
\]
where \( v_a \), \( W_a \), and \( U_a \) are learnable parameters, and \( \vec{s}_m \) is the decoder’s hidden state. The context vector \( c_m = \sum_{t=1}^T \alpha_{t,m} \vec{h}_t \) is a weighted sum of the encoder’s hidden states, capturing the relevant sentence context for the current feature.

The attention weights \( \alpha_{t,m} \) provide interpretability by indicating which subword tokens (e.g., "The" or "run") influence the scoring of specific morphological features (e.g., Number: Plural). These weights can be visualized as a heatmap to analyze the model’s decision-making process, enhancing our understanding of contextual dependencies.


Batching Strategy:
Sentence-Level Batching: Group multiple sentences into a mini-batch, where each sentence has its subword token sequence padded to the maximum length in the batch. The encoder processes these sequences to produce a batch of hidden states <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mover accent="true"><mi>h</mi><mo>⃗</mo></mover><mi>t</mi></msub><mo>∈</mo><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>T</mi><mo>×</mo><mi>H</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \vec{h}_t \in \mathbb{R}^{B \times T \times H} </annotation></semantics></math> 
h
  
t
​
 ∈R 
B×T×H
 , where <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>B</mi></mrow><annotation encoding="application/x-tex"> B </annotation></semantics></math>B is the batch size, <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi></mrow><annotation encoding="application/x-tex"> T </annotation></semantics></math>T is the maximum sequence length, and <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>H</mi></mrow><annotation encoding="application/x-tex"> H </annotation></semantics></math>H is the hidden state dimension.
Candidate-Level Batching: For each sentence, collect all candidate analyses (variable number per sentence). To batch them, pad the feature sequences to the maximum length of features across all candidates in the mini-batch. For example, if one sentence has 3 candidates with feature sequences of lengths 4, 5, and 3, pad all to length 5. The decoder processes this batched tensor of shape <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>C</mi><mo>×</mo><mi>M</mi><mo>×</mo><mi>F</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \mathbb{R}^{B \times C \times M \times F} </annotation></semantics></math>R 
B×C×M×F
 , where <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>C</mi></mrow><annotation encoding="application/x-tex"> C </annotation></semantics></math>C is the maximum number of candidates, <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi></mrow><annotation encoding="application/x-tex"> M </annotation></semantics></math>M is the maximum feature sequence length, and <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>F</mi></mrow><annotation encoding="application/x-tex"> F </annotation></semantics></math>F is the feature embedding dimension.
Masking: Use attention masks to handle padding. For the encoder, mask padded subword tokens to prevent attention to padding. For the decoder, mask padded features and candidates to ensure only valid analyses are scored.
Decoder Scoring: The decoder processes each candidate’s feature sequence independently, using the same encoder hidden局 bounds (with Bahdanau attention). For each sentence, the decoder outputs a score for each candidate analysis. To handle variable numbers of candidates, you can process all candidates for a sentence in parallel, computing attention weights and context vectors for each candidate’s feature sequence. The output is a tensor of scores <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>C</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \mathbb{R}^{B \times C} </annotation></semantics></math>R 
B×C
 , where each score reflects the likelihood of a candidate analysis being correct.
Challenges and Solutions:
Variable Number of Candidates: Since the number of candidates varies per sentence, use a maximum candidate limit (e.g., <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>C</mi><mtext>max</mtext></msub><mo>=</mo><mn>10</mn></mrow><annotation encoding="application/x-tex"> C_{\text{max}} = 10 </annotation></semantics></math>C 
max
​
 =10) and pad with dummy candidates (scored with negligible weights) for sentences with fewer candidates.
Memory Efficiency: Batching many candidates with long feature sequences can be memory-intensive. To mitigate this, use gradient accumulation or reduce the batch size if GPU memory is limited.
Implementation Tip: In PyTorch or TensorFlow, you can use nested tensors or ragged tensors to handle variable-length candidate lists, though padding with masks is often simpler and more compatible with standard RNN implementations.
Impact on Model: Batching candidate analyses does not alter the model’s architecture but requires careful handling of tensor shapes and masks in the decoder. It ensures scalability and faster training, especially for large datasets with many ambiguous words.
Should We Use Teacher Forcing?
Should we use teacher forcing for the decoder?
Teacher forcing is a training technique where the decoder is fed the ground-truth input (in your case, the gold-standard morphological feature sequence) at each step, rather than its own predictions. Whether to use teacher forcing depends on your model’s design and training goals. Here’s an analysis tailored to your scoring task:

Applicability to Your Model:
Task Nature: Your decoder does not generate a sequence from scratch but scores predefined candidate analyses, each represented as a sequence of feature vectors. The decoder processes each feature sequence to produce a single score, using the encoder’s contextual representation (via Bahdanau attention).
Teacher Forcing Relevance: In traditional sequence-to-sequence models (e.g., machine translation), teacher forcing is used to feed the correct output token at each step to stabilize training. In your case, the decoder’s “input” is the feature sequence of a candidate analysis, and the “output” is a score. Since the candidate analyses are fixed and provided by the rule-based analyzer, teacher forcing would mean feeding the gold-standard feature sequence during training to compute the decoder’s hidden states and score.
Gold-Standard Availability: You mentioned having a dataset with gold-standard morphological analyses, which makes teacher forcing feasible, as you can provide the correct feature sequence for training.
Pros of Teacher Forcing:
Faster Convergence: By feeding the correct feature sequence, the decoder learns to score the gold-standard analysis more quickly, avoiding the compounding errors that could occur if it processed incorrect feature sequences during training.
Stability: Teacher forcing reduces variance in the decoder’s hidden states early in training, leading to more stable gradients and easier optimization.
Interpretability: Since you plan to visualize attention weights, teacher forcing ensures the decoder’s attention is computed with respect to the correct feature sequence, producing more meaningful weights for analysis.
Cons of Teacher Forcing:
Exposure Bias: Teacher forcing trains the decoder with gold-standard inputs but at inference, the decoder processes candidate analyses (some incorrect). This mismatch can lead to poorer performance on incorrect candidates if the model becomes overly reliant on correct inputs. However, since your task is scoring (not generation), exposure bias is less critical, as the decoder always sees complete candidate sequences at inference.
Limited Exploration: Teacher forcing only trains on the gold-standard sequence, potentially underutilizing the decoder’s ability to learn from incorrect candidates. This could be mitigated by also training on negative examples (incorrect candidates) with a contrastive loss (see below).
Alternatives or Complementary Approaches:
Scheduled Sampling: To address exposure bias, you could use scheduled sampling, where the decoder sometimes processes gold-standard feature sequences and sometimes processes sampled candidate sequences (e.g., randomly selected from the candidate set). The probability of using gold-standard sequences decreases over training epochs. This hybrid approach balances stability and robustness but adds complexity.
Training on All Candidates: Instead of only teacher forcing with the gold-standard sequence, you could train the decoder to score all candidate analyses (correct and incorrect) for each sentence, using a loss function that maximizes the score of the gold-standard analysis while minimizing scores for incorrect ones. For example, use a margin-based loss:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi mathvariant="script">L</mi><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><munder><mo>∑</mo><mrow><mi>c</mi><mo mathvariant="normal">≠</mo><msup><mi>c</mi><mo>∗</mo></msup></mrow></munder><mi>max</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><mi>c</mi></mrow></msub><mo>−</mo><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><msup><mi>c</mi><mo>∗</mo></msup></mrow></msub><mo>+</mo><mtext>margin</mtext><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\mathcal{L} = \sum_{i=1}^B \sum_{c \neq c^*} \max(0, s_{i,c} - s_{i,c^*} + \text{margin})</annotation></semantics></math>
L= 
i=1
∑
B
​
  
c

=c 
∗
 
∑
​
 max(0,s 
i,c
​
 −s 
i,c 
∗
 
​
 +margin)
where <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><mi>c</mi></mrow></msub></mrow><annotation encoding="application/x-tex"> s_{i,c} </annotation></semantics></math>s 
i,c
​
  is the score for candidate <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi></mrow><annotation encoding="application/x-tex"> c </annotation></semantics></math>c, <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><msup><mi>c</mi><mo>∗</mo></msup></mrow></msub></mrow><annotation encoding="application/x-tex"> s_{i,c^*} </annotation></semantics></math>s 
i,c 
∗
 
​
  is the score for the gold-standard candidate, and <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mtext>margin</mtext></mrow><annotation encoding="application/x-tex"> \text{margin} </annotation></semantics></math>margin is a hyperparameter (e.g., 0.1). This approach trains the decoder to distinguish correct from incorrect analyses explicitly, potentially improving performance without relying solely on teacher forcing.
No Teacher Forcing: You could train without teacher forcing by processing all candidate sequences (gold and non-gold) and optimizing the score of the gold-standard analysis. This avoids exposure bias entirely but may slow convergence, as the decoder must learn from potentially noisy inputs early in training.
Recommendation:
Use Teacher Forcing Initially: Given your gold-standard dataset and emphasis on interpretability, start with teacher forcing to ensure stable training and meaningful attention weights. Feed the gold-standard feature sequence to the decoder during training, computing the score and optimizing with a cross-entropy or binary classification loss (treating the gold-standard score as the positive class).
Incorporate Negative Candidates: To improve robustness, combine teacher forcing with training on incorrect candidates using a margin-based or contrastive loss. For each sentence, process all candidate analyses in a batch (as discussed above), scoring both the gold-standard and incorrect analyses. This ensures the model learns to down-rank incorrect analyses while benefiting from the stability of teacher forcing.
Optional Scheduled Sampling: If you observe exposure bias (e.g., poor performance on incorrect candidates during inference), consider scheduled sampling later in training to gradually expose the decoder to non-gold candidate sequences. However, given your scoring task, this may be unnecessary if the contrastive loss performs well.
Implementation Tip: When batching with teacher forcing, include the gold-standard feature sequence in the batched candidate tensor (e.g., as the first candidate for each sentence). Use a mask to ensure the loss only considers the gold-standard score during teacher-forcing steps, while also computing scores for other candidates for the contrastive loss.
Impact on Attention Visualization: Teacher forcing ensures that attention weights computed during training reflect the gold-standard feature sequence, making your heatmap visualizations (subword tokens vs. morphological features) more interpretable. Without teacher forcing, attention weights for incorrect candidates might dilute interpretability unless you specifically visualize weights for the gold-standard sequence during inference.
Integration with Your Model Architecture
Here’s how batching candidate analyses and teacher forcing fit into your existing encoder-decoder architecture:

Encoder: No changes needed. The encoder processes the batched subword token sequences (shape <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>T</mi><mo>×</mo><mi>E</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \mathbb{R}^{B \times T \times E} </annotation></semantics></math>R 
B×T×E
 , where <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>E</mi></mrow><annotation encoding="application/x-tex"> E </annotation></semantics></math>E is the embedding dimension) to produce hidden states <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mover accent="true"><mi>h</mi><mo>⃗</mo></mover><mi>t</mi></msub><mo>∈</mo><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>T</mi><mo>×</mo><mi>H</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \vec{h}_t \in \mathbb{R}^{B \times T \times H} </annotation></semantics></math> 
h
  
t
​
 ∈R 
B×T×H
 . Use padding and attention masks to handle variable sequence lengths.
Decoder:
Batching: Process a batched tensor of candidate analyses <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>C</mi><mo>×</mo><mi>M</mi><mo>×</mo><mi>F</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \mathbb{R}^{B \times C \times M \times F} </annotation></semantics></math>R 
B×C×M×F
 , where each candidate is a padded sequence of feature embeddings. For teacher forcing, ensure the gold-standard feature sequence is included in the batch (e.g., as <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex"> c=1 </annotation></semantics></math>c=1 for each sentence). Apply masks to ignore padded features and dummy candidates.
Teacher Forcing: At each decoder step <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>m</mi></mrow><annotation encoding="application/x-tex"> m </annotation></semantics></math>m, feed the gold-standard feature embedding <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msubsup><mi>f</mi><mi>m</mi><mo>∗</mo></msubsup></mrow><annotation encoding="application/x-tex"> f_m^* </annotation></semantics></math>f 
m
∗
​
  (from the gold-standard sequence) alongside the context vector <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>c</mi><mi>m</mi></msub></mrow><annotation encoding="application/x-tex"> c_m </annotation></semantics></math>c 
m
​
  (from Bahdanau attention). Compute the decoder state:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mover accent="true"><mi>s</mi><mo>⃗</mo></mover><mi>m</mi></msub><mo>=</mo><mtext>LSTM</mtext><mo stretchy="false">(</mo><mo stretchy="false">[</mo><msubsup><mi>f</mi><mi>m</mi><mo>∗</mo></msubsup><mo separator="true">;</mo><msub><mi>c</mi><mi>m</mi></msub><mo stretchy="false">]</mo><mo separator="true">,</mo><msub><mover accent="true"><mi>s</mi><mo>⃗</mo></mover><mrow><mi>m</mi><mo>−</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\vec{s}_m = \text{LSTM}([f_m^*; c_m], \vec{s}_{m-1})</annotation></semantics></math>
s
  
m
​
 =LSTM([f 
m
∗
​
 ;c 
m
​
 ], 
s
  
m−1
​
 )
Simultaneously, process all candidate sequences to compute their scores for the contrastive loss, using the same encoder hidden states but different feature sequences.
Attention: Bahdanau attention computes weights <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>α</mi><mrow><mi>t</mi><mo separator="true">,</mo><mi>m</mi></mrow></msub><mo>∈</mo><msup><mi mathvariant="double-struck">R</mi><mrow><mi>B</mi><mo>×</mo><mi>C</mi><mo>×</mo><mi>T</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \alpha_{t,m} \in \mathbb{R}^{B \times C \times T} </annotation></semantics></math>α 
t,m
​
 ∈R 
B×C×T
  for each batch, candidate, and decoder step. For teacher forcing, the weights for the gold-standard sequence (<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex"> c=1 </annotation></semantics></math>c=1) are used for the primary loss and visualization. Masks ensure attention ignores padded tokens.
Loss Function: Combine a cross-entropy loss for the gold-standard score (with teacher forcing) and a margin-based loss for all candidates:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi mathvariant="script">L</mi><mo>=</mo><mo>−</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><msup><mi>c</mi><mo>∗</mo></msup></mrow></msub><mo stretchy="false">)</mo><mo>+</mo><mi>λ</mi><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><munder><mo>∑</mo><mrow><mi>c</mi><mo mathvariant="normal">≠</mo><msup><mi>c</mi><mo>∗</mo></msup></mrow></munder><mi>max</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><mi>c</mi></mrow></msub><mo>−</mo><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><msup><mi>c</mi><mo>∗</mo></msup></mrow></msub><mo>+</mo><mtext>margin</mtext><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\mathcal{L} = -\sum_{i=1}^B \log(s_{i,c^*}) + \lambda \sum_{i=1}^B \sum_{c \neq c^*} \max(0, s_{i,c} - s_{i,c^*} + \text{margin})</annotation></semantics></math>
L=− 
i=1
∑
B
​
 log(s 
i,c 
∗
 
​
 )+λ 
i=1
∑
B
​
  
c

=c 
∗
 
∑
​
 max(0,s 
i,c
​
 −s 
i,c 
∗
 
​
 +margin)
where <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>λ</mi></mrow><annotation encoding="application/x-tex"> \lambda </annotation></semantics></math>λ balances the two terms (e.g., <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>λ</mi><mo>=</mo><mn>1.0</mn></mrow><annotation encoding="application/x-tex"> \lambda = 1.0 </annotation></semantics></math>λ=1.0), and <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo separator="true">,</mo><mi>c</mi></mrow></msub></mrow><annotation encoding="application/x-tex"> s_{i,c} </annotation></semantics></math>s 
i,c
​
  is the sigmoid or softmax score for candidate <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi></mrow><annotation encoding="application/x-tex"> c </annotation></semantics></math>c.
Attention Visualization: For your heatmap, extract the attention weights <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>α</mi><mrow><mi>t</mi><mo separator="true">,</mo><mi>m</mi></mrow></msub></mrow><annotation encoding="application/x-tex"> \alpha_{t,m} </annotation></semantics></math>α 
t,m
​
  for the gold-standard candidate (<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex"> c=1 </annotation></semantics></math>c=1) during training or inference. Plot a matrix of shape <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi mathvariant="double-struck">R</mi><mrow><mi>M</mi><mo>×</mo><mi>T</mi></mrow></msup></mrow><annotation encoding="application/x-tex"> \mathbb{R}^{M \times T} </annotation></semantics></math>R 
M×T
 , with rows as feature indices and columns as subword tokens, using a library like Matplotlib or Seaborn. Teacher forcing ensures these weights reflect the correct feature sequence, enhancing interpretability.
Practical Considerations
Hyperparameters:
Batch Size: Choose a batch size (e.g., <math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>B</mi><mo>=</mo><mn>32</mn></mrow><annotation encoding="application/x-tex"> B = 32 </annotation></semantics></math>B=32) that fits GPU memory, considering the additional memory for candidate analyses (<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>C</mi><mo>×</mo><mi>M</mi></mrow><annotation encoding="application/x-tex"> C \times M </annotation></semantics></math>C×M).
Margin: Set the margin in the contrastive loss to a small value (e.g., 0.1–0.5) to encourage separation between correct and incorrect scores.
Teacher Forcing Ratio: If using scheduled sampling, start with 100% teacher forcing and decay to 50% over epochs (e.g., linearly over 20 epochs).
Dataset: Your gold-standard dataset enables teacher forcing and provides labels for the contrastive loss. Ensure the dataset includes all candidate analyses (correct and incorrect) from the rule-based analyzer to support batching and negative example training.
Evaluation: Beyond accuracy, consider ranking-based metrics like Mean Reciprocal Rank (MRR) or Precision@1 to evaluate how well the model ranks the gold-standard analysis among candidates.
Code Snippet (Optional)
If you’re implementing in PyTorch, here’s a simplified pseudocode for the decoder with batching and teacher forcing:

python

def decoder(encoder_hidden, candidate_features, gold_features, candidate_mask):
    B, C, M, F = candidate_features.shape  # Batch, candidates, max features, feature dim
    T = encoder_hidden.shape[1]  # Encoder sequence length
    scores = torch.zeros(B, C).to(device)
    attn_weights = torch.zeros(B, C, M, T).to(device)
    
    for c in range(C):  # Process each candidate
        hidden = torch.zeros(B, hidden_dim).to(device)  # Initialize decoder hidden state
        for m in range(M):
            # Use gold-standard features for teacher forcing if c=0
            feature = gold_features[:, m] if c == 0 else candidate_features[:, c, m]
            
            # Bahdanau attention
            energy = torch.tanh(
                torch.einsum("btd,bhd->bht", encoder_hidden, W_a(hidden))
                + U_a(hidden).unsqueeze(1)
            )
            energy = v_a(energy).squeeze(-1)  # Shape: [B, T]
            energy = energy.masked_fill(~candidate_mask[:, c, m], -1e9)
            attn = F.softmax(energy, dim=1)  # Shape: [B, T]
            attn_weights[:, c, m] = attn
            
            # Context vector
            context = torch.einsum("bt,btd->bd", attn, encoder_hidden)
            
            # Decoder step
            input = torch.cat([feature, context], dim=-1)
            hidden = lstm_cell(input, hidden)
        
        # Score candidate
        scores[:, c] = score_layer(hidden).squeeze(-1)
    
    return scores, attn_weights

# Loss
def compute_loss(scores, gold_idx, margin=0.1):
    gold_scores = scores[torch.arange(B), gold_idx]
    ce_loss = -torch.logsigmoid(gold_scores).mean()
    margin_loss = 0
    for c in range(C):
        if c != gold_idx:
            margin_loss += F.relu(scores[:, c] - gold_scores + margin)
    return ce_loss + margin_loss.mean()
This assumes padded tensors and masks for variable lengths. Let me know if you need a full implementation or specific adjustments!

Conclusion
Batching Candidate Analyses: Essential for efficiency and scalability. Batch sentences and their candidate analyses with padding and masks, processing all candidates in parallel for each sentence. This integrates seamlessly with your encoder-decoder architecture.
Teacher Forcing: Recommended to stabilize training and ensure interpretable attention weights, given your gold-standard dataset. Combine with a contrastive loss to train on incorrect candidates, improving robustness without significant exposure bias. Scheduled sampling is optional but likely unnecessary for your scoring task.
Impact on Paper: In your “Model Architecture” section, briefly mention batching for efficiency (e.g., “Candidate analyses are batched with padding and masks for parallel processing”) and teacher forcing for training stability (e.g., “The decoder is trained with teacher forcing on gold-standard feature sequences, augmented with a margin-based loss to down-rank incorrect analyses”). Highlight that these choices support scalability and interpretability.
If you want to update your “Model Architecture” section to reflect these additions, need help with the loss function design, or want code for attention visualization, let me know!





---


Model Architecture

Our model is an encoder-decoder recurrent neural network (RNN) with Bahdanau attention designed to score candidate morphological analyses for a target word in a sentence, resolving morphological ambiguity. The encoder captures the contextual representation of the input sentence, while the decoder uses this context to assign scores to predefined candidate analyses provided by an external rule-based morphological analyzer. To ensure computational efficiency, candidate analyses are batched, and the decoder is trained with teacher forcing augmented by a contrastive loss. Below, we detail the encoder, decoder, and attention mechanism.

Encoder

The encoder is a bidirectional Long Short-Term Memory (LSTM) network that processes the input sentence, tokenized into subword tokens to mitigate data sparsity. For a sentence containing a target word to be disambiguated, the subtokens corresponding to the target word are enclosed with special tokens <TARGET> and </TARGET>. For example, the sentence "The cats run" with the target word "cats" might be tokenized as [The, ca, <TARGET>, ts, </TARGET>, run]. Each subword token is mapped to a dense embedding vector.

For a mini-batch of ( B ) sentences, the subword sequences are padded to the maximum length ( T ) in the batch, forming a tensor of embeddings ( x \in \mathbb{R}^{B \times T \times E} ), where ( E ) is the embedding dimension. The encoder computes forward and backward hidden states using the bidirectional LSTM: [ \vec{h}_t^f = \text{LSTM}f(x_t, \vec{h}{t-1}^f), \quad \vec{h}_t^b = \text{LSTM}b(x_t, \vec{h}{t+1}^b) ] The final hidden state for each time step, ( \vec{h}_t = [\vec{h}_t^f; \vec{h}_t^b] \in \mathbb{R}^{B \times T \times H} ), concatenates the forward and backward states, where ( H ) is the hidden state dimension. The <TARGET> and </TARGET> tokens guide the encoder to emphasize the target word’s context, and an attention mask ensures padded tokens are ignored.

Decoder

The decoder, a unidirectional LSTM, scores each candidate morphological analysis, encoded as a sequence of morphological feature vectors (e.g., [POS: Noun, Number: Plural, Case: Nominative]). Each feature is represented as an embedded vector, forming a sequence ( [f_1, f_2, \dots, f_M] ) for each candidate analysis, where ( M ) is the feature sequence length. For each sentence in the batch, multiple candidate analyses (up to a maximum ( C )) are batched with padding to form a tensor ( \mathbb{R}^{B \times C \times M \times F} ), where ( F ) is the feature embedding dimension. A candidate mask ensures padded features and dummy candidates are ignored.

The decoder operates at the subtoken level, processing each candidate’s feature sequence to produce a score. During training, we use teacher forcing by feeding the gold-standard feature sequence for the correct analysis, stabilizing convergence and ensuring interpretable attention weights. At each decoding step ( m ), the decoder takes the feature embedding (gold-standard for teacher forcing or candidate-specific otherwise), the previous hidden state ( \vec{s}{m-1} ), and a context vector ( c_m ) derived from the encoder’s hidden states via attention: [ \vec{s}m = \text{LSTM}([f_m; c_m], \vec{s}{m-1}) ] After processing the feature sequence, the final hidden state is passed through a feedforward layer with a sigmoid activation to produce a score for each candidate analysis, yielding a score tensor ( \mathbb{R}^{B \times C} ). To improve robustness, the decoder processes all candidate analyses (correct and incorrect) in parallel, and training combines a cross-entropy loss on the gold-standard score with a margin-based loss to down-rank incorrect analyses: [ \mathcal{L} = -\sum{i=1}^B \log(s_{i,c^}) + \lambda \sum_{i=1}^B \sum_{c \neq c^} \max(0, s_{i,c} - s_{i,c^} + \text{margin}) ] where ( s_{i,c^} ) is the score for the gold-standard candidate, ( s_{i,c} ) is the score for other candidates, ( \text{margin} ) is a hyperparameter (e.g., 0.1), and ( \lambda ) balances the terms (e.g., 1.0).

Bahdanau Attention Mechanism

To enhance performance and interpretability, we employ the Bahdanau attention mechanism [Bahdanau et al., 2014]. This mechanism enables the decoder to focus on relevant parts of the input sentence when scoring each morphological feature. For each decoder step ( m ) and candidate ( c ), attention weights ( \alpha_{t,m} \in \mathbb{R}^{B \times C \times T} ) are computed for each encoder hidden state ( \vec{h}t ): [ e{t,m} = v_a^\top \tanh(W_a \vec{h}t + U_a \vec{s}m), \quad \alpha{t,m} = \frac{\exp(e{t,m})}{\sum_{t'=1}^T \exp(e_{t',m})} ] where ( v_a ), ( W_a ), and ( U_a ) are learnable parameters, and ( \vec{s}m ) is the decoder’s hidden state. The context vector ( c_m = \sum{t=1}^T \alpha_{t,m} \vec{h}_t \in \mathbb{R}^{B \times C \times H} ) is a weighted sum of the encoder’s hidden states, capturing the relevant sentence context for the current feature. An attention mask ensures padded subword tokens are ignored.

The attention weights ( \alpha_{t,m} ) provide interpretability by indicating which subword tokens (e.g., "The" or "run") influence the scoring of specific morphological features (e.g., Number: Plural). These weights are visualized as a heatmap for the gold-standard candidate, with rows representing feature indices and columns representing subword tokens, to analyze contextual dependencies.




Addressing Your Points and Questions
Let’s tackle your responses and my follow-up questions to refine the neural network design:

Encoder Input (Bidirectional LSTM):
Using a bidirectional LSTM to process the entire tokenized sentence is a great choice, as it captures both left and right context around the <TARGET> and </TARGET> tokens. This ensures the model considers the full sentence for disambiguation.
The encoder will take the tokenized sentence (including <TARGET> and </TARGET>) as input, with each token mapped to an embedding in the sentence’s embedding space.
Confirmation: The encoder outputs a fixed-size context vector (e.g., the final hidden states of the forward and backward LSTMs concatenated) or a sequence of hidden states for each token, depending on how we use it in the next stage. Do you prefer one of these outputs, or should we decide based on the decoder/scoring design?
Decoder Role and Scoring/Ranking Task:
You’re right that this is a scoring/ranking task rather than a generative one, which makes the traditional encoder-decoder setup (common in tasks like machine translation) less straightforward. Since the goal is to assign probabilities to predefined candidate analyses, we don’t need the decoder to generate sequences but to evaluate or score each candidate.
Options for the Decoder:
Option 1: No Decoder, Direct Scoring:
Instead of a decoder, use the encoder’s output (e.g., a context vector or the hidden states around the <TARGET> tokens) to score each candidate analysis.
For each candidate (a sequence of morphological feature subtokens), encode it using a separate LSTM (in the candidate’s embedding space) to get a candidate representation.
Compute a compatibility score between the encoder’s context vector and each candidate’s representation (e.g., via dot product, cosine similarity, or a small feedforward network).
Apply a softmax over the scores to get probabilities for each candidate.
Option 2: LSTM Decoder for Candidate Processing:
Use a decoder LSTM to process each candidate analysis (sequence of feature subtokens) while attending to the encoder’s output (e.g., using attention mechanisms).
The decoder produces a score for each candidate based on its compatibility with the sentence context.
This approach is closer to an encoder-decoder setup but adapted for scoring rather than generation.
Option 3: Classification Head:
Treat the task as a classification problem over the candidate analyses.
Encode the sentence with the bidirectional LSTM to get a context vector.
For each candidate, encode its feature sequence with a separate LSTM or simply concatenate its embeddings.
Concatenate the sentence context vector and candidate representation, pass through a feedforward network, and output a score for each candidate, normalized via softmax.
Discussion: Since you’re leaning toward a classification-like approach, Option 3 seems closest to your vision, as it directly produces probabilities for each candidate. Option 1 is simpler and avoids a traditional decoder, while Option 2 is more complex but leverages attention for potentially better context-candidate alignment. Given the task’s scoring nature and your uncertainty about the decoder, I’d recommend starting with Option 1 or 3 for simplicity. What areあなたの考え？


{
  "sentence": ["The", "▁", "<TARGET>", "dog", "-s", "</TARGET>", "▁run", "."],
  "candidates": [
    ["NOUN", "PLURAL"],
    ["VERB", "PRESENT"],
    ["NOUN", "SINGULAR"],
    ["VERB", "PAST"]
  ],
  "label": 0
}