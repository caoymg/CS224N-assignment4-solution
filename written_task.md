##### **CS 224n: Assignment #4 [written]** 

#### *writer: Yiming Cao*

**1. Neural Machine Translation with RNNs** 

**(g)** 

**Effect on the entire attention computation:** 

• The mask lets the attention scores(*e_t*) of ‘pad’ token be ‘-inf’, since the ‘pad’ token is just a 

placeholder to normalize sentences' length. 

• Thus, the attention distribution(*alpha_t*) is only determined by the attention scores of 

non-pad tokens, which zeros out the probability of ‘pad’ token affecting the attention output. 

**Necessary because:** 

• When passing a tensor of ‘-inf’ to *nn.Softmax*, it returns a tensor of nan, thus the ‘pad’ tokens 

won’t affect the attention distribution and attention output. 

**(h)** 

The model’s corpus BLEU Score: 22.402383626794006 

**(i)** 

i.

• Dot product attention is computationally cheap and direct since it just involves dot product 

between encoder hidden states(*h**i*) and decoder hidden state(*s**t*) at time step *t*. 

• Dot product couldn’t get as much general information as multiplicative attention since the 

latter contains a weight matrix to compute the correlation between source and target words. 

ii. 

• Additive attention outperforms multiplicative attention for larger dimensions since it has 

more sophisticated computation and a hyperparameter to control the attention dimensionality. 

• Additive attention is computationally slower and consumes more space than multiplicative 

attention which just use a matrix multiplication . 

**2. Analyzing NMT Systems** 

**(a)** 

• It’s more space-efficient to model at a subword-level since the embeddings matrix is much 

smaller than that of word-level. 

• Modeling at subword-level can alleviate the problems of out-of-vocabulary entries and 

spelling mistakes since Cherokee is a polysynthetic language. 

**(b)** 

• Character-level and subword embeddings do not encode the same deep sematics as word 

embeddings do. 

• Character-level and subword embeddings have fewer entries to be embedded.**(c)** 

• Multilingual models learn shared representations between linguistically similar high-resource 

language and low-resource language without the need for external constraints, which forms a 

positive transfer towards the low-resource languages, improving the NMT performance. 

**(d)** 

i.

• **Reason:** the model pay attention to the same place for many times, thus the result seems to 

stammer. In the process of translation, the ideal state should be paying attention to relatively 

different information at every time step. 

• **Solution:** use multi-head self-attention in both encoder and decoder to make the model focus 

on different information from different subspaces at every time step. 

ii. 

• **Reason:** there is a one-to-many alignment between the source*(Ulihelisdi)* and target 

words*(he's happy/thankful)* and the source words*(Ulihelisdi)* does have the meaning of 

“*joy*” and contains even more meanings like “*merry christmas*”, thus the word is easily 

mistranslated. 

• **Solution:** since the translation have the right lingustic structure and overall semantic but 

wrong subject, so increasing the depth of the model to improve the learning ability of it may 

help. 

iii. 

• **Reason:** due to the mechanism of bidirectional attention, the exclamation mark at the end of 

the source sentence makes the adjacent contents be translated into exclamatory sentence form, 

thus affecting the semantics of the whole sentence. 

• **Solution:** we can mask the exclamation mark in the attention computation process, and add it 

directly in the final translation result. 

**(e)** 

i.

• **Reference Translation:** *And I always tell people I don't want to seem like a scientist.* 

• **NMT Translation:** *And I always tell people that I don't want to show up looking like a* 

*scientist.* 

• The training file does not contain that string (almost) verbatim. This means the MT system 

learned to understand all the information about the source sentence and then do reordering, 

transformation, compression, or expansion of source to the output sentence. 

ii. 

• **Reference Translation:** *And I'd like you to pay attention to the shape of the iceberg and* 

*where it is at the waterline.* 

• **NMT Translation:** *And I'd like to pay attention to the shape of the iceberg and your line of* 

*<unk>* 

• The model’s decoder makes the newly generated word vector as part of the next input to the 

decoder, which provides a similar contextual representation as in the encoder, but is just 

based on the previous words and in only one direction. Therefore, once there is a translation 

error in the middle of the sentence, it is very likely to cause the whole translation error later.**(f)** 

i. 

c1: p1 = (0+1+1+1+0) / (1+1+1+1+1) = 0.6 

p2 = (0+1+1+0) / (1+1+1+1) = 0.5 

len(c) = 5 

len(r) = 4 

BP = 1 

BLEU = exp(0.5*log0.6+0.5*log0.5) = exp(0.5*log0.3) = 0.5477225575051662 

c2: p1 = (1+1+0+1+1) / (1+1+1+1+1) = 0.8 

p2 = (1+0+0+1) / (1+1+1+1) = 0.5 

len(c) = 5 

len(r) = 4 

BP = 1 

BLEU = exp(0.5*log0.8+0.5*log0.5) = exp(0.5*log0.4) = 0.6324555320336759 

c2 is the better translation according to the BLEU score. I agree with it. 

ii. 

c1: p1 = (0+1+1+1+0) / (1+1+1+1+1) = 0.6 

p2 = (0+1+1+0) / (1+1+1+1) = 0.5 

len(c) = 5 

len(r) = 6 

BP = exp(1 - 6/5) = exp(-0.2) 

BLEU = exp(-0.2) * exp(0.5*log0.6+0.5*log0.5) = 0.448437301984003 

c2: p1 = (1+1+0+0+0) / (1+1+1+1+1) = 0.4 

p2 = (1+0+0+0) / (1+1+1+1) = 0.25 

len(c) = 5 

len(r) = 6 

BP = exp(1 - 6/5) = exp(-0.2) 

BLEU = exp(-0.2) * exp(0.5*log0.4+0.5*log0.25) = 0.25890539701513365 

c1 is the better translation according to the BLEU score. I do not agree with it. 

iii. 

• Since there are many valid ways to translate, it is easy to give a good translation a low BLEU 

score if it has low n-gram overlap with the only reference translation. 

iv. 

**Advantages** 

• Using BLEU score as an evaluation metric for Machine Translation is more time-efficient 

and labor-saving. 

• Human evaluation is lack of a unified evaluation rule and hard to quantify, while the BLEU 

computes is based on n-gram precision plus a penalty for too-short system translations which 

can be quantified and provide more accurate evaluation results. 

**Disadvantages** 

• No consideration is given to sentence structure or word order. 

• The accuracy of the BLEU score is affected by the richness of the reference translation. It 

may misjudge a valid translation if it has low n-gram overlap with the reference translation.
