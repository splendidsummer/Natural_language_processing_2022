# CS224N & UPC Course 
# UPC Course 2 NERC
## Modern Technology (from paper with code)



# Lecture 7 Seq2Seq & Neural Machine Translation

**Sequence-to-Sequence model** is a **Conditional Language Model**
**Advantage:**
* Better performance: more fluent; better use context; better use of phrase similarities. 
* A single neural network to be optimized end-to-end
* Require much less human engineering effort

**Disadvantage**
* Less interpretable
* Diffcult to control: for example, can not easily specify rules or guidelines for translation; safety concerns!! 

## Application of Seq2Seq 
* Summarization 
* Dialogue (previous utterances $\rightarrow$ next utterance)
* Parsing 
* Code Generation 

## Neural Machine Translation 

### Perplexity 

In general, perplexity is a measurement of how well a probability model predicts a sample. In the context of Natural Language Processing, perplexity is one way to evaluate language models.
A language model is a probability distribution over sentences: it’s both able to generate plausible human-written sentences (if it’s a good language model) and to evaluate the goodness of already written sentences. Presented with a well-written document, a good language model should be able to give it a higher probability than a badly written document, i.e. it should not be “perplexed” when presented with a well-written document. 

$$perplexity(S) = p(w_1, w_2, w_3, ..., w_m) ^ {-\frac{1}{m}}$$

### ROUGE Metrics
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation,[1] is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.


## Training a Neural Machine Translation System 
End-to-end encoder-decoder architecture 
* Single Layer LSTM
![encoder-decoder](images/encoder-decoder.png)

* Multi Layer LSTM
![multi-layer-encoder-decoder](images/multi-layer-encoder-decoder.png) 

### Greedy Decoding 
Generate the target sentence by taking argmax on each step of the decoder 

### Exhauustive Search decoding 
Find a (length T) translation y that maximizes 
$$P(y_1|x) = P(y_1|x)P(y_2|y_1,x) P(y_3|y_2, y_2, x) \\
= \prod_{t=1}^T P(y_t| y_1,..., y_{t-1} | x)  $$

**Complexity is far too expensive!!**

### **Beam Search**
**On each step of decoder, keep track of the k most probable partial translations** (which we call hypotheses)
* **k** is the beam size (in practice **around 5~10**) 
* Score of the sequence of translation is defined as its log probability: 
$$Score(y_1, y_2, ... , y_t) = log P_{LM}(y_1,y_2, ..., y_t | x) = \sum_{t=1}^t P_{LM}(y_i | y_1, ..., y_{i-1}, x)$$

* Problem with this: **longer hypotheses have lower scores.**
* Fix this problem: Normalize by length. Use this to select top one instead: 
$$\frac{1}{t} log P_{LM}(y_i | y_1, ..., y_{i-1}, x) $$  

### Evaluation of MT
#### BLEU (Bilingual Evaluation Understudy)  
BLEU compares the machine-written translation to one or several human-written translation(s), and computes a similarity score based on:
* **n-gram precision** (usually 1, 2, 3, 4 grams) 
* Plus a penalty for too-short system translations 

But BLEU is useful but imperfect!! 
#### Diffuculties of NMT 
Many difficulties remains:
* OOV words 
* Domain mismatch 
* Maintaining context over longer text 
* Low-resource language pairs
* Failures to accurately capture sentence meaning 
* Pronoun (or zero pronoun) resolution errors 
* Morphological agreement errors 
* NMT bais (gender bais)
* Uninterpretable 

### Attention in NMT (from Coursera DL by Andrew Ng)

//

### Lecture 9 Self Attention and Transformers

Issues with recurrent models:
1. Linear interaction distance 
2. Lack of parallelizability: GPU can perform a bunch of independent computations at once! But future RNN hidden state can not be computed in full before past RNN hidden states have been computed. 
3. Word window models aggregate local contexts 
4. Maximum Interaction distance = sequence length / window size

[1] [Attention is all you need](https://arxiv.org/abs/1706.03762)

#### Self Attention 
Barriers and solutions for Self-Attention as a building block. 
1. **Problem**: Since there are no element wise non linearities, self attention is simply performing a re averaging of the value vectors; **Solution**: Apply a feedforward layer to the output of attention, providing non linear activation (and additional expressive power). 
2. **Problem**: Since self attention doesn’t build in order information, we need to encode the order of the
sentence in our keys, queries, and values. **Solutions**: Sinusoidal position representations - concatenate sinusoidal functions of varying periods. 
![postional_encoding](images/positional_encoding.png)



#### Transformers Architecture 

![Multi-head Self-Attention](images/transformers.png) 
1. Multi-head Self-Attention 
   $$A_h = Softmax(\alpha Q_h K_h^T) V_h$$

2. 


##    # Code Implementation 


### ELMO 
![ELMO_VS_BILSTM](images/EIMO_vs_BILSTM.png)
#### Structure
1. Char cnn 
   * **Inputs**: [b, max_sentence_len, max_word_len]      
   * **Preprocess**: Transpose Input: [b*max_sentence_len, embedding_dim, max_word_len].    
   * **CNN layers**： 接下来将其送入到多个kernel size（卷积核大小）和out channel（通道维度）不同的卷积层中，每个filter对应的卷积层最后输出[b*max_sentence_len, out_channel, new_h], 
   * Filter configuration: [ [1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024] ]. After concatenation we finally get 2048 channel in total. ( $new_h=[h-kernelsize+2p/stride] +1 $, $[]$ 表示向下取整。)  
   * 然后[b * max_sentence_len, out_channel, new_h]会经过一个max pool层，在new_h维度上做最大池化，得到[b * max_sentence_len, out_channel]，然后把多个卷积层得到的输出在out_channel维度上进行concat，假设concat最后的维度是n_filters，得到[b*max_sentence_len, n_filters]，其中n_filters=2048. 
   
   * Two High way layer： $$y = g*x + (1-g) * f(A(x)), \quad g = Sigmoid(B(x))$$, here the output dimension keeps $[b*max-sentence-len, n-filters]$
   * Finally a linear projection layer, out_dim = 512. Then reshape the output into $[b, max-sentence-len, output-dim]$ which is the encoding of Char cnn. 

2. Bi-LSTM 

    * char cnn encoder的输出经过forward layer得到[b, max_sentence_len, hidden_size]，其中hidden_size=512  
    * char cnn encoder的输出经过backward layer得到[b, max_sentence_len, hidden_size]，其中hidden_size=512 将前向和后向的在hidden_size为维度做concat得到[b, max_sentence_len, 2*hidden_size]
  
    *注意： Bi_LSTM 有2层，并且他们之间有Skip connections，即两层BiLSTM之间有残差网络相连，也就是说第一层的输出不仅作为第二层的输入，同时也会和第二层的输出相加。返回的时候，也会分别返回2层最后得到的representation，即在Bi_LSTM层最后返回的是[num_layers, b, max_sentence_len, 2*hidden_size]，num_layer=2，表示2层Bi_LSTM层.* 


经过上面，最后句子的表示会得到3层representation：
最底下的层是token_embedding，基于char cnn得到，对上下文不敏感
第一层Bi LSTM得到的对应token的embedding，这一层表示包含了更多的语法句法信息（syntax）
第二层Bi LSTM得到的对应token的embedding，这一层表示包含了更多的词义的信息（word sense）


#### ELMO Implementation 

##### NN Configuration 

```

_options = {"lstm":
               {"use_skip_connections": 'true',
                "projection_dim": 512,
                "cell_clip": 3,
                "proj_clip": 3,
                "dim": 4096,
                "n_layers": 2},
            "char_cnn":
               {"activation": "relu",
                "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
                "n_highway": 2,
                "embedding": {"dim": 16},
                "n_characters": 262,
                "max_characters_per_token": 50}
    }
``` 

##### 

    
```

# pytorch 1.8.0

from allennlp.modules.elmo import Elmo, batch_to_ids

model_dir = 'E:/pretrained_model/elmo_original/'
options_file = model_dir+'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weights_file = model_dir+'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

num_output_representations = 2 # ??? 

elmo = Elmo(
		options_file=options_file,
        weight_file=weights_file,
        num_output_representations=num_output_representations,
        dropout=0
		)

sentence_lists = [['I', 'have', 'a', 'dog', ',', 'it', 'is', 'so', 'cute'],
                  ['That', 'is', 'a', 'question'],
                  ['an']]
    
character_ids = batch_to_ids(sentence_lists) #    
print('character_ids:', character_ids.shape) # [3, 11, 50]    

res = elmo(character_ids)    
print(len(res['elmo_representations']))  # 2   
print(res['elmo_representations'][0].shape)  # [3, 9, 1024]    
print(res['elmo_representations'][1].shape)  # [3, 9, 1024]  


```

#### BLEU 
#### BPE 

#### ELMO













