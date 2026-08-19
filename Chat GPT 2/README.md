# Recreating GPT 2

## The Transformer architecture

The transformer architecture consists of two main steps, Attention and a Neural Network.

The input to our model is `B x T x C`, where `B` is the batch size, `T` is the sequence length, and `C` is the embedding.

## Post Norm vs Prenorm
In the original transformer paper, they perform post norm normalization, where the normalization occurs after the residual connections.
In our GPT2 model, we employ pre norm normalization. This can be seen in the Block class.
In the Block class, our forward method performs the normalization first, THEN does the residual connection.
This keeps a "clean" residual stream, where our gradients can flow through residual connections unchanged and not have normalization applied to them.

## Attention

The final output of attention is more "contextualized" representation of a user's input tokens.

This is accomplished using Query, Key, and Value matrices.

- The Query matrix is "what a token looks for".
- The Key matrix is "what a token offers".
- The Value matrix is "how a token affects the representation of other tokens" when other tokens attend to this one.

In `models.py`, attention is implemented in the `CausalSelfAttention` class.

We initialize `self.c_attn` to be `n_embd, 3*n_embd` to create 3 separate `n_embd` dimension matrices, for Query, Key, and Value.

For multi-headed attention, we split the embedding dimension of Q K and V into #head number of matrices. We apply each attention head separately, before concatenating each result back together.

A projection matrix mixes the results of all the separate attention heads.

Multi headed attention allows us to represent multiple relationships in parallel.

## Causal Masking

GPT-2 is an autoregressive model, meaning it predicts the next token using only the tokens that came before it.
If we didn't perform masking, we would be leaking information on the token we want to predict next.
This is why the attention layer is causal. Token 5 can attend to tokens 1 through 5, but it cannot attend to token 6 or any future token. 

When training our model, transformers often train next token prediction on every word in the sentence, not JUST the final word.
For example, "the cat sat on the rat", "the" -> "cat", "the cat" -> "sat", "the cat sat on the" -> "rat".
When predicting cat, we mask that row so "the" can only attend to itself, for "the cat sat on the", we allow the latest "the" to attend to everything except rat.
## The math

$$
Q, K, V \in \mathbb{R}^{T \times \frac{C}{\text{num heads}}}
$$

Attention Calculation:

$$
A = \frac{QK^T}{\sqrt{\text{head size}}}
$$

$$
A \in \mathbb{R}^{T \times T}
$$

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{\text{head size}}}\right)V
$$

$$
\text{Attention}(Q, K, V) \in \mathbb{R}^{T \times \frac{C}{\text{num heads}}}
$$

Finally, we concat all these individual heads together, resulting in a `T x C` output, and then mix their results together using a projection matrix.

## MLP

Why do we need an MLP in a transformer?

The MLP allows for individual NONLINEAR transformation of our tokens.

Our MLP is in `models.py`, and it uses a GELU nonlinearity.

## The Transformer Block

The transformer block is composed of a layer normalization, self attention, another layer normalization, and our MLP.

## The GPT Class

### GPTConfig

- Block Size is the maximum context length of our model, set to 1024 tokens.
- Vocab Size is how many tokens our model can predict which is 50,257 tokens.
- n_layer is how many layers are in the transformer, which is 12
- n_head is how many attention heads we do per self attention calculation, which is 12
- n_embd is our transformers representation embedding, which is 768

GPT class consists of our token embeddings, positional embeddings, our transformer blocks, and a classification head.

Positional embeddings give our model information on the order of the input tokens, which it otherwise would not have.

The classification head/language modeling head takes in `n_embd` dimension inputs and outputs logits over the `vocab_size`, and we do softmax to select the next most likely token for next token prediction.

## Weight Tying

In our GPT class, we set:

```python
self.transformer.wte.weight = self.lm_head.weight
```

This means the token embedding matrix and the language modeling head share the same weights. The embedding matrix is used to convert token IDs into vectors, while the lm_head uses those same token vectors to predict which token should come next.

This makes sense because both matrices represent tokens: one for reading tokens as input, and one for predicting tokens as output. If the model's hidden state is similar to a token's embedding, that token gets a higher logit.

This also reduces the number of parameters the model has to learn.

## Flash Attention

In `models.py`, the `CausalSelfAttention` class uses PyTorch's scaled dot-product attention:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
Flash Attention allows us perform attention computation much faster by avoiding creating the T x T attention matrix.

## Learning Rate Scheduler

Our training run uses a **Cosine Learning Rate Scheduler**.

In `trainer.py`, the `get_lr` function controls the learning rate over time. At the start of training, the learning rate increases linearly during warmup until it reaches `max_lr`.

After warmup, the learning rate follows a cosine decay curve down to `min_lr`, which is set to 10% of the maximum learning rate.

This gives training a gentler start, when the random initial weights are shaky and probably inaccurate, then slowly reduces the step size as the model gets closer to convergence.
