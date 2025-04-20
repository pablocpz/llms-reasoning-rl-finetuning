import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    #stacking all the training data into a tensor of shape (batch_size, block_size)
    #so we have a batch_size number of sequences of length block_size
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """ #see notebook for reference

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #this basically helps use use this combination of functions for the masked attn grid

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        
        #each head of attention basically creates new enriched token representations
        #the head_size determines how many new features we get for each token
        
        B,T,C = x.shape

        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        
        #updating each token representation (value matrix) with the computed attention weights
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return out 
        #we get n=head_size enriched token representations/patterns of their imfluence in respect
        #to itself and the other tokens, as well as their meaning and position in the
        #sentence

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #nn.ModuleList is a subclass of nn.Module that stores submodules in a list
        
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #concatenating all the enriched token representations from the different attn heads
        
        out = self.dropout(self.proj(out))
        return out
    
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    #we now gather all these enriched token representations (of the self attention heads)
    #and we pass them through a FCC to make it learn patterns between them, followed by a non-linearity
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( #feed forward network
            
            nn.Linear(n_embd, 4 * n_embd),
            #we make the output n of neurons 4 times bigger than the input (paper says 4x in the inner dimension)
            
            nn.ReLU(),
            
            nn.Linear(4 * n_embd, n_embd),
            #then we scale down the neurons to the same size as the input (n_embd is the feature dim of the tokens)
            nn.Dropout(dropout),
            #dropout to prevent overfitting (randomly shutting down neurons apportations to the output)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    #this is the main building block of the transformer architecture
    #- self attention (n_heads) where tokens communicate with each other
    #- followed by a feed forward network (FNN) to learn patterns between the enriched token representations
    

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        #head_size is the size of the token projections for each head in the self attention 
        
        
        self.sa = MultiHeadAttention(n_head, head_size) #attn with n heads with m dimension size
        self.ffwd = FeedFoward(n_embd) #fcc
        
        #the two layer norms applied to each sub-block
        self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        
        #residual connections around each of the two sub-blocks 
        #this basically means the result of the operations here is added to the original input
        
        #in the self-attn case, it means that the token representations get added these new computed information to them
        #this fork is useful for the gradient flow
                
        x = x + self.sa(self.ln1(x)) #layer norm before self attn

        x = x + self.ffwd(self.ln2(x))  #layer attn before fully connected layer
        return x
    
    
#we'll also use this proposed technique to keep with mean=0 and std=1 for each samples batch 
#this is a normalization technique that helps the model learn better and faster

class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


#FINAL WRAPPER of the decoder-only LLM (GPT)
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        

        if targets is None: #in case we are not training the model, we just want to get the logits computed above
            loss = None
        else:
            
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #in order to generate max_new_tokens, we need to run the model for each new token
            #and add each new prediction to the input sequence
            
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            
            
            #running the forward() of the same class, also called with self()
            logits, loss = self(idx_cond)
            
            #-------------------------------------------!
            # focus only on the last time step
            #renember we did all these masked-self-attn +linear for each token in the sequence
            #so we need to get the last one as it encodes the whole information of the sequence, so the corresponding logits are what we want
            
            
            logits = logits[:, -1, :] # becomes (B, C), we get the logits for each batch (prob for the next token given the whole vocab)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution of probabilities of each vocab token of taking place
          
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence #we add the token to the input sequence (autoregressive model)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx