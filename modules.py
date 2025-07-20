import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
from torch.utils.data import Dataset ,DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride) : ## we need the stride to determine where to start the next input-output pair
          
        self.input_ids=[]
        self.target_ids=[]
        
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
        for i in range(0,len(token_ids)-max_length,stride): # We are using sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i:i+max_length]        
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self): ## total number of raws in the dataset   
        return len(self.input_ids)
    
    def __getitem__(self,idx) : ## return a single raw from the dataset 
        return self.input_ids[idx] , self.target_ids[idx]
    
def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0) :
    ## for the batch_size : it is used for the model to update its parameters after each batch_size inputs being processed
    ## num_workers is for parallel processing over multiple threads of the cpu
    
    tokenizer = tiktoken.get_encoding("gpt2")
    ## creating the dataset
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    
    ## creating the dataloader 
    
    Data_Loader = DataLoader( ## this method will check the __getitem__() and __len__() method in the  GPTDatasetV1 dataset class 
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return Data_Loader ## input-output pairs 

def generate_next_token(model,idx,max_new_tokens,context_size) : 
    for _ in range(max_new_tokens) : 
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond) ## logits will be of shape [batch_size , num_tokens , vocab_size]
        
        logits = logits[:,-1,:] ## we need only to focus on the last time step (which correspond to the position of the next token to predict)
        ## logits is now of shape [batch_size , vocab_size]
        probs  = F.softmax(logits,dim=-1) ## we want the softmax to be calculated over the 50257 values / [batch_size , 1]
        idx_next = torch.argmax(probs,dim=-1,keepdim=True)
        idx = torch.cat((idx , idx_next) , dim=1)
    return idx


class MultiHeadAttention(nn.Module) : 
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,bias=False):
        super().__init__()
        assert(d_out % num_heads ==0 ) ,"d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        
        self.head_dim = d_out // num_heads
        
        self.w_q = nn.Linear(d_in,d_out,bias=bias)
        self.w_k = nn.Linear(d_in,d_out,bias=bias)
        self.w_v = nn.Linear(d_in,d_out,bias=bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout  = nn.Dropout(dropout) 
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
    
    def forward(self,x):
        b,num_tokens,d_embed= x.shape
        
        queries = self.w_q(x) ## (b , num_tokens , d_out)
        #print(f"queries :\n {queries}")
        keys    = self.w_k(x) ## (b , num_tokens , d_out)
        #print(f"keys    : \n {keys}")
        values  = self.w_v(x) ## (b , num_tokens , d_out)
        #print(f"values  : \n {values}")
        ## (b , num_tokens , d_out) -------->  (b , num_tokens , num_heads , head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        keys    = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values  = values.view(b,num_tokens,self.num_heads,self.head_dim)
        #print(f"queries after view :\n {queries}")
        #print(f"keys after view    :\n {keys}")
        #print(f"values after view  :\n {values}")
        # (b , num_tokens , num_heads , head_dim) ------> (b , num_heads , num_tokens , head_dim)
        ## in this case each head should be able to access all the tokens but with different embeddings (keys values splitted over the different heads)
        keys    = keys.transpose(1,2) 
        queries = queries.transpose(1,2) 
        values  = values.transpose(1,2) 
        #print(f"queries after transpose :\n {queries}")
        #print(f"keys after transpose    :\n {keys}")
        #print(f"values after transpose  :\n {values}")
        attn_scores = queries @ keys.transpose(-1,-2)
        #print(f"Attention Scores : \n {attn_scores}")
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores_masked = attn_scores.masked_fill_(mask_bool,-torch.inf)
        #print(f"Attention Scores masked : \n {attn_scores_masked}")
        attn_weights = F.softmax(attn_scores_masked / keys.shape[-1]**0.5,dim=-1)
        #print(f"Attention Weights : \n {attn_weights}")
        attn_weights = self.dropout(attn_weights)
        
        context = ( attn_weights @ values ).transpose(1,2)# (b , num_heads , num_tokens , head_dim) ------> (b , num_tokens , num_heads , head_dim) 
        
        context = context.contiguous().view(b,num_tokens,self.d_out)
        
        context = self.out_proj(context) # optional linear projection layer 
        
        return context

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.MHattention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            bias=cfg["qkv_bias"]
        )
        self.layer_norm_1 = LayerNorm(emb_dim=cfg['emb_dim'])
        self.layer_norm_2 = LayerNorm(emb_dim=cfg['emb_dim'])
        self.ffn          = FeedForward(cfg)
        self.dropout      = nn.Dropout(cfg["drop_rate"])
        
        
    def forward(self,x) : 
        res_1 = x 
        x = self.layer_norm_1(x)
        x = self.MHattention(x)
        x = self.dropout(x)
        x = x + res_1
        
        res_2 = x 
        
        x = self.layer_norm_2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + res_2
        
        return x 

#@@@@@ Overview of the model architecture @@@@@ 

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb    = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"] , cfg["vocab_size"],bias=False)
        
        
    def forward(self,in_idx) : 
        batch_size , seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits

## building the layer normalization class 
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5 ## since we are dividing by the square root of the variance , we use eps to prevent division by zero 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        ## scale and shift are learned parameters that the model can update if this can enhance the training
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
        
class FeedForward(nn.Module): 
    def __init__(self,cfg) : 
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"] , 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"] , cfg["emb_dim"])
        )
    def forward(self,x):
        return self.layers(x)
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))



def assign(left,right):
    if left.shape != right.shape : 
        raise ValueError(f" Shape mismatch .Left: {left.shape}  Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


## Loading the weights into our gpt model
import numpy as np 
def load_weights_into_gpt(gpt , params) : 
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight,params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight,params["wte"])
    
    for b in range(len(params["blocks"])) : 
        ## Attention Q,K,V weights
        q_w,k_w,v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"] , 3 , axis=-1)
        gpt.trf_blocks[b].MHattention.w_q.weight =assign(gpt.trf_blocks[b].MHattention.w_q.weight ,q_w.T)
        gpt.trf_blocks[b].MHattention.w_k.weight   =assign(gpt.trf_blocks[b].MHattention.w_k.weight ,k_w.T)
        gpt.trf_blocks[b].MHattention.w_v.weight =assign(gpt.trf_blocks[b].MHattention.w_v.weight ,v_w.T)
        ## Attention Q,K,V bias
        q_b , k_b , v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"],3,axis=-1)
        gpt.trf_blocks[b].MHattention.w_q.bias =assign(gpt.trf_blocks[b].MHattention.w_q.bias ,q_b)
        gpt.trf_blocks[b].MHattention.w_k.bias   =assign(gpt.trf_blocks[b].MHattention.w_k.bias ,k_b)
        gpt.trf_blocks[b].MHattention.w_v.bias =assign(gpt.trf_blocks[b].MHattention.w_v.bias ,v_b)
        ## Attention out_proj weights and bias
        gpt.trf_blocks[b].MHattention.out_proj.weight = assign(gpt.trf_blocks[b].MHattention.out_proj.weight,params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].MHattention.out_proj.bias   = assign(gpt.trf_blocks[b].MHattention.out_proj.bias,params["blocks"][b]["attn"]["c_proj"]["b"])
        ## feed forward layer weights and bias 
        gpt.trf_blocks[b].ffn.layers[0].weight = assign(gpt.trf_blocks[b].ffn.layers[0].weight,params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[0].bias   = assign(gpt.trf_blocks[b].ffn.layers[0].bias,params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ffn.layers[2].weight = assign(gpt.trf_blocks[b].ffn.layers[2].weight,params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[2].bias   = assign(gpt.trf_blocks[b].ffn.layers[2].bias,params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].layer_norm_1.scale = assign(gpt.trf_blocks[b].layer_norm_1.scale,params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layer_norm_1.shift = assign(gpt.trf_blocks[b].layer_norm_1.shift,params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].layer_norm_2.scale = assign(gpt.trf_blocks[b].layer_norm_2.scale,params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layer_norm_2.shift = assign(gpt.trf_blocks[b].layer_norm_2.shift,params["blocks"][b]["ln_2"]["b"])
        
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        
        gpt.out_head.weight  = assign(gpt.out_head.weight,  params["wte"])



def generate(model,idx,max_new_tokens,context_size,
             temp=0.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens) : 
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        if top_k is not None : 
            top_logits , _ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )
        if temp > 0.0 : 
            logits = logits / temp
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
        else : 
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next == eos_id : 
            break
        
        idx = torch.cat((idx,idx_next),dim=1)
    return idx

def text_to_token_ids(text , tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) ## add the batch dimension 
    return encoded_tensor


def ids_token_to_text(ids , tokenizer) : 
    flat = ids.squeeze(0) ## remove batch dimension
    decoded = tokenizer.decode(flat.tolist())
    return decoded
 
def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0,1) , target_batch.flatten())
    return loss

    
def calc_loss_loader(data_loader,model,device,num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0 : 
        return float("nan")
    elif num_batches is None : 
        num_batches = len(data_loader) ## if no batch size is given , we iterate over all batches 
    else : 
        num_batches = min(num_batches , len(data_loader))
    
    for i ,(input_batch , target_batch) in enumerate(data_loader) : 
        if i<num_batches : 
            loss = calc_loss_batch(input_batch , target_batch , model , device)
            total_loss += loss.item() ##    sums loss for each batch
        else : 
            break
    return total_loss / num_batches ## average loss over all batches 


def train_model_simple(model,train_loader,val_loader,optimizer,device,
                       num_epochs,eval_freq,eval_iter,start_context,tokenizer) : 
    train_losses , val_losses ,track_tokens_seen = [],[],[]
    tokens_seen ,global_step = 0,-1
    for epoch in range(num_epochs):
        model.train()
        for input_batch , target_batch in train_loader : 
            optimizer.zero_grad() ## resets the gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()  ## calculates loss gradients
            optimizer.step() ## updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1 
            
            if global_step % eval_freq ==0 : 
                train_loss , val_loss = evaluate_model(
                    model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train Loss {train_loss:.3f}  "
                      f"Val Loss {val_loss:.3f}")
        generate_print_sample(model,tokenizer,device,start_context)
    return train_losses , val_losses , track_tokens_seen
     
def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss , val_loss

def generate_print_sample(model,tokenizer,device,start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_next_token(model,idx=encoded,max_new_tokens=50,context_size=context_size)
        decoded = ids_token_to_text(token_ids,tokenizer)
        print(decoded.replace("\n" , " "))
    model.train()    
    
import matplotlib.pyplot as plt
def plot_values(epochs_seen,examples_seen,train_values,val_values,label="loss") : 
    fig , ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen,train_values,label=f"Training {label}")
    ax1.plot(epochs_seen,val_values,linestyle='-.',label=f"validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    
    ax2 = ax1.twiny()
    ax2.plot(examples_seen,train_values,alpha=0)
    ax2.set_xlabel("examples seen")
    fig.tight_layout()        
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
    
    