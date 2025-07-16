from torch.utils.data import Dataset ,DataLoader
import torch
import tiktoken

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
## primarily we need the DataLoader to perform parallel processing (analyze multiple batches at one time)
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