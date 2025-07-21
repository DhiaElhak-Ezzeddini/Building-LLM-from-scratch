import tiktoken
import torch
import chainlit
from modules import GPTModel
from modules import(
    generate,
    text_to_token_ids,
    ids_token_to_text,
)
from gpt_download import download_and_load_gpt2 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")


def get_model_and_tokenizer():
    GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True
    }
    tokenizer = tiktoken.get_encoding("gpt2")
    chosen_model = "gpt2_medium (355M)"
    model_size = chosen_model.split(" ")[-1].lstrip("(").rstrip(")")
    print(f"model size : ",model_size)
    settings , params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
    )
    model_fine_tuned = GPTModel(GPT_CONFIG_355M)
    state_dict= torch.load("instruction.pth" , weights_only=True)
    model_fine_tuned.load_state_dict(state_dict)
    model_fine_tuned.to(device)
    model_fine_tuned.eval()
    return tokenizer , model_fine_tuned , GPT_CONFIG_355M
def extract_response(response_text , input_text) : 
    return response_text[len(input_text):].replace("### Response:","").strip()


tokenizer , model , model_config = get_model_and_tokenizer()

@chainlit.on_message
async def main(message : chainlit.Message) : 
    """ 
        the main chainlit function
    """
    torch.manual_seed(123)
    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """
    
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt,tokenizer).to(device),
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )
    text = ids_token_to_text(token_ids , tokenizer)
    response = extract_response(text , prompt)
    await chainlit.Message(
        content=f"{response}",
    ).send()
