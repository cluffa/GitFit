import torch
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
device = "cuda"

output_dir = './model_save/'
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model.to(device)

model.eval()

prompt = "<|startoftext|>"

def get_workout(n = 1):
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(
        generated, 
        #bos_token_id=random.randint(1,30000),
        do_sample=True,   
        top_k=50, 
        max_length = 256,
        top_p=0.95, 
        num_return_sequences=n
    )

    for i, sample_output in enumerate(sample_outputs):
        outstr = tokenizer.decode(sample_output, skip_special_tokens=True)
        outstr = outstr.replace("<|newline|>", "\n")
        print(f"\n{outstr}\n")

if __name__ == "__main__":
    get_workout(10)