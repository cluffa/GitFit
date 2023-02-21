from trainfuncs import *

def get_workout(model, tokenizer, n = 1, extra = ""):
    model = model.to(device)
    model.eval()

    generated = torch.tensor(tokenizer.encode("<|startoftext|>" + extra)).unsqueeze(0)
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

    for sample_output in sample_outputs:
        outstr = tokenizer.decode(sample_output, skip_special_tokens=True)
        outstr = outstr.replace("<|newline|>", "\n")
        print(f"\n{outstr}\n")

if __name__ == "__main__":
    model, tokenizer = new_model("cluffa/gitfit-model")
    get_workout(model, tokenizer, 10)