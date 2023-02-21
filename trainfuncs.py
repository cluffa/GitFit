# %%
import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

# %%
import nltk
nltk.download('punkt')

# %%
device = "cuda"

# %%
def get_cfdc():
    urls = [f"https://github.com/cluffa/crossfit-workouts/raw/main/data/{year}.csv" for year in range(2002, 2023)]
    dfs = [pd.read_csv(url) for url in urls]
    df = pd.concat(dfs, ignore_index = True)
    return [w.replace("<|newline|>", "\n") for w in df.text]

def get_cflp():
    text = pd.read_csv("https://github.com/cluffa/crossfit-workouts/raw/main/cflp/wods.txt", sep = "    ", header = None)[0]
    return [w.replace("<|newline|>", "\n") for w in text]

def get_wc():
    df = pd.read_csv("https://github.com/cluffa/crossfit-workouts/raw/main/data/wc.csv")
    return [w.replace("<|newline|>", "\n") for w in df.text]

def get_workouts():
    return get_cfdc() + get_cflp() + get_wc()

def get_posts():
    urls = [f"https://github.com/cluffa/crossfit-workouts/raw/main/cff/data/wl{i}.csv" for i in range(1, 5)]
    dfs = [pd.read_csv(url) for url in urls]
    df = pd.concat(dfs, ignore_index = True)
    ws = df.text.copy()
    ws = [str(w).replace("<|newline|>", "\n") for w in ws]
    return ws

# %%
def text_info(text):
    doc_lengths = []

    for t in text:

        # get rough token count distribution
        tokens = nltk.word_tokenize(t)

        doc_lengths.append(len(tokens))

    doc_lengths = np.array(doc_lengths)

    print(f"percent longer than max: {len(doc_lengths[doc_lengths > 256])/len(doc_lengths)}\n")
    print(f"average length: {np.average(doc_lengths)}\n")

    return sns.histplot(doc_lengths)

# %%
class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=256):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 

# %%
def make_dataset(text, tokenizer, max_length=256):
    return GPT2Dataset(text, tokenizer, max_length = max_length)

# %%
def new_model(model_name = "gpt2", device = device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)

    model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

    model.resize_token_embeddings(len(tokenizer))

    device = torch.device(device)
    model.to(device)

    return model, tokenizer

# %%
def train_model(
    model,
    dataset,
    batch_size = 8,
    epochs = 13,
    learning_rate = 5e-5,
    warmup_steps = 1e2,
    epsilon = 1e-8,
    sample_every = 250,
    prev_stats = []
    ):

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    
    optimizer = AdamW(
        model.parameters(),
        lr = learning_rate,
        eps = epsilon
    )

    total_steps = train_size * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = total_steps
    )

    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))
        
    total_t0 = time.time()

    training_stats = prev_stats
    prev_epochs = len(training_stats)

    model = model.to(device)

    for epoch_i in range(0, epochs):
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()        

            outputs = model(  b_input_ids,
                            labels=b_labels, 
                            attention_mask = b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]  

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                eta = format_time((time.time() - t0) * (len(train_dataloader) - step) / step)
                eta_epoch = format_time((time.time() - t0) * (len(train_dataloader) - step) / step * epochs)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.  Eta: {:} for epoch, {:} for all'.format(step, len(train_dataloader), batch_loss, elapsed, eta, eta_epoch))
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():        

                outputs  = model(
                    b_input_ids,
                    #token_type_ids=None,
                    attention_mask = b_masks,
                    labels=b_labels
                    )
            
                loss = outputs[0]  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)    

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': prev_epochs + epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            },
            ignore_index = True
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    return df_stats

# %%
def plot_df_stats(df_stats):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.xticks([1, 2, 3, 4])

    plt.show()

# %%
def save_model(model, tokenizer, name):
    output_dir = './' + name + '/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# # %%
# model = new_model()

# # %%
# train_dl_posts, validation_dl_posts = make_dataset(get_posts() + get_workouts())

# # %%
# # model trained on posts and workouts combined
# basestats = train_model(
#     model,
#     train_dl_posts,
#     validation_dl_posts,
#     epochs = 5,
#     learning_rate = 5e-4
# )

# # %%
# save_model(model, tokenizer, '-base')

# # %%
# plot_df_stats(basestats)

# # %%
# train_dl_workouts, validation_dl_workouts = make_dataset(get_workouts())

# # %%
# # model trained on workouts only
# stats = train_model(
#     model,
#     train_dl_workouts,
#     validation_dl_workouts,
#     epochs = 15,
#     learning_rate = 5e-6,
#     prev_stats = basestats
# )

# # %%
# plot_df_stats(stats)

# # %%
# # Get all of the model's parameters as a list of tuples.
# params = list(model.named_parameters())

# print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))

# print('==== Embedding Layer ====\n')

# for p in params[0:2]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# print('\n==== First Transformer ====\n')

# for p in params[2:14]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# print('\n==== Output Layer ====\n')

# for p in params[-2:]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# # %%
# save_model(model, tokenizer, '')

# # %%
# model.eval()

# prompt = "<|startoftext|>"

# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# generated = generated.to(device)

# #print(generated)

# sample_outputs = model.generate(
#     generated, 
#     #bos_token_id=random.randint(1,30000),
#     do_sample=True,   
#     top_k=50, 
#     max_length = 300,
#     top_p=0.95, 
#     num_return_sequences=10
# )

# with open("output.txt", "w", encoding="utf-8") as f:
#     for i, sample_output in enumerate(sample_outputs):
#         outstr = "---- {} ----\n{}\n\n".format(i+1, tokenizer.decode(sample_output, skip_special_tokens=True))
#         outstr = outstr.replace("<|newline|>", "\n")
#         f.writelines(outstr)
#         print(outstr)
        
# model.train()

# # %%
# output_dir = './gitfit-model/'
# model = GPT2LMHeadModel.from_pretrained(output_dir)
# tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
# model.to(device)

# # %%



