from utils import load_reddit_dataset
import logging
import sys
import os
import random
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments, 
    AutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    set_seed
)

seed = random.randint(0, 1000)
set_seed(seed)

os.environ["WANDB_PROJECT"] = "YOUR_PROJECT_NAME"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="./models/gpt2",
    cache_dir="./models/gpt2",
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.model_max_length = 1024
tokenizer.model_max_length = 1024

model = AutoModelWithLMHead.from_pretrained(
    pretrained_model_name_or_path="./models/gpt2",
    cache_dir="./models/gpt2",
).to("cuda:0")

max_length = tokenizer.model_max_length

'''
	load datasets
'''
dataset = load_reddit_dataset()

def preprocess(examples):
    instructions = examples["question"]
    replys = examples["chosen"]
    for i in range(len(instructions)):
        instruction = instructions[i]
        reply = replys[i]
        reply = f"Reply to: {instruction}" + tokenizer.eos_token + reply
        replys[i] = reply

    return tokenizer(replys, padding="max_length", truncation=True, max_length=max_length)

dataset = dataset.map(preprocess, batched=True, batch_size=1000, remove_columns=["question", "chosen", "rejected", "question_id", "upvotes_chosen" ,"upvotes_rejected"], desc="Running tokenizer on dataset")

dataCollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

'''
    train the model
'''

training_args = TrainingArguments(
    output_dir="./models/reddit",
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=10, # number of training epochs
    report_to="wandb",
    evaluation_strategy='epoch',
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4, # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    save_steps=10000, # after # steps model is saved
    warmup_steps=10000,# number of warmup steps for learning rate scheduler,
    learning_rate=1e-5,
    run_name="reddit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=dataCollator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
trainer.save_model()

