from utils import load_reddit_dataset
import logging
import sys
import os
import random
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from trl import RewardConfig, RewardTrainer
os.environ["WANDB_PROJECT"] = "YOUR_PROJECT_NAME"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="./models/gpt2",
    cache_dir="./models/gpt2",
    local_files_only=True
)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token

tokenizer.model_max_length = 512

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="./models/gpt2",
    cache_dir="./models/gpt2",
    num_labels = 1,
).to("cuda:0")

model.config.pad_token_id = tokenizer.pad_token_id

max_length = tokenizer.model_max_length

'''
	load datasets
'''
train_dataset, eval_dataset = load_reddit_dataset()

def preprocess(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": []
    }
    head = "Reply to: "

    for question, chosen, rejected in zip(examples["question"], examples["chosen"], examples["rejected"]):
        chosen = head + question + tokenizer.sep_token + chosen
        rejected = head + question + tokenizer.sep_token + rejected
        tokenizer_output_chosen = tokenizer(chosen, padding="max_length", truncation=True, max_length=max_length)
        tokenizer_output_rejected = tokenizer(rejected, padding="max_length", truncation=True, max_length=max_length)
        new_examples["input_ids_chosen"].append(tokenizer_output_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenizer_output_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenizer_output_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenizer_output_rejected["attention_mask"])
    return new_examples


train_dataset = train_dataset.map(preprocess, batched=True, batch_size=1000, remove_columns=["question", "chosen", "rejected", "question_id", "upvotes_chosen" ,"upvotes_rejected"], desc="Running tokenizer on dataset")
eval_dataset = eval_dataset.map(preprocess, batched=True, batch_size=1000, remove_columns=["question", "chosen", "rejected", "question_id", "upvotes_chosen" ,"upvotes_rejected"], desc="Running tokenizer on dataset")



'''
    train the model
'''
reward_args = RewardConfig(
    output_dir="./models/rm",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=1.41e-5,
    report_to="wandb",
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=500,
    eval_steps=0.1,
    evaluation_strategy="steps",
    max_length=512,
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=reward_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
trainer.save_model()

