# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

from utils import load_reddit_dataset

tqdm.pandas()
import os
os.environ["WANDB_PROJECT"] = "YOUR_PROJECT_NAME"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="./models/gpt2",
            query_dataset="./data/reddit/reddit.csv",
            reward_model="./models/rm",
            learning_rate=1.41e-5,
            log_with="wandb",
            mini_batch_size=4,
            batch_size=4,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    use_seq2seq: bool = False
    """whether to use seq2seq models"""
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})


args = tyro.cli(ScriptArguments)


# We then define the arguments to pass to the reddit analysis pipeline.
# We set `return_all_scores` to True to get the reddit score for each token.
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    tokenizer.sep_token = tokenizer.eos_token
    # load imdb with datasets
    train_dataset, eval_dataset = load_reddit_dataset()
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(f"Reply to: {sample['question']}{tokenizer.eos_token}")
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    train_dataset = train_dataset.map(tokenize, batched=False)
    eval_dataset = eval_dataset.map(tokenize, batched=False)
    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")
    dataset = concatenate_datasets([train_dataset, eval_dataset])
    return dataset


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    args.ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the reddit analysis pipeline, passing the model name and the
# reddit analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
model_name = args.ppo_config.reward_model
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        reddit_pipe = pipeline("sentiment-analysis", model=model_name, device=device)
else:
    reddit_pipe = pipeline("sentiment-analysis", model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if reddit_pipe.tokenizer.pad_token_id is None:
    reddit_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if reddit_pipe.model.config.pad_token_id is None:
    reddit_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute reddit score
    sep_token = tokenizer.eos_token
    texts = [q + sep_token + r for q, r in zip(batch["query"], batch["response"])]
    # print(f"Query: {batch['query'][0]}, Response: {batch['response'][0]}")

    pipe_outputs = reddit_pipe(texts, **sent_kwargs)
    # print(pipe_outputs[0])
    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    ref_texts = [q + sep_token + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = reddit_pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

model.save_pretrained("./models/rlhf")

bs = 16
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
    gen_len = 256
    output = ref_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), **generation_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_ref.append(output)
    output = model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), **generation_kwargs
    ).squeeze()[-gen_len:]
    response_tensors.append(output)

#### decode responses
game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q +tokenizer.eos_token + r for q, r in zip(game_data["query"], game_data["response (before)"])]
game_data["rewards (before)"] = [output[0]["score"] for output in reddit_pipe(texts, **sent_kwargs)]

texts = [q + tokenizer.eos_token + r for q, r in zip(game_data["query"], game_data["response (after)"])]
game_data["rewards (after)"] = [output[0]["score"] for output in reddit_pipe(texts, **sent_kwargs)]

print(game_data)