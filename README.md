# MineCraftGPT

Implementation of course project for PKU 2023 NLPDL.

## Requirements

```bash
conda install --file requirements.txt
```
And install the trl package by
```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
```

## Datasets

We use the datasets provided by [MineDojo](https://minedojo.org/knowledge_base.html).

To get the reddit dataset, you first need to follow instructions on [PRAW](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html) to get your own reddit `client_id`, `client_secret` and `usr_agent` and fill them in `get_reddit_data` and `preprocess_reddit_data` in `utils.py`.

Then run `utils.py` to get the wiki and reddit dataset.

## Training

To train the models for wiki generation, run
```bash
python wiki_train.py
```
after changing the wandb project name and model path in it.

To train the models for reddit reply, run
```bash
python reddit_train.py
```

If you want to try RLHF on the reddit dataset, first run
```bash
python reward_model.py
```
to train the reward model, then run
```bash
python PPO.py
```
to train the RLHF model.

## Evaluation

Access to the Google Gemini model is required to run the evaluation. Please follow the instructions on [Gemini](https://ai.google.dev/docs/).

Run `python elo_rating.py` to see the result.