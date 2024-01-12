from transformers import (
    pipeline
)
import os
import json

import google.generativeai as genai

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
evaluator = genai.GenerativeModel('gemini-pro')

model_names = ['gpt2', 'gpt2_wiki', 'reddit_wiki', 'gemini']

gpt2 = pipeline('text-generation',model='./models/gpt2', tokenizer='./models/gpt2')
wiki = pipeline('text-generation',model='./models/gpt2_wiki', tokenizer='./models/gpt2')
reddit2wiki = pipeline('text-generation',model='./models/reddit_wiki', tokenizer='./models/gpt2')
gemini = genai.GenerativeModel('gemini-pro')

models = [gpt2, wiki, reddit2wiki, gemini]

ratings = {
    'gpt2': 1000,
    'gpt2_wiki': 1000,
    'reddit_wiki': 1000,
    'gemini': 1000
}


evaluate_prompt = """
    Rate two model for Minecraft wiki generation from the perspective of Minecraft knowledge quality, readability, and fluency.
    The topic is:
"""

topics = ['Apple', 'Cooked Mutton', 'Diamond', 'Iron Ingot', 'Creeper', 'Villager', 'End Portal', 'Crafting Table']

with open('./data/wiki/topic.json', 'r') as f:
    references = json.load(f)

def get_output(name, model, topic):
    if name != 'gemini':
        prompt = f"Generate wiki for {topic}:"
        return model(prompt, max_length=512)[0]['generated_text']
    else:
        prompt = f"Generate Minecraft wiki for {topic}"
        return model.generate_content(prompt).text

import random
for topic in topics:
    for name, model in zip(model_names, models):
        # shuffle the index
        index = list(range(len(models)))
        random.shuffle(index)
        for idx in index:
            opponent_name = model_names[idx]
            opponent = models[idx]
            if name == opponent_name:
                continue
            
            print(f"Running {name} v.s. {opponent_name} on topic {topic}")
            reference = references[topic][:512]
            prompt = evaluate_prompt + f" {topic}.\nHere is the reference wiki:\n{reference}\n"
            prompt += "The first model's output is: \n"
            prompt += get_output(name, model, topic)
            prompt += "\nThe second model's output is: \n"
            prompt += get_output(opponent_name, opponent, topic)
            prompt += "\nPlease output whose result is better, if the first is better, output '1', else output '2'; do not output anything else\n"

            try:
                result = evaluator.generate_content(prompt).text
            except:
                print(evaluator.generate_content(prompt).prompt_feedback)
                continue         
            print(result)
            e1 = 1/(1+10**((ratings[opponent_name]-ratings[name])/400))
            e2 = 1/(1+10**((ratings[name]-ratings[opponent_name])/400))            

            if '1' in result:
                ratings[name] += 32*(1-e1)
                ratings[opponent_name] += 32*(0-e2)
            elif result == '2':
                ratings[name] += 32*(0-e1)
                ratings[opponent_name] += 32*(1-e2)
            print(f"New ratings: {name}: {ratings[name]}, {opponent_name}: {ratings[opponent_name]}")
print(ratings)


model_names = ['gpt2', 'gpt2_reddit', 'wiki_reddit', 'gemini']
gpt2 = pipeline('text-generation',model='./models/gpt2', tokenizer='./models/gpt2')
reddit = pipeline('text-generation',model='./models/gpt2_reddit', tokenizer='./models/gpt2')
wiki2reddit = pipeline('text-generation',model='./models/wiki_reddit', tokenizer='./models/gpt2')
gemini = genai.GenerativeModel('gemini-pro')

models = [gpt2, reddit, wiki2reddit, gemini]
ratings = {
    'gpt2': 1000,
    'gpt2_reddit': 1000,
    'wiki_reddit': 1000,
    'gemini': 1000
}

evaluate_prompt = """
    Rate two model for Minecraft Reddit reply from the perspective of Minecraft knowledge quality, Reddit-like, and fluency.
    The topic is:
"""

topics = [
    'Some redstone tips for beginners.',
    'How to survive in Minecraft?',
    'What can be eaten in Minecraft?',
    'How to build a house in Minecraft?'
]

def get_output(name, model, topic):
    if name != 'gemini':
        prompt = f"Reply to: {topic}<|endoftext|>"
        return model(prompt, max_length=512)[0]['generated_text']
    else:
        prompt = f"Reply to '{topic}' in Reddit/Minecraft"
        return model.generate_content(prompt).text


    
import random
for topic in topics:
    for name, model in zip(model_names, models):
        # shuffle the index
        index = list(range(len(models)))
        random.shuffle(index)
        for idx in index:
            opponent_name = model_names[idx]
            opponent = models[idx]
            if name == opponent_name:
                continue
            
            print(f"Running {name} v.s. {opponent_name} on topic {topic}")
            prompt = evaluate_prompt + f" '{topic}'.\n"
            prompt += "The first model's output is: \n"
            prompt += get_output(name, model, topic)
            prompt += "\nThe second model's output is: \n"
            prompt += get_output(opponent_name, opponent, topic)
            prompt += "\nPlease output whose result is better, if the first is better, output '1', else output '2'; do not output anything else\n"
            try:
                result = evaluator.generate_content(prompt).text
            except:
                print(evaluator.generate_content(prompt).prompt_feedback)
                continue
            print(result)
            e1 = 1/(1+10**((ratings[opponent_name]-ratings[name])/400))
            e2 = 1/(1+10**((ratings[name]-ratings[opponent_name])/400))            

            if '1' in result:
                ratings[name] += 32*(1-e1)
                ratings[opponent_name] += 32*(0-e2)
            elif result == '2':
                ratings[name] += 32*(0-e1)
                ratings[opponent_name] += 32*(1-e2)
            print(f"New ratings: {name}: {ratings[name]}, {opponent_name}: {ratings[opponent_name]}")
print(ratings)

topic = "How to build a house in Minecraft?"

gpt2_ouput = get_output('gpt2', gpt2, topic)
reddit_output = get_output('gpt2_reddit', reddit, topic)
wiki2reddit_output = get_output('wiki_reddit', wiki2reddit, topic)

print(gpt2_ouput)
print(reddit_output)
print(wiki2reddit_output)