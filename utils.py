def get_wiki_data():
    from minedojo.data import WikiDataset
    wiki_dataset = WikiDataset(
            full=True,    # full=False for a sample subset of Wiki data (quick to download) or 
                        # full=True for the full Wiki dataset
            download=True, # download=True to automatically download data or 
                        # download=False to load data from download_dir
            download_dir="data/wiki"
                        # default: "~/.minedojo". You can also manually download data from 
                        # https://doi.org/10.5281/zenodo.6640448 and extract it to download_dir.
        ) 
    return wiki_dataset
                                            
def get_reddit_data():
    from minedojo.data import RedditDataset
    reddit_dataset = RedditDataset(
        client_id="", 
        client_secret="", 
        user_agent="",
        download=True, # download=True to automatically download data or
                       # download=False to load data from download_dir
        download_dir="data/reddit"
                       # default: "~/.minedojo". You can also manually download data from 
                       # https://doi.org/10.5281/zenodo.6641114 and put it in download_dir.
    ) 
    return reddit_dataset


def process_wiki_item(wiki_item):
    skip_words = ["For other uses", "For the achievement", "For the advancement", "Achievements", "Advancements", "See also", "Items[show]", "Issues", "Report issues", "report issues", "Twitter", "[show]","[edit]"]
    wiki = ""
    for unit in wiki_item:
        if unit['text'] == 'Gallery':
            break
        if len(unit['text']) >= 50:
            flag = False
            for word in skip_words:
                if word in unit['text']:
                    flag = True
            if flag:
                continue
            if unit['text'].startswith("1") and ("\n2" in unit['text'] or "1.1" in unit['text']):
                continue
            wiki+=unit['text']
            wiki+="\n"
    return wiki

def preprocess_wiki_data():
    from minedojo.data import WikiDataset
    from tqdm import tqdm
    import os
    import csv
    wiki_dataset = WikiDataset(
            full=True,    # full=False for a sample subset of Wiki data (quick to download) or 
                        # full=True for the full Wiki dataset
            download=False, # download=True to automatically download data or 
                        # download=False to load data from download_dir
            download_dir="data/wiki"
                        # default: "~/.minedojo". You can also manually download data from 
                        # https://doi.org/10.5281/zenodo.6640448 and extract it to download_dir.
        ) 

    # f = open("data/wiki/wiki.txt", "w")
    f = open("data/wiki/wiki.csv", "w")
    writer = csv.writer(f)
    writer.writerow(["instruction", "wiki"])
    for i in tqdm(range(len(wiki_dataset))):
        try:
            wiki_item = wiki_dataset[i]['texts']
        except:
            continue
        head = wiki_item[0]['text']
        if "Edition" in head or "Server" in head or "Launcher" in head or "List" in head:
            continue
        instruction = f"Generate wiki for {wiki_item[0]['text']}:"
        wiki_item = wiki_item[1:]
        # print(instruction)
        # exit()
        wiki = process_wiki_item(wiki_item)
        if wiki == "":
            print(wiki_item)
            continue
        wiki = f"{wiki_item[0]['text']}:{wiki}"
        writer.writerow([instruction, wiki])
    f.close()

def load_wiki_dataset():
    from datasets import load_dataset, Dataset
    import os
    if not os.path.exists("data/wiki/wiki.csv"):
        preprocess_wiki_data()
    dataset = load_dataset("csv", data_files="data/wiki/wiki.csv")
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=2022)
    print(f"Loaded {len(dataset['train'])} training samples and {len(dataset['test'])} testing samples from Wiki dataset.")
    return dataset

def preprocess_reddit_data():
    from minedojo.data import RedditDataset
    from tqdm import tqdm
    import csv
    reddit_dataset = RedditDataset(
        client_id="", 
        client_secret="", 
        user_agent="",
        download=False, # download=True to automatically download data or
                       # download=False to load data from download_dir
        download_dir="data/reddit"
                       # default: "~/.minedojo". You can also manually download data from 
                       # https://doi.org/10.5281/zenodo.6641114 and put it in download_dir.
    ) 
    f = open("data/reddit/reddit.csv", "w")
    writer = csv.writer(f)
    writer.writerow(["question_id", "upvotes_chosen", "upvotes_rejected", "question", "chosen", "rejected"])
    import time
    for i in tqdm(range(10000)):
        try:
            reddit_item = reddit_dataset[i]
        except:
            continue
        prompt = reddit_item['title']
        # metric: upvotes number, second metric: cotent length
        comments = reddit_item['comments']
        chosen = None
        reject = None
        for comment in comments:
            if len(comment['content']) >= 25:
                chosen = comment
                break
        for comment in comments[int(0.5*len(comments)):]:
            if len(comment['content']) >= 15:
                reject = comment
                break
        if chosen is None or reject is None:
            continue
        writer.writerow([reddit_item['id'], chosen['score'], reject['score'], prompt, chosen['content'], reject['content']])
    f.close()



if __name__ == "__main__":
    get_wiki_data()
    get_reddit_data()
    preprocess_wiki_data()
    preprocess_reddit_data()

