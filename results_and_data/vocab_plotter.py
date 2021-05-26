"""
This script creates the vocab plot curves (`vocab_curves.png`) from vocabulary samples from the data in pickle form
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open('amazon_sample.pickle', 'rb') as f:
    results_amazon = pickle.load(f)

with open('apnews_sample.pickle', 'rb') as f:
    results_ap = pickle.load(f)

# vocab, total word count
df_amazon = pd.DataFrame(results_amazon)
df_ap = pd.DataFrame(results_ap)
vocab_name = "Vocabulary Size (1e7)"
word_name = "Number of Total Words (1e8)"
df_amazon.columns = [vocab_name, word_name]
df_amazon["Dataset"] = "Amazon"

df_ap.columns = [vocab_name, word_name]
df_ap["Dataset"] = "AP News"

full_df = pd.concat([df_amazon, df_ap], axis=0)
full_df[vocab_name] /= 1000000 # scale
ax = sns.lineplot(data=full_df, x=word_name, y=vocab_name, hue="Dataset")
plt.tight_layout()
plt.savefig("vocab_curves.png")

