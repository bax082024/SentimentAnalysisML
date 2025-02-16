import os
import pandas as pd

def load_reviews_from_folder(folder, label):
    reviews = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
            reviews.append((file.read(), label))
    return reviews

pos_reviews = load_reviews_from_folder('C:/HjemmeKode/IMDB Dataset/aclImdb/train/pos', 'Positive')
neg_reviews = load_reviews_from_folder('C:/HjemmeKode/IMDB Dataset/aclImdb/train/neg', 'Negative')

all_reviews = pos_reviews + neg_reviews
df = pd.DataFrame(all_reviews, columns=['Text', 'Sentiment'])

df.to_csv('Data/imdb_reviews.csv', index=False)

print("Dataset combined and saved successfully!")
