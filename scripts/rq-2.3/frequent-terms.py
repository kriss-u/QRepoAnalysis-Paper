import re
from concurrent.futures import ProcessPoolExecutor

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")


def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    lemmatized_words = [token.lemma_ for token in doc]
    return " ".join(lemmatized_words)


def clean_text(text):
    # Combined regex pattern for all specified removals
    pattern = r"""                                                                                                                                                                                                                                            
        !\[.*?\]\(.*?\)        # Markdown Images                                                                                                                                                                                                              
        | ```[\s\S]*?```       # Multiline Code Blocks                                                                                                                                                                                                        
        | https?://[^\s]+      # URLs                                                                                                                                                                                                                         
        | \b\d+\.\d+\b         # Float numbers                                                                                                                                                                                                                
        | \b\d+\b              # Integer numbers                                                                                                                                                                                                              
        | <[^>]+>.*?</[^>]+>   # Tags and everything in between                                                                                                                                                                                                                                                                                                                                                                                          
        | \S*[!@#$%^&*()_+\-=\[\]{};:'",.<>?/\\|`~]\S* # Words with special characters                                                                                                                                                                        
        """
    # Substitute all matches with an empty string
    result = re.sub(pattern, "", text, flags=re.VERBOSE)
    # Remove extra whitespace left after removals
    result = re.sub(r"\s+", " ", result).strip()
    return result


df = pd.read_json("all_titles.json")
print("original_df : ", len(df))

# Remove rows where 'text' column is NaN
df = df.dropna(subset=[0])
print("original_df removed null : ", len(df))
sentences = df[0].tolist()

rev = []
for sen in sentences:
    csen = clean_text(sen)
    rev.append(csen)

sentences = rev
print("cleaned up sentences : ", len(sentences))

sentences = list(set(sentences))
print("unique :", len(sentences))

# Load the English model
nlp = spacy.load("en_core_web_sm")
lem = []
with ProcessPoolExecutor() as executor:
    # Map the lemmatize_sentence function to the sentences
    lem = list(executor.map(lemmatize_sentence, sentences))

sentences = lem
print(len(sentences))
torch.save(sentences, "sentences.pt")


sentences = torch.load("sentences.pt")
print(len(sentences))
sentences = [item for item in sentences if item]
sentences = list(set(sentences))
print(len(sentences))
print()
print()

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_vector = tfidf_vectorizer.fit_transform(sentences)


tfidf_df = pd.DataFrame(
    tfidf_vector.toarray(),
    index=np.arange(len(sentences)),
    columns=tfidf_vectorizer.get_feature_names_out(),
)
print("fitting done")


tfidf_df.loc["00_Document Frequency"] = tfidf_df.sum(axis=0)

scores_row = tfidf_df.loc["00_Document Frequency"]

top_10_scores = scores_row.nlargest(400)

for column, score in top_10_scores.items():
    print(f"{column}: {score}")
