from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import pandas as pd
import torch
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU : {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification",
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False,
)


model = model.cuda()
model.eval()

model_directory = "DistilBERT-Multi-Label-Commit-Classification"

tokenizer = DistilBertTokenizer.from_pretrained(model_directory)

model = DistilBertForSequenceClassification.from_pretrained(model_directory)


with open("subject_values.json", "r") as file:
    data = json.load(file)

print("len of data at start : ", len(data))
print("len after uniqueness : ", len(list(set(data))))

data = list(set(data))

data = [s for s in data if not s == None]
print("len after removing null : ", len(data))

data = [s for s in data if s.strip()]
print("len after removing empty : ", len(data))

max_sen = 0
for sen in data:
    if max_sen < len(sen):
        max_sen = len(sen)
print(f"max_sen : {max_sen}")

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd

model_path = "DistilBERT-Multi-Label-Commit-Classification"
model = DistilBertForSequenceClassification.from_pretrained(model_path).to("cuda")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

co = 0
filter_data = []
for sen in data:
    tokens = tokenizer.tokenize(sen)
    # Count the number of words
    # Each word, even if split into subwords, counts as one word
    # unique_words = set([token if token.startswith("##") else token for token in sen])

    if len(tokens) > 512:
        co += 1
        print("outlier len", len(tokens))
    else:
        filter_data.append(sen)


print(f"no of outliers : {co}")
print(f"no of non-outliers : {len(filter_data)}")

data = filter_data


def create_batches(commits, batch_size=32):
    batches = [commits[i : i + batch_size] for i in range(0, len(commits), batch_size)]
    return batches


def predict_batches(batches):
    predictions = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batches):
            if idx % 100 == 0:
                print(idx, "de ", len(batches))
            inputs = tokenizer(
                [commit for commit in batch if commit is not None],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)
            labels = (probs > 0.5).int()
            predictions.extend(labels.cpu().numpy().tolist())
    return predictions


def labels_to_string(labels):
    label_mapping = {0: "Corrective", 1: "Adaptive", 2: "Perfective"}

    str_labels = []
    for label_list in labels:
        current_labels = [
            label_mapping[idx] for idx, val in enumerate(label_list) if val == 1
        ]
        str_labels.append(" ".join(current_labels))

    return str_labels


commit_batches = create_batches(data)
predictions = predict_batches(commit_batches)

string_predictions = labels_to_string(predictions)

df = pd.DataFrame(data, columns=["message"])
df["classification"] = string_predictions

print(df)
df.to_csv("pred.csv", index=False)
