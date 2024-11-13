from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import csv
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

learning_rate = 2e-5
epochs = 3
batch_size = 2



class MyDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=512):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize the pair of sentences (with padding and truncation)
        sentence1, sentence2 = self.data[idx]
        encoding = self.tokenizer(
            sentence1, sentence2,
            padding='max_length',  # Pad to max_length
            truncation=True,  # Truncate if necessary
            max_length=self.max_length,  # Set max length to avoid variable sequence lengths
            return_tensors="pt"
        )

        # Extract the input_ids and attention_mask from the encoding
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove the batch dimension

        # Get the label for the sentence pair
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return input_ids, attention_mask, label



def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.squeeze()
    embedding2 = embedding2.squeeze()

    cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    return cos_sim.item()


def embed(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    pooled_output = outputs.pooler_output
    return pooled_output


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    i = 0
    for batch in dataloader:
        i += 1
        print(f"\r{i}' / '{len(dataloader)}", end="", flush=True)
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)


        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, d_dl, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    i = 0
    with torch.no_grad():
        for batch in d_dl:
            i += 1
            print(f"\r{i}' / '{len(d_dl)}", end="", flush=True)

            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predicted labels
            predictions = torch.argmax(logits, dim=-1)
            print("Pred: " + str(predictions) + " Real: " + str(labels) )
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

language_pairs = [('en', 'es'), ('en', 'de'), ('en', 'fr'), ('en', 'ja'),
                  ('en', 'ko'), ('en', 'zh')]


for pair in language_pairs:
    with open(pair[0]+ '-' + pair[1] + '-' + 'train' + '.tsv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = []
        ls = []
        for line in reader:
            ls.append(line[3])
            a = line[1]
            b = line[2]

            data.append((a,b))

        dataset = MyDataset(data, ls, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with open(pair[0]+ '-' + pair[1] + '-' + 'dev_2k' + '.tsv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = []
        ls = []
        for line in reader:
            ls.append(line[3])
            a = line[1]
            b = line[2]

            data.append((a,b))

        dev_dataset = MyDataset(data, ls, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)




    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model for a few epochs
    for epoch in range(epochs):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # Optionally, evaluate the model on the validation set or training set
    accuracy = evaluate(model, dev_dataloader, device)
    print(f"Accuracy for language pair {pair}: {accuracy}")

