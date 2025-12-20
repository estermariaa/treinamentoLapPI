import torch
import pandas as pd
import numpy as np
import time, datetime, random
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def train_bert_fine_tuning(texts, labels, epochs=3,batch_size=16,return_model=False):    
    """
    Fine-tuning do BERT para classificação binária (ex: ham vs spam).
    """
    texts = [str(t) for t in texts]
    encoded_labels = labels

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = min(
        256,
        max(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts)
    )

    input_ids, attention_masks = [], []
    for t in texts:
        encoded_dict = tokenizer.encode_plus(
            t,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(encoded_labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
   
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    total_t0 = time.time()

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validação
        model.eval()
        preds, true_labels = [], []
        val_loss = 0

        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            logits = outputs.logits
            loss = outputs.loss
            val_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

        avg_val_loss = val_loss / len(validation_dataloader)
        accuracy = accuracy_score(true_labels, preds)
        f1_macro = f1_score(true_labels, preds, average="macro")

        elapsed = datetime.timedelta(seconds=int(time.time() - total_t0))
    
    if return_model:
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "val_loss": avg_val_loss,
            "elapsed": str(elapsed),
            "model": model,
            "tokenizer": tokenizer,
            "label_encoder": le
        }

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "val_loss": avg_val_loss,
        "elapsed": str(elapsed)
    }

if __name__ == "__main__":
    train_bert_fine_tuning()
