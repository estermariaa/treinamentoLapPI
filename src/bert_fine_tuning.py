import torch
import pandas as pd
import numpy as np
import time, datetime, random
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder

def train_bert_fine_tuning(csv_path="df_ex1_clean.csv", epochs=4, batch_size=32):
    """
    Fine-tuning do BERT para classificação binária (ex: ham vs spam).
    """
    df_clean = pd.read_csv(csv_path)
    df_clean = df_clean.dropna(subset=['Message_clean', 'Label'])
    messages = df_clean['Message_clean'].astype(str).tolist()
    labels = df_clean['Label']

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Treinando em: {device}")
    print('Exemplo de tokenização:', tokenizer.tokenize(messages[0]))

    max_len = max(len(tokenizer.encode(m, add_special_tokens=True)) for m in messages)
    print(f"Maior sequência: {max_len} tokens")

    input_ids, attention_masks = [], []
    for message in messages:
        encoded_dict = tokenizer.encode_plus(
            message,
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

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print(f"\n======== Época {epoch_i+1} / {epochs} ========")
        t0 = time.time()
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
        print(f"  Perda média de treino: {avg_train_loss:.4f}")

        # Validação
        model.eval()
        total_eval_accuracy, total_eval_loss = 0, 0

        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(logits.detach().cpu().numpy(), b_labels.cpu().numpy())

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print(f"  Acurácia validação: {avg_val_accuracy:.4f} | Perda: {avg_val_loss:.4f}")

    print(f"\nTreinamento finalizado em {datetime.timedelta(seconds=int(time.time()-total_t0))}")
    torch.save(model.state_dict(), "bert_finetuned.pth")
    print("Modelo salvo em 'bert_finetuned.pth'.")


if __name__ == "__main__":
    train_bert_fine_tuning()
