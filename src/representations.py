import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def tfidf_representation(train_texts, test_texts=None):
    '''
    Retorna representações TF-IDF dos textos
    '''
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        return X_train, X_test, vectorizer
    return X_train, vectorizer

def word2vec_representation(texts, vector_size=100, window=5, min_count=1, workers=-1):
    '''
    Treina um modelo Word2Vec e retorna os embeddings médios por documento'''

    sentences = [t.split() for t in texts if isinstance(t, str)]
    model = Word2Vec(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers)
    model.init_sims(replace=True)

    # Representação média por documento
    def doc_vector(doc):
        words = [w for w in doc if w in model.wv]
        if not words:
            return np.zeros(vector_size)
        return np.mean(model.wv[words], axis=0)

    X = np.array([doc_vector(s) for s in sentences])
    return X, model

def fasttext_representation(texts, vector_size=100, window=5, min_count=1, workers=-1):
    """
    Treina um modelo FastText e retorna os embeddings médios por documento.
    """
    sentences = [t.split() for t in texts if isinstance(t, str)]
    model = FastText(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers)
    model.init_sims(replace=True)

    def doc_vector(doc):
        words = [w for w in doc if w in model.wv]
        if not words:
            return np.zeros(vector_size)
        return np.mean(model.wv[words], axis=0)

    X = np.array([doc_vector(s) for s in sentences])
    return X, model

# BERT Representation (Zero-Shot)
class TextDataset(Dataset):
    # Essa classe cuida da tokenização do texto, tratamento do comprimento da sequência e fornece um pacote
    # organizado com IDs de entrada, máscaras de atêncção e rótulos para o modelo aprender 
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def bert_representation(texts, labels, batch_size=16, max_length=128, model_name='bert-base-uncased'):
    """
    Extrai representações de texto usando o BERT pré-treinado (sem fine-tuning).
    Retorna vetores CLS e rótulos codificados.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = TextDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features, labels_out = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extraindo embeddings BERT"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_vectors = outputs.last_hidden_state[:, 0, :]  # token [CLS]
            features.append(cls_vectors.cpu())
            labels_out.append(labels_b)

    X = torch.cat(features, dim=0).numpy()
    y = torch.cat(labels_out, dim=0).numpy()

    return X, y

def get_representation(method, texts, labels=None, test_texts=None, **kwargs):
    """
    Retorna a representação escolhida de acordo com o método informado.
    """
    method = method.lower()
    if method == "tfidf":
        return tfidf_representation(texts, test_texts, **kwargs)
    elif method == "word2vec":
        return word2vec_representation(texts, **kwargs)
    elif method == "fasttext":
        return fasttext_representation(texts, **kwargs)
    elif method == "bert":
        return bert_representation(texts, labels, **kwargs)
    else:
        raise ValueError(f"Método de representação desconhecido: {method}")