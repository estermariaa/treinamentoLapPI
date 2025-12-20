from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def zero_shot(df, text_col="Message_clean", label_col="Label", model_name="facebook/bart-large-mnli"):
    """
    Executa classificação Zero-Shot usando BART ou outro modelo NLI compatível.
    
    Args:
        df (pd.DataFrame): DataFrame contendo colunas de texto e rótulo.
        text_col (str): Nome da coluna de textos.
        label_col (str): Nome da coluna de rótulos.
        model_name (str): Nome do modelo HuggingFace (default: facebook/bart-large-mnli).
    """

    print(f"Carregando modelo {model_name} para zero-shot classification...")

    texts = df[text_col].astype(str).fillna("").tolist()
    labels_true = df[label_col].tolist()

    candidate_labels = ["ham", "spam"]
    classifier = pipeline("zero-shot-classification", model=model_name)

    predictions = []
    for text in texts:
        result = classifier(text, candidate_labels)
        predicted_label = result["labels"][0]
        predictions.append(predicted_label)

    acc = accuracy_score(labels_true, predictions)
    print(f"\nAcurácia Zero-Shot: {acc:.4f}")
    print("\nRelatório de Classificação:\n", classification_report(labels_true, predictions))

    return predictions, acc


if __name__ == "__main__":
    df_clean = pd.read_csv("df_ex1_clean.csv")
    zero_shot(df_clean)
