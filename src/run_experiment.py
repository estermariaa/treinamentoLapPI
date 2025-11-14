import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from data_loader import load_data
from preprocess import preprocess_text
from representations import get_representation
from classification_models import arvore_de_decisao, knn, svm_modelo
from validation import stratified_k_fold, holdout

"""
Pipeline completo para experimentos de classificação de textos em inglês.
O script permite testar múltiplas representações (TF-IDF, Word2Vec, FastText)
e múltiplos modelos clássicos (Árvore, KNN, SVM), salvando os resultados
automaticamente em arquivos CSV e TXT.

⚠️ Observação:
O Fine-Tuning do BERT é computacionalmente pesado e deve ser executado
em script separado (veja bert_fine_tuning.py).

"""

RESULTS_DIR = "results"
LOG_PATH = f"{RESULTS_DIR}/experiment_log.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_FUNCS = {
    "DecisionTree": arvore_de_decisao,
    "KNN": knn,
    "SVM": svm_modelo,
}

def save_results_csv(results_df, representation):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/results_{representation}_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"Resultados salvos em: {filename}")

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {msg}\n")
    print(msg)

def run_experiment(dataset_path, text_col, label_col,
                   representations=["tfidf", "word2vec", "fasttext"],
                   models=["DecisionTree", "KNN", "SVM"], rodar_finetuning="no"):
    """
    Executa experimentos de classificação de texto combinando múltiplas
    representações e modelos clássicos.

    Args:
        dataset_path (str): Caminho do arquivo CSV/TSV.
        representations (list[str]): Representações a testar.
        models (list[str]): Modelos a testar.
    """

    open(LOG_PATH, "w").close()
    log("========== INICIANDO EXPERIMENTO ==========")

    start_time = time.time()

    # --- 1. Carregar dataset ---
    texts, labels = load_data(dataset_path, label_col, text_col)

    if not texts:
        log("ERRO: Falha ao carregar o dataset.")
        return
    
    log(f"Dataset carregado: {dataset_path}")
    log(f"Total de linhas carregadas: {len(texts)}")
    log(f"Coluna texto: {text_col} | Coluna label: {label_col}")

    # --- 2. Pré-processar textos ---
    log("Pré-processando textos...")
    texts_clean = [preprocess_text(t) for t in texts]
    log(f"Total de textos preprocessados: {len(texts_clean)}")

    # --- 3. Codificar rótulos ---
    le = LabelEncoder()
    y = le.fit_transform(labels)
    log(f"Classes encontradas: {list(le.classes_)}")

    all_results = []

    # Loop principal: combina todas as representações e modelos
    for rep in representations:
        log(f"\n================ REPRESENTAÇÃO: {rep.upper()} ================")

        # --- 4. Gerar representação ---
        try:
            start_rep = time.time()
            X, vectorizer = get_representation(rep, texts_clean)
            rep_time = time.time() - start_time

            log(f"Representação '{rep}' gerada.")
            log(f"Dimensão da matriz: {X.shape}")
            log(f"Tempo para gerar representação: {rep_time:.2f}s")       
            
        except Exception as e:
            log(f"ERRO ao gerar representação {rep}: {e}")
            continue

        # --- 5. Rodar modelos ---
        for model_name in models:
            
            model_func = MODEL_FUNCS[model_name]
            log(f"\n>>> MODELO: {model_name}")

            try:
                model_start = time.time()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=0
                )

                log("Executando GridSearchCV...")

                result_grid = model_func(X_train, y_train, X_test, y_test)
                best_params = result_grid["melhores_parametros"]
                model_class = result_grid["model_class"]

                log(f"Melhores parâmetros encontrados: {best_params}")

                log("Executando Stratified K-Fold...")
                acc_kf, f1_kf = stratified_k_fold(model_class, X, y, best_params)
                mean_acc_kf = np.mean(acc_kf)
                mean_f1_kf = np.mean(f1_kf)

                log(f"KFold - Acc média: {mean_acc_kf:.4f} | F1 média: {mean_f1_kf:.4f}")

                log("Executando Holdout (5 repetições)...")
                ho_acc, ho_f1 = holdout(model_class, X, y, rep=5, best_params=best_params)
                mean_acc_ho = np.mean(ho_acc)
                mean_f1_ho = np.mean(ho_f1)

                log(f"Holdout - Acc média: {mean_acc_ho:.4f} | F1 média: {mean_f1_ho:.4f}")

                all_results.append({
                    "Representation": rep,
                    "Model": model_name,
                    "BestParams": str(best_params),
                    "KFold_Accuracy": mean_acc_kf,
                    "KFold_F1_Score": mean_f1_kf,
                    "Holdout_Accuracy": mean_acc_ho,
                    "Holdout_F1_Score": mean_f1_ho,
                })
                
                log(f"Modelo {model_name} finalizado em {(time.time() - model_start):.2f}s")

            except Exception as e:
                log(f"ERRO ao treinar modelo {model_name}: {e}")
                continue


    # Salvar resultados
    df = pd.DataFrame(all_results)

    if len(df) > 0:
        save_results_csv(df, "ALL")
    else:
        log("Nenhum resultado foi gerado.")

    log("\nExperimento concluído!")
    log(f"Tempo total: {time.time() - start_time:.2f}s\n")


# Execução direta
if __name__ == "__main__":

    run_experiment(
    dataset_path="../data/smsspamcollection/SMSSpamCollection",
    label_col="Label",
    text_col="Message",
    representations=["tfidf", "word2vec", "fasttext"],
    models=["DecisionTree", "KNN", "SVM"],
    rodar_finetuning="no"
)

    print("\nDica: Para Fine-Tuning BERT, use o script separado 'bert_fine_tuning.py',")
    print("pois ele requer GPU com boa memória.")