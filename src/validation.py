"""
evaluation.py
Contém funções para:
- Cálculo de intervalo de confiança
- Teste t pareado com correção de Bonferroni
- Avaliação via Holdout e Stratified K-Fold
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

def calcular_intervalo_confianca(valores):
    media = np.mean(valores)
    desvio_padrao = np.std(valores)
    n = len(valores)

    t_critico = stats.t.ppf(0.975, df=n-1) # t Student (95% de confiança)

    erro_padrao = desvio_padrao/np.sqrt(n)

    intervalo_confianca = t_critico * erro_padrao

    return media, desvio_padrao, intervalo_confianca

def comparar_modelos_teste_t(resultados):
    modelos = list(resultados.keys())
    acuracias = [resultados[m] for m in modelos]

    comparacoes = []

    print("\n=== Teste T pareado entre modelos ===")

    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            t_stat, valor_p = stats.ttest_ind(acuracias[i], acuracias[j])
            comparacoes.append(((modelos[i], modelos[j]), valor_p))
            print(f"{modelos[i]} vs {modelos[j]}: p = {valor_p:.5f}")

    # Correção de Bonferroni
    k = len(comparacoes)
    print("\n--- Correção de Bonferroni ---")
    for (m1, m2), valor_p in comparacoes:
        p_corrigido = min(valor_p * k, 1.0)
        status = "Diferentes" if p_corrigido < 0.05 else "Equivalentes"
        print(f"{m1} vs {m2} | p_corrigido = {p_corrigido:.5f} → {status}")

def holdout(modelo_cls, X, y, rep, test_size=0.2, best_params=None, vectorizer=None):
    print("\n--- Holdout ---")
    acuracias = []
    f1_scores = []

    for i in range(rep):
        print(f"\nRepetição {i + 1}/{rep}:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        # Caso tenha um vetor TF-IDF
        if vectorizer:
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

        if best_params is not None:
            modelo = modelo_cls(**best_params)
        else:
            modelo = modelo_cls()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        acuracias.append(acc)
        f1_scores.append(f1)

        print(f"  Acurácia: {acc:.4f} | F1 (macro): {f1:.4f}")

    acuracia_media, acuracia_desvio, acuracia_ic = calcular_intervalo_confianca(acuracias)
    f1_media, f1_desvio, f1_ic = calcular_intervalo_confianca(f1_scores)

    print("\nMÉDIA FINAL HOLDOUT:")
    print(f"  Acurácia média: {acuracia_media:.4f} ± {acuracia_ic:.4f}")
    print(f"  F1 (macro) médio: {f1_media:.4f} ± {f1_ic:.4f}")

    return acuracias, f1_scores

def stratified_k_fold(model_class, X, y, best_params=None, k=10, vectorizer=None):
    print("\n--- Stratified K-Fold ---")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acuracias = []
    f1_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {i + 1}/{k}:")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if vectorizer:
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

        if best_params:
                model = model_class(**best_params)
        else:
            model = model_class()
            
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            acuracias.append(acc)
            f1_scores.append(f1)

            print(f"  Acurácia: {acc:.4f} | F1: {f1:.4f}")
        except Exception as e:
            print(f"Erro ao treinar modelo: {e}")
            continue

    if not acuracias:
        print("\nNenhum fold executado com sucesso.")
        return [], []

    # Cálculo dos intervalos de confiança
    acuracia_media, acuracia_desvio, acuracia_ic = calcular_intervalo_confianca(acuracias)
    f1_media, f1_desvio, f1_ic = calcular_intervalo_confianca(f1_scores)

    print("\nMÉDIA FINAL K-FOLD:")
    print(f"  Acurácia média: {acuracia_media:.4f} ± {acuracia_ic:.4f}")
    print(f"  F1 (macro) médio: {f1_media:.4f} ± {f1_ic:.4f}")

    return acuracias, f1_scores
