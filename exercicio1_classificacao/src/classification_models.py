"""
Contém os modelos de classificação utilizados no experimento:
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Cada função:
    - Recebe X_train, y_train, X_test, y_test
    - Faz busca em grade (GridSearchCV)
    - Retorna o modelo treinado, suas previsões e métricas
"""
import numpy as np
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def arvore_de_decisao(X_train, y_train, X_test, y_test):
    parametros = {
        'criterion': ['gini', 'entropy'], # Função usada para medir a qualidade de uma divisão
        'max_depth': [None, 10, 20], # Profundidade máxima 
        'min_samples_split': [2, 5, 10], # Mínimo de amostras exigido para dividir um nó
        'min_samples_leaf': [1, 2, 4] # Mínimo de amostras exigido em um nó folha
    }

    clf = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=parametros, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    melhor_clf = grid_search.best_estimator_ 
    y_pred = melhor_clf.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print("\n===== Árvore de Decisão =====")
    print("Melhores parâmetros:", grid_search.best_params_)
    print(f"Acurácia: {acuracia:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")

    return {
        "modelo": melhor_clf,
        "melhores_parametros": grid_search.best_params_,
        "acuracia": acuracia,
        "f1_macro": f1_macro,
        "y_pred": y_pred,
        "model_class": tree.DecisionTreeClassifier
    }

def knn(X_train, y_train, X_test, y_test):
    parametros = {'n_neighbors': [1, 3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']}

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=parametros, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train) 

    melhor_knn = grid_search.best_estimator_
    melhor_y_pred = melhor_knn.predict(X_test)

    acuracia = accuracy_score(y_test, melhor_y_pred)
    f1_macro = f1_score(y_test, melhor_y_pred, average='macro')

    print("\n===== KNN =====")
    print("Melhores parâmetros:", grid_search.best_params_)
    print(f"Acurácia: {acuracia:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")

    return {
        "modelo": melhor_knn,
        "melhores_parametros": grid_search.best_params_,
        "acuracia": acuracia,
        "f1_macro": f1_macro,
        "y_pred": melhor_y_pred,
        "model_class": KNeighborsClassifier
    }

def svm_modelo(X_train, y_train, X_test, y_test):
    parametros = {
        'C': [1, 10],  # Parâmetro C
        'gamma': [0.1, 'auto', 'scale'],  # Parâmetro gamma
        'kernel': ['rbf', 'linear']  # Tipos de kernel
    }
    
    svc = svm.SVC()
    grid_search = GridSearchCV(estimator=svc, param_grid=parametros, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    melhor_svm = grid_search.best_estimator_
    y_pred = melhor_svm.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print("\n===== SVM =====")
    print("Melhores parâmetros:", grid_search.best_params_)
    print(f"Acurácia: {acuracia:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")

    return {
        "modelo": melhor_svm,
        "melhores_parametros": grid_search.best_params_,
        "acuracia": acuracia,
        "f1_macro": f1_macro,
        "y_pred": y_pred,
        "model_class": svm.SVC
    }



