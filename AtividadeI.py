# Para apresentar no Streamlit:
import streamlit as st

# Título do App
st.title("Comparação de Modelos de Classificação - Machine Learning Agrário")

st.subheader("Visualização:")

# Bibliotecas de manipulação e visualização de dados
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Bibliotecas de aprendizado de máquina
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Carregamento do dataset
caminho_do_arquivo = 'dataset_agrario_pratica.csv'
df = pd.read_csv(caminho_do_arquivo)

# Exibir primeiras linhas do dataset
st.write("Amostra do Dataset:")
st.dataframe(df.head())

# Separação de variáveis
X = df.drop(columns=['Status de Produção'])
y = df['Status de Produção']

# Codificação da variável alvo
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Codificação de variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinamento dos modelos
rf_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# Avaliação dos modelos
y_pred_rf = rf_clf.predict(X_test)
y_pred_svm = svm_clf.predict(X_test)

st.write("**Métricas - Random Forest**")
st.text(classification_report(y_test, y_pred_rf))

st.write("**Métricas - SVM**")
st.text(classification_report(y_test, y_pred_svm))

# Comparação com outros modelos
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {'Acurácia': accuracy, 'Relatório': report}

# Exibição dos resultados comparativos
st.subheader("Comparação de Modelos")
for name, metrics in results.items():
    st.markdown(f"**Modelo: {name}**")
    st.write(f"Acurácia: {metrics['Acurácia']:.2f}")
    st.text(classification_report(y_test, models[name].predict(X_test)))

# Exibir o próprio código-fonte
st.subheader("Código-Fonte")
with open(__file__, "r", encoding="utf-8") as f:
    codigo_fonte = f.read()
st.code(codigo_fonte, language='python')

# Conclusão
st.subheader("Conclusão")
st.markdown("""
A ideia aqui foi identificar o modelo de aprendizado de máquina mais eficaz para classificar dados no contexto agrícola. As ferramentas escolhidas foram selecionadas com base em critérios técnicos e práticos.

A escolha do modelo ideal depende do contexto da aplicação. Neste experimento, foram utilizados todos os modelos apresentados em sala, para análise comparativa.
""")
