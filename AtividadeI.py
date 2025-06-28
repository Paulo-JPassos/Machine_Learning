#Para apresentar no streamlit:p
import streamlit as st

# Título do App
st.title("Comparação de Modelos de Classificação - Machine Learning Agrário")

# Exibição do código original
st.subheader("Código-Fonte")

code = '''
#Bibliotecas de manipulação e visualização de dados
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Bibliotecas de aprendizado de máquina
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# Carregamento do dataset
# Caminho relativo do arquivo (pode ser adaptado conforme necessidade)

caminho_do_arquivo = 'dataset_agrario_pratica.csv'  # Substitua pelo nome do seu arquivo
df = pd.read_csv(caminho_do_arquivo, sep=',')  # Leitura com separador ponto e vírgula
print(df.head())


#Preparando os dados para modelagem: #Nesta etapa, separamos as variáveis preditoras (X) da variável alvo (y), 
# que indica se o aluno foi aprovado ou reprovado. Em seguida, os dados são divididos em conjuntos de treino e teste.

X = df.drop(columns=['Status de Produção']) #Porque drop?
y = df['Status de Produção']

# Codificação da variável alvo
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 1 Alta, 0 para Baixa


#ATENÇÃO NESTE COMANDO!!!
# Codificação das variáveis categóricas em X
X = pd.get_dummies(X, drop_first=True)  # Usa OneHotEncoding para variáveis categóricas

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Ajusta e transforma os dados de treino
X_test = scaler.transform(X_test)  # Apenas transforma os dados de teste

#Treinando os modelos
#Treinamos dois modelos diferentes: Random Forest e SVM (Support Vector Machine).

rf_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Ajuste dos modelos
rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)


#Avaliando os modelos
#Aqui utilizamos métricas como precisão, recall e f1-score, além da matriz de confusão, para avaliar o desempenho dos modelos.

y_pred_rf = rf_clf.predict(X_test)
y_pred_svm = svm_clf.predict(X_test)

print('Random Forest Metrics:')
print(classification_report(y_test, y_pred_rf))
print('\nSVM Metrics:')
print(classification_report(y_test, y_pred_svm))

#Comparação de Modelos
#Treinamos e avaliamos três modelos de classificação: Regressão Logística, Random Forest e Gradient Boosting.

# Inicializar os modelos
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Treinar e avaliar cada modelo
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {'Acurácia': accuracy, 'Relatório': report}

# Exibir os resultados
for name, metrics in results.items():
    print(f"Modelo: {name}")
    print(f"Acurácia: {metrics['Acurácia']:.2f}")
    print(classification_report(y_test, models[name].predict(X_test)))
'''
st.code(code, language='python')

# Conclusão
st.subheader("Conclusão")

st.markdown("""
A ideia aqui foi identificar o modelo de aprendizado de máquina mais eficaz para classificar dados no contexto agrícola. As ferramentas escolhidas foram selecionadas com base em critérios técnicos e práticos, conforme detalhado a seguir:
### Regressão Logística
Utilizada como **modelo de referência (baseline)**. É simples, interpretável e eficiente para identificar relações lineares entre as variáveis e a classe de saída.

### Random Forest
Modelo de aprendizado de máquina baseado em múltiplas árvores de decisão. É **robusto a overfitting**, trabalha bem com **variáveis categóricas**, e entrega resultados estáveis mesmo sem ajustes finos.

### Support Vector Machine (SVM)
Adequado para situações onde há **separação clara entre as classes**. Mostra bons resultados com datasets de **alta dimensionalidade**, embora seja mais sensível a ajustes de escala e parâmetros.

### Gradient Boosting
Apresenta alto desempenho em tarefas de classificação complexas. Constrói **modelos sequenciais otimizados**, corrigindo erros iterativamente, o que frequentemente resulta na **maior acurácia geral** — embora exija maior poder computacional e ajuste cuidadoso de hiperparâmetros.

A escolha do modelo ideal vai depender da contexto da sua aplicação. como me basiei nos exemplos passados pelo professor, acabei utilizando todos os modelos apresentados em sala.

""")