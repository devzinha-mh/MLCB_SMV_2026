# ============================================================
# LAB 03 - AULA 02 (MLCB): Preencha os blocos TODO
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset de Suporte Técnico
dados_tech = {
    'mensagem': [
        'Esqueci minha senha de acesso', 'Não consigo entrar no sistema', 'Como redefinir minha senha?',
        'A internet esta muito lenta', 'Sem conexao de rede no escritorio', 'Minha conexao caindo toda hora',
        'Impressora nao esta funcionando', 'Nao consigo imprimir documentos', 'Impressora travada com papel'
    ],
    'intencao': [
        'reset_senha', 'reset_senha', 'reset_senha',
        'problema_conexao', 'problema_conexao', 'problema_conexao',
        'suporte_impressora', 'suporte_impressora', 'suporte_impressora'
    ]
}

df3 = pd.DataFrame(dados_tech)

# TODO 1: Separe o dataset em X (coluna 'mensagem') e y (coluna 'intencao')
X = None
y = None

# TODO 2: Realize a divisão em treino (70%) e teste (30%) com random_state=42
# X_train, X_test, y_train, y_test = ...

# TODO 3: Instancie o CountVectorizer e ajuste/transforme os dados de treino e teste
# vectorizer = ...
# X_train_vec = ...
# X_test_vec = ...

# TODO 4: Instancie o DecisionTreeClassifier e treine o modelo com .fit()
# modelo_arvore = ...
# modelo_arvore.fit(...)

# TODO 5: Gere as predições para o X_test_vec e exiba a acurácia
# predicoes = modelo_arvore.predict(...)
# acuracia = accuracy_score(...)
# print(f"Acurácia do Modelo: {acuracia * 100:.2f}%")

