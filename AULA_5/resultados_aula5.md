============
EXERCICIO 1
============
import re
import nltk
from nltk.corpus import stopwords
import spacy

nltk.download('stopwords', quiet=True)
stop_words_pt = set(stopwords.words('portuguese'))

# Baixar o modelo de linguagem do spacy para português
!python -m spacy download pt_core_news_sm

# Carregar o modelo morfológico do Spacy em Português
nlp = spacy.load("pt_core_news_sm")

def limpar_e_lemmatizar(texto):
    """
    Função para tratar texto bruto:
    1. Converte para minúsculas
    2. Remove caracteres especiais, pontuações e números via Regex
    3. Remove Stop Words e realiza Lemmatization via Spacy
    """
    # TODO 1: Converter para minúsculas
    texto_limpo = texto.lower()

    # TODO 2: Remover tudo que não for letra ou espaço usando re.sub
    texto_limpo = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', texto_limpo)

    # Processamento com Spacy
    doc = nlp(texto_limpo)

    # TODO 3: Extrair o lema (token.lemma_) ignorando stop words e espaços vazios
    tokens_filtrados = [
        token.lemma_ for token in doc
        if token.text not in stop_words_pt and not token.is_space and len(token.text) > 1
    ]

    return " ".join(tokens_filtrados)

# === TESTE DO EXERCÍCIO 1 ===
frase_teste = "Gostaria de saber se vocês estão DEVOLVENDO os valores das mesas compradas!!!"
print("Frase Original:", frase_teste)
print("Frase Limpa & Lemmatizada:", limpar_e_lemmatizar(frase_teste))

---------- RESULTADO EXERCICIO 1 -------------
Frase Original: Gostaria de saber se vocês estão DEVOLVENDO os valores das mesas compradas!!!
Frase Limpa & Lemmatizada: gostar saber devolver valor meso comprada




============
EXERCICIO 2 
============
!pip install gensim
import numpy as np
import pandas as pd
import gensim.downloader as api

print("Carregando modelo de Embeddings FastText (Gensim)...")
# Utilizando o modelo pré-treinado do Gensim em português ou substituto equivalente
fasttext_model = api.load("glove-wiki-gigaword-50") # Exemplo leve de 50 dimensões para testes em aula

def obter_vetor_frase(frase, model):
    """
    Calcula a média dos vetores das palavras de uma frase (Mean Pooling).
    """
    palavras = frase.split()
    vetores = []

    for palavra in palavras:
        if palavra in model:
            # TODO 1: Adicionar o vetor da palavra na lista de vetores
            vetores.append(model[palavra])

    if len(vetores) == 0:
        # Se nenhuma palavra estiver no vocabulário, retorna vetor de zeros
        return np.zeros(model.vector_size)

    # TODO 2: Retornar a média ao longo do eixo 0 (np.mean)
    vetor_medio = np.mean(vetores, axis=0)
    return vetor_medio

# === TESTE DO EXERCÍCIO 2 ===
df = pd.read_csv("sac_moveis_ac2.csv")
X_vetores = np.array([obter_vetor_frase(msg, fasttext_model) for msg in df['mensagem']])
print("Formato da Matriz de Vetores Densos (Exemplos, Dimensões):", X_vetores.shape)

---------- RESULTADO EXERCICIO 2 -------------
Carregando modelo de Embeddings FastText (Gensim)...
Formato da Matriz de Vetores Densos (Exemplos, Dimensões): (32, 50)




============
EXERCICIO 3
============
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar dados e matriz de vetores
df = pd.read_csv('sac_moveis_ac2.csv')

y = df['intencao']

print("Quantidade de mensagens:", len(df))
print("Quantidade de dimensões por vetor:", X_vetores.shape[1])
print("=====================================")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_vetores,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Mensagens para treino:", len(X_treino))
print("Mensagens para teste:", len(X_teste))
print("=====================================")



# Assumindo a matriz X_vetores gerada no Exercício 2
# Treinar o modelo de Regressão Logística
modelo_regressao = LogisticRegression(max_iter=1000)
modelo_regressao.fit(X_treino, y_treino)


def classificar_com_fallback_linear(mensagem_usuario, modelo, model_emb, limiar=0.50):
    """
    Classifica a intenção e verifica se a probabilidade atinge o limiar mínimo de confiança.
    """
    # 1. Obter o vetor denso da mensagem do usuário
    vetor_msg = obter_vetor_frase(mensagem_usuario, model_emb).reshape(1, -1)

    # TODO 1: Obter as probabilidades para cada classe usando predict_proba
    probabilidades = modelo.predict_proba(vetor_msg)[0]

    # TODO 2: Obter a maior probabilidade e o índice correspondente
    max_prob = np.max(probabilidades)
    idx_classe = np.argmax(probabilidades)
    intencao_prevista = modelo.classes_[idx_classe]

    # TODO 3: Aplicar a regra de Fallback
    if max_prob < limiar:
        return "FALLBACK_HUMANO", max_prob
    else:
        return intencao_prevista, max_prob

# === TESTE DO EXERCÍCIO 3 ===
testes = [
    "Quero saber o valor do frete do sofá",             # Intenção esperada: vendas_orcamento
    "Gostaria de ver receitas de bolo de cenoura"       # Frase fora do domínio -> Deve acionar Fallback
]

for t in testes:
    intencao, conf = classificar_com_fallback_linear(t, modelo_regressao, fasttext_model)
    print(f"Frase: '{t}' | Resultado: {intencao} | Confiança: {conf:.2%}")

y_pred = modelo_regressao.predict(X_teste)

print("\n--- Relatório de Classificação ---")
print(
    classification_report(
        y_teste,
        y_pred,
        zero_division=0
    )
)

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_teste, y_pred))


---------- RESULTADO EXERCICIO 3 -------------

Quantidade de mensagens: 32
Quantidade de dimensões por vetor: 50
=====================================
Mensagens para treino: 25
Mensagens para teste: 7
=====================================
Frase: 'Quero saber o valor do frete do sofá' | Resultado: FALLBACK_HUMANO | Confiança: 36.70%
Frase: 'Gostaria de ver receitas de bolo de cenoura' | Resultado: vendas_orcamento | Confiança: 67.85%

--- Relatório de Classificação ---
                    precision    recall  f1-score   support

logistica_entregas       0.00      0.00      0.00         2
   suporte_tecnico       0.50      1.00      0.67         1
 trocas_devolucoes       0.50      0.50      0.50         2
  vendas_orcamento       0.67      1.00      0.80         2

          accuracy                           0.57         7
         macro avg       0.42      0.62      0.49         7
      weighted avg       0.40      0.57      0.47         7

--- Matriz de Confusão ---
[[0 1 1 0]
 [0 1 0 0]
 [0 0 1 1]
 [0 0 0 2]]




============
EXERCICIO 4
============
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# TODO 1: Instanciar e treinar o KNN com n_neighbors=3
modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(X_treino, y_treino)

# Predições nas duas abordagens
y_pred_linear = modelo_regressao.predict(X_teste)
y_pred_knn = modelo_knn.predict(X_teste)

# TODO 2: Calcular a acurácia de cada modelo
acuracia_linear = accuracy_score(y_teste, y_pred_linear)
acuracia_knn = accuracy_score(y_teste, y_pred_knn)

print(f" Acurácia - Regressão Logística (Linear): {acuracia_linear:.2%}")
print(f" Acurácia - KNN (Distância K=3): {acuracia_knn:.2%}")


---------- RESULTADO EXERCICIO 4-------------
Acurácia - Regressão Logística (Linear): 57.14%
Acurácia - KNN (Distância K=3): 71.43%
