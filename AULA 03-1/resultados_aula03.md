--- RESULTADOS DO LAB 01 (AULA 03) ---
Mensagem: 'Preciso urgente da segunda via da fatura'
Intenção Predita: [segunda_via]
Vocabulário Filtrado (sem stopwords): ['2a', '2a via', 'aberto', 'acordo', 'acordo pagar', 'alterar', 'alterar endereço', 'app', 'atrasada', 'atualizo', 'atualizo dados', 'boleto', 'cadastramento', 'dados', 'dados residenciais', 'débito', 'débito aberto', 'dívida', 'emitir', 'emitir segunda', 'endereço', 'endereço cadastramento', 'fatura', 'fatura atrasada', 'fazer', 'fazer um', 'gostaria', 'gostaria alterar', 'negociar', 'negociar pagamento', 'no', 'no app', 'onde', 'onde atualizo', 'pagamento', 'pagamento dívida', 'pagar', 'pagar débito', 'posso', 'posso emitir', 'residenciais', 'residenciais no', 'segunda', 'segunda via', 'um', 'um acordo', 'via', 'via boleto', 'via fatura']


#========== PRODUÇÃO DO RELATÓRIO:==============
A remoção de stopwords diminui o tamanho do vocabulário removendo palavras de baixo valor, resultando em um vocabulário mais limpo e focado em termos relevantes para o modelo.

A configuração ngram_range=(1, 2) no TfidfVectorizer foi usada no modelo , significa que o vetorizador irá considerar não apenas palavras individuais, mas também pares de palavras consecutivas ao criar o vocabulário do modelo.

A remoção de palavras genéricas permite que o modelo identifique e de preferencia a palavras mais relevantes, levando a classificações mais precisas.

#========== FIM ==============







--- RESULTADOS DO LAB 02 (AULA 03) ---

--- Relatório de Classificação ---
                     precision    recall  f1-score   support

horario_atendimento       0.50      1.00      0.67         1
        localizacao       0.00      0.00      0.00         1
    troca_devolucao       0.00      0.00      0.00         1

           accuracy                           0.33         3
          macro avg       0.17      0.33      0.22         3
       weighted avg       0.17      0.33      0.22         3

--- Matriz de Confusão ---
[[1 0 0]
 [1 0 0]
 [0 1 0]]


 #========== PRODUÇÃO DO RELATÓRIO:==============

Neste método de treinamento a IA avalia cada intenção utilizando métricas, a precision indica quantas vezes ela acertou o palpite, por exemplo: A cada mensagem observada a IA atribui uma intenção, considerando que a precisão do horario de atendimento foi de 50% significa que para cada analise geral feita ela acertou metade das vezes. 
Já o recall indica a capacidade da mesma de encontrar todas as mensagens reais daquela intenção, sendo assim se o recall do horário de atendimento foi de 100% então todas as mensagens referentes a essa intenção foram encontradas. 
E por fim o F1-Score é utilizado para encontrar uma média harmônica entre o precision e o recall, dito isso, quanto mais alto o score melhor foi o desempenho da IA.

A Matriz de Confusão apresentada no resultado pode ser interpretada da seguinte maneira, as linhas representam as classes verdadeiras(O que deveria ser), e as colunas as classes que o modelo previu. A diagonal principal, que são os numeros que vão do canto superior esquerdo ao canto inferior direito representam as previsões corretas, já os numeros fora da diagonal são as previsões incorretas.

Em contextos de classes desbalanceadas, onde a distribuição de categorias não é uniforme, a acurácia como métrica de desempenho pode ser enganosa. Um modelo pode atingir alta acurácia simplesmente por classificar majoritariamente a classe dominante, falhando, no entanto, na identificação das classes minoritárias. Para uma avaliação mais robusta e representativa da performance do modelo, especialmente em cenários com classes de interesse sub-representadas, é imprescindível complementar a análise com métricas como Precisão, Recall e F1-score. Estas métricas fornecem uma visão detalhada da capacidade do modelo em lidar com cada classe individualmente, revelando a ocorrência de falsos positivos e falsos negativos de maneira mais específica.
#========== FIM ==============






# ============================================================
# LAB 03 - AULA 03 (MLCB): Scikit-Learn Pipeline (Modo TODO)
# ============================================================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados_rh = {
    'mensagem': [
        'Como solicitar minhas ferias?', 'Quero agendar meu periodo de ferias',
        'Onde baixo meu holerite do mes?', 'Preciso do comprovante de rendimentos',
        'Como cadastrar meu atestado medico?', 'Onde envio o atestado de consulta?'
    ],
    'intencao': [
        'solicitar_ferias', 'solicitar_ferias',
        'obter_holerite', 'obter_holerite',
        'enviar_atestado', 'enviar_atestado'
    ]
}

df3 = pd.DataFrame(dados_rh)

X = df3['mensagem']
y = df3['intencao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=['meu', 'de', 'do', 'minhas', 'o'])),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)

predicoes = pipeline.predict(X_test)
print(f"Acuracia via Pipeline: {accuracy_score(y_test, predicoes) * 100:.2f}%")


--- RESULTADOS DO LAB 03 (AULA 03) ---
Acuracia via Pipeline: 0.00%


#========== PRODUÇÃO DO RELATÓRIO:==============
# 1 - Cole o código corrigido e a acurácia obtida.
# 2 - Qual é a grande vantagem de utilizar o objeto Pipeline no Scikit-Learn?
# 3 - Por que o Pipeline evita que erros de pré-processamento ocorram entre treino e teste?
# Todos os resultados devem ser inseridos no arquivo resultados_aula03.md
#========== FIM ==============
