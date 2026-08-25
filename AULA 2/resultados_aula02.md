--- RESULTADOS DO LAB 01 ---
Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]

O resultado apresentado, não foi satisfatório, pois as intenções não bateram com as mensagens. Sendo assim a melhor maneira de lidar com a situação
de forma que os erros não se repetissem seria, alimentar o chat com mais frases, para que ele possa ter uma base mais ampla de pesquisa.

A função de regressão logística atua no código sendo dividida em 3 partes, na parte 1 ela vai inicializar uma instancia no classificador que poderá
ser treinada. Na parte 2 ele vai fazer a associação das mensagens e intenções, pois na parte 3 isso será usado para prever as intenções de novas frases.





--- RESULTADOS DO LAB 02 ---
Mensagem de Teste: 'Gostaria de devolver o produto que comprei'
Intenção Predita: troca_devolucao

--- Distribuição de Probabilidades por Classe ---
Classe [duvida_frete]: 27.99%
Classe [rastrear_pedido]: 24.54%
Classe [troca_devolucao]: 47.46%

O resultado apresentado foi bem sucedido, pois a intenção bateu com a mensagem e a probabilidade foi assertiva também. Se por acaso houvesse algum erro, uma boa maneira de lidar seria alimentando o chat com uma mensagem mais objetiva, de modo que a analise da mensagem faça uma avaliação mais precisa. 

A função do naive bayes no código começa com a absorção das informações de mensagem e intenção, logo em seguida converte essas informações de texto em númerico de forma que o computador entenda. Ele utiliza um algoritmo de naive bayes a reconhecer padrões para associa-los a intenções corretas, por fim prevê qual a intenção mais provavel e também mostra a porcentagem da probabilidade de acerto.







--- RESULTADOS DO LAB 03 ---
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

X = df3['mensagem']
y = df3['intencao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(X_train_vec, y_train)

predicoes = modelo_arvore.predict(X_test_vec)
acuracia = accuracy_score(y_test, predicoes)

print(f"Acurácia do Modelo: {acuracia * 100:.2f} %\n")
print("Previsões Detalhadas:")
for i, (mensagem, real, pred) in enumerate(zip(X_test, y_test, predicoes)):
    print(f"Mensagem {i+1}: '{mensagem}'")
    print(f"  Intenção Real: {real}")
    print(f"  Previsão do Modelo: {pred}\n")


Acurácia do Modelo: 33.33 %

Previsões Detalhadas:
Mensagem 1: 'Nao consigo imprimir documentos'
  Intenção Real: suporte_impressora
  Previsão do Modelo: problema_conexao

Mensagem 2: 'Não consigo entrar no sistema'
  Intenção Real: reset_senha
  Previsão do Modelo: problema_conexao

Mensagem 3: 'Minha conexao caindo toda hora'
  Intenção Real: problema_conexao
  Previsão do Modelo: problema_conexao


  #========== PRODUÇÃO DO RELATÓRIO:==============
O modelo acertou só 33% das vezes como usamos só 9 frases no total 6 para ensinar e 3 para testar, esse número engana muito.
Com tão pouca informação, as frases que sobraram para o teste podem ser muito diferentes das que usamos para treinar, o computador acaba se apegando a palavras que apareceram por pura coincidência e, por isso, ele não consegue aprender a lógica de verdade para acertar frases novas que chegarem depois.


A Árvore de Decisão funciona como um jogo de perguntas. Primeiro, o computador conta quantas vezes cada palavra aparece nas frases. Depois, ele começa a criar regras simples para separar os grupos, tipo:
"Se a frase tiver a palavra senha, coloque no grupo de reset_senha. Se não tiver, mas tiver a palavra lenta, coloque no grupo de conexão."
Ele vai fazendo essas perguntas uma atrás da outra até conseguir separar todas as intenções que ele aprendeu.


O risco é o computador "decorar" as frases em vez de entender o que elas significam de verdade.
Se você não colocar um limite, a árvore vai criando tantas regrinhas detalhadas que ela acaba decorando até os erros ou o jeito específico que cada pessoa escreveu no treino. No fim, ela fica ótima para as frases que ela já viu, mas se perde totalmente quando aparece qualquer frase um pouquinho diferente, porque ela ficou focada demais nos detalhes e esqueceu de aprender a lógica principal.







--- RESULTADOS DO LAB 04 ---
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#Atendimento Agência de Viagens
dados_agencia = {
    'mensagem' : [
        'Qual o peso maximo da bagagem', 
        'O que não posso levar na mala', 
        'Qual o valor de despacho da mala extra', 
        'Minha mala não chegou na esteira',

        'Gostaria de uma passagem pra Brasilia',
        'Quais são os pacotes de voo em promoção', 
        'Preciso de um voo para o Rio de Janeiro', 
        'Quais são as opções de passagem mais em baratas',

        'Fiquei doente e não vou poder viajar, como peço meu reembolso?', 
        'Infelizmente terei de cancelar, como peço estorno?', 
        'O meu hotel fechou e preciso receber o meu dinheiro de volta.', 
        'Comprei a passagem errada preciso do dinheiro de volta'
    ],

    'intencao' : [
        'informacao_bagagem', 'informacao_bagagem', 'informacao_bagagem', 'informacao_bagagem',
        'comprar_passagem', 'comprar_passagem', 'comprar_passagem', 'comprar_passagem',
        'solicitar_reembolso', 'solicitar_reembolso', 'solicitar_reembolso', 'solicitar_reembolso'
    ]
}

df4 = pd.DataFrame(dados_agencia)

X = df4 ['mensagem']
y = df4 ['intencao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Tfidf = TfidfVectorizer()
X_train_vec = Tfidf.fit_transform(X_train)
X_test_vec = Tfidf.transform(X_test)

modelo_nb = MultinomialNB()
modelo_nb.fit(X_train_vec, y_train)

frases_ineditas = [
    "Posso levar uma mochila extra no avião?",
    "Quero comprar um bilhete de ida para Salvador",
    "Como faço para receber o estorno do voo cancelado?"
]

frases_ineditas_vec = Tfidf.transform(frases_ineditas)

predicoes = modelo_nb.predict(frases_ineditas_vec)

print("=== RESULTADO DAS PREDICÕES ===")           
for frase, intencao in zip(frases_ineditas, predicoes):
    print(f"Mensagem: '{frase}' -> Intenção: [{intencao}]")


=== RESULTADO DAS PREDICÕES ===
Mensagem: 'Posso levar uma mochila extra no avião?' -> Intenção: [informacao_bagagem]
Mensagem: 'Quero comprar um bilhete de ida para Salvador' -> Intenção: [comprar_passagem]
Mensagem: 'Como faço para receber o estorno do voo cancelado?' -> Intenção: [comprar_passagem]


#========== PRODUÇÃO DO RELATÓRIO:==============


