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
# 1 - O que representam as métricas Precision, Recall e F1-Score no relatório?

# 2 - Como interpretar a diagonal principal da Matriz de Confusão?
# 3 - Por que a acurácia isolada pode ser enganosa quando temos classes desbalanceadas?
# Todos os resultados devem ser inseridos no arquivo resultados_aula03.md
#========== FIM ==============
