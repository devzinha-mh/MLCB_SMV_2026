--- RESULTADOS DO LAB 01 (AULA 03) ---
Mensagem: 'Preciso urgente da segunda via da fatura'
Intenção Predita: [segunda_via]
Vocabulário Filtrado (sem stopwords): ['2a', '2a via', 'aberto', 'acordo', 'acordo pagar', 'alterar', 'alterar endereço', 'app', 'atrasada', 'atualizo', 'atualizo dados', 'boleto', 'cadastramento', 'dados', 'dados residenciais', 'débito', 'débito aberto', 'dívida', 'emitir', 'emitir segunda', 'endereço', 'endereço cadastramento', 'fatura', 'fatura atrasada', 'fazer', 'fazer um', 'gostaria', 'gostaria alterar', 'negociar', 'negociar pagamento', 'no', 'no app', 'onde', 'onde atualizo', 'pagamento', 'pagamento dívida', 'pagar', 'pagar débito', 'posso', 'posso emitir', 'residenciais', 'residenciais no', 'segunda', 'segunda via', 'um', 'um acordo', 'via', 'via boleto', 'via fatura']


#========== PRODUÇÃO DO RELATÓRIO:==============
A remoção de stopwords diminui o tamanho do vocabulário removendo palavras de baixo valor, resultando em um vocabulário mais limpo e focado em termos relevantes para o modelo.

A configuração ngram_range=(1, 2) no TfidfVectorizer foi usada no modelo , significa que o vetorizador irá considerar não apenas palavras individuais, mas também pares de palavras consecutivas ao criar o vocabulário do modelo.

A remoção de palavras genéricas permite que o modelo identifique e de preferencia a palavras mais relevantes, levando a classificações mais precisas.

#========== FIM ==============
