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
