--- RESULTADOS DO LAB 01 ---
Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]

O resultado apresentado, não foi satisfatório, pois as intenções não bateram com as mensagens. Sendo assim a melhor maneira de lidar com a situação
de forma que os erros não se repetissem seria, alimentar o chat com mais frases, para que ele possa ter uma base mais ampla de pesquisa.

A função de regressão logística atua no código sendo dividida em 3 partes, na parte 1 ela vai inicializar uma instancia no classificador que poderá
ser treinada. Na parte 2 ele vai fazer a associação das mensagens e intenções, pois na parte 3 isso será usado para prever as intenções de novas frases.
