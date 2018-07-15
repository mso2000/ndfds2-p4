# Udacity Nanodegree Fundamentos de Data Science 2 - Projeto 4 (Machine Learning)
Autor: Márcio Souza de Oliveira

**Questionário de avaliação dos resultados do projeto**

#### 1) Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: &quot;data exploration&quot;, &quot;outlier investigation&quot;]

**Resposta:** O objetivo desse projeto é usar técnicas de Machine Learning para treinar algoritmos (classificadores) com dados reais de fraudes que ocorreram na empresa americana [ENRON](https://en.wikipedia.org/wiki/Enron) para podermos prever o envolvimento de possíveis suspeitos em casos similares em outras empresas.

O conjunto de dados possui informações financeiras de cada funcionário da ENRON como salário, bônus, valores em ações, pagamentos realizados e recebidos, quantidades de e-mails enviados e recebidos, conteúdo dos e-mails, etc. Além disso, temos a informação extra dos funcionários que foram classificados como &quot;POI&quot; (_person of interest_), ou seja, daqueles que realmente foram acusados de estarem envolvidos com as fraudes, que é uma informação relevante para treinar os algoritmos.

No total temos 146 registros de funcionários com 21 atributos cada onde todos foram todos utilizados, com exceção do atributo _&quot;email\_address&quot;_ que era o único não numérico e que não era relevante para os testes. No projeto também foram disponibilizados os conteúdos de diversos e-mails que talvez pudessem ser relevantes para criar novos atributos, mas não foram utilizados. Desse conjunto percebi que apenas 18 registros foram marcados como &quot;POI&quot; e temos bastantes atributos com dados disponíveis, como é o caso dos &quot;_loan\_advances&quot;_ que só aparece em 3 registros ou _&quot;restricted\_stock\_deffered&quot;_ que só aparece em 17 registros

Três registros foram removidos: o primeiro era um totalizador (_TOTAL_) criado na planilha original e foi importado incorretamente como um dado, portanto destacava-se como _&quot;outlier&quot;_, o segundo (_LOCKHART EUGENE E_) não possuía dados em nenhum atributo e o último (_THE TRAVEL AGENCY IN THE PARK_) não parecia ser um funcionário.

Por fim, também reparei que 2 registros foram importados incorretamente da planilha original (os dados estavam deslocados de suas colunas corretas) e fiz o acerto manual dos mesmos (_BELFER ROBERT_ e _BHATNAGAR SANJAY_).

Com isso a disponibilidade final dos dados em cada atributo está distribuída dessa maneira:

![Disponibilidade dos dados](/other/Disponibilidade.jpg)

#### 2) What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: &quot;create new features&quot;, &quot;intelligently select features&quot;, &quot;properly scale features&quot;]

**Resposta:** Os atributos escolhidos foram selecionados usando o algoritmo _&quot;SelectKBest&quot;._ Para chegar na quantidade de atributos ideais, usei uma técnica de força bruta que começava com apenas 1 atributo para ser selecionado pelo _&quot;SelectKBest&quot;_ e em seguida testava os classificadores com apenas esse atributo, em seguida utilizei 2 atributos selecionados pelo _&quot;SelectKBest&quot;_ e repeti os testes. E isso foi se repetindo até chegar no total dos 21 atributos possíveis (incluindo os 2 novos, explicados mais abaixo) para serem testados com os classificadores. Após análise de todos resultados, o número ideal de atributos para o melhor classificador se resumiu a 14 com os seguintes scores do _&quot;SelectKBest&quot;_:

1. Valor total em ações (22.51)
2. Opções de ações exercidas (22.35)
3. Bônus (20.79)
4. Salário (18.29)
5. Receita diferida (11.42)
6. Incentivo de longo prazo (9.92)
7. Pagamentos totais (9.28)
8. Ações restritas (8.83)
9. Recibos compartilhados com POI (8.59)
10. Adiantamentos de empréstimos (7.18)
11. % dos e-mails totais recebidos de um POI (5.83)
12. Despesas (5.42)
13. Quantidade de e-mails recebidos de um POI (5.24)
14. % dos e-mails totais destinados para um POI (4.60)

Não houve necessidade de escalonamento dos atributos uma vez que essa técnica não afeta os algoritmos selecionados para os testes.

Dois atributos foram computados e adicionados ao conjunto original:

- % dos e-mails totais destinados para um POI
- % dos e-mails totais recebidos de um POI

Como e-mails são um meio comum de comunicação em empresas, a quantidade total poderia não ser relevante para identificar um suspeito, então optei por trabalhar apenas com o percentual das mensagens que foram trocadas com um POI para tentar identificar uma possível afinidade do suspeito nas fraudes. Para funcionários que já estavam marcados como POI não computei esses dados para não enviesar o classificador.

A inclusão dos 2 novos atributos acabou se mostrando uma boa escolha visto que apareceram no ranking dentre os 14 atributos selecionados para o melhor classificador, como listado acima. Curiosamente, o atributo de adiantamentos de empréstimos (_&quot;loan advances&quot;_) que só foi informado em 3 registro, como já comentado na resposta da pergunta 1, ficou em décimo lugar.

Abaixo podemos observar o impacto da inclusão dos novos atributos e a quantidade total de atributos selecionados nas métricas (precisão e abrangência) de 2 classificadores ( _&quot;Decision Tree&quot;_ e  _&quot;Gaussian Naive Bayes&quot;_):

<p align="center">
  <img src="/other/Prec_DT (new x old).jpg" width="400" title="Árvore de Decisão - Precisão por tipo e quantidade de atributos do K-Best">
  <img src="/other/Recl_DT (new x old).jpg" width="400" title="Árvore de Decisão - Abrangência por tipo e quantidade de atributos do K-Best"><br><br>
  <img src="/other/Prec_NB (new x old).jpg" width="400" title="Naive Bayes - Precisão por tipo e quantidade de atributos do K-Best">
  <img src="/other/Recl_NB (new x old).jpg" width="400" title="Naive Bayes - Abrangência por tipo e quantidade de atributos do K-Best">
</p>

**OBS:** Podemos perceber que a precisão e abrangência do _&quot;Decision Tree&quot;_ atingem o pico de performance com 14 atributos (usando os novos atributos), enquanto que no _&quot;Gaussian Naive Bayes&quot;_ a inclusão dos novos atributos praticamente não influenciou o seu desempenho.

Aqui uma re-leitura dos gráficos acima olhando para ambas as métricas simultâneamente em cada classificador usando tantos os atributos novos quanto apenas os originais:

<p align="center">
  <img src="/other/Resultados_DT (novos).jpg" width="400" title="Árvore de Decisão - Resultados por número de atributos do K-Best (com os novos)">
  <img src="/other/Resultados_DT (originais).jpg" width="400" title="Árvore de Decisão - Resultados por número de atributos do K-Best (apenas originais)"><br><br>
  <img src="/other/Resultados_NB (novos).jpg" width="400" title="Naive Bayes - Resultados por número de atributos do K-Best (com os novos)">
  <img src="/other/Resultados_NB (originais).jpg" width="400" title="Naive Bayes - Resultados por número de atributos do K-Best (apenas originais)">
</p>

**OBS:** Todos os dados detalhados sobre os relatórios gerados com diversos testes dos classificadores estão relacionados nos apêndices deste documento.


#### 3) What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: &quot;pick an algorithm&quot;]

**Resposta:** O classificador selecionado foi o _&quot;Decision Tree&quot;_ com os seguintes parâmetros ajustados:

- _criterion=&#39;entropy&#39;_
- _min\_samples\_split=15_

Também foi testado o _&quot;Gaussian Naive Bayes&quot;_.

Quando usei os atributos novos, o melhor resultado obtido foram com 11 no total selecionados pelo _&quot;SelectKBest&quot;_ para o _&quot;Gaussian Naive Bayes&quot;_ (onde apenas um dos novos foram selecionados, apesar de que a performance foi praticamente a mesma com 14 atributos estando os 2 novos presentes) e 14 no total para o  _&quot;Decision Tree&quot;_ (com ambos os novos atributos):

| **Classificador** | **Acurácia (****&quot;Accuracy&quot;****)** | **Precisão (****&quot;Precision&quot;****)** | **Abrangência (****&quot;Recall&quot;****)** |
| --- | --- | --- | --- |
| **Gaussian Naive Bayes (k=11)** | 0.8223 | 0.3258 | 0.3115 |
| **DecisionTree (ajustado, k=14)** | **0.9402** | **0.8785** | **0.6400** |

Fazendo a mesma análise usando apenas os atributos originais, o melhor resultado obtido foi com 12 atributos selecionados pelo _&quot;SelectKBest&quot;_ e o _&quot;Gaussian Naive Bayes&quot;_ mostrou melhor performance dessa vez por atingir a meta mínima de 0.3 para as métricas de precisão e abrangência exigidas no projeto:

| **Classificador** | **Acurácia (****&quot;Accuracy&quot;****)** | **Precisão (****&quot;Precision&quot;****)** | **Abrangência (****&quot;Recall&quot;****)** |
| --- | --- | --- | --- |
| **Gaussian Naive Bayes (k=12)** | **0.8227** | **0.3267** | **0.3105** |
| **DecisionTree (ajustado, k=12)** | 0.8451 | 0.3917 | 0.2920 |

Aqui podemos observar o desempenho de cada um dos classificadores para as métricas de precisão e abrangência, usando tanto os novos atributos quanto apenas os originais:

<p align="center">
  <img src="/other/Prec_NB_x_DT (novos).jpg" width="400" title="Precisão por número de atributos do K-Best (com os novos) x Classificador">
  <img src="/other/Recl_NB_x_DT (novos).jpg" width="400" title="Abrangência por número de atributos do K-Best (com os novos) x Classificador"><br><br>
  <img src="/other/Prec_NB_x_DT (originais).jpg" width="400" title="Precisão por número de atributos do K-Best (apenas originais) x Classificador">
  <img src="/other/Recl_NB_x_DT (originais).jpg" width="400" title="Abrangência por número de atributos do K-Best (apenas originais) x Classificador">
</p>


#### 4) What does it mean to tune the parameters of an algorithm, and what can happen if you don&#39;t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  ****[relevant rubric items: &quot;discuss parameter tuning&quot;, &quot;tune the algorithm&quot;]

**Resposta:** Ajustar os parâmetros de um classificador significa encontrar uma combinação de parâmetros que resultará na melhor performance do classificador de acordo com as métricas que estamos avaliando. Quando realizei testes iniciais com os parâmetros padrão de cada classificador, os resultados para algumas métricas, como precisão e abrangência, ficaram bem baixos, sequer atingindo o valor mínimo de 0.3 exigidos nesse projeto para essas mesmas métricas. Portanto é algo muito importante fazer o ajuste dos parâmetros.

Os ajustes foram feitos tanto manualmente quanto automaticamente. A parte manual é a escolha dos atributos dos dados e a quantidade dos mesmos que serão utilizados para o ajuste dos parâmetros de cada classificador. Isso também é essencial para algoritmos como o Gaussian Naive Bayes que não possui parâmetros para serem ajustados, então selecionar bons atributos é essencial para atingir o melhor desempenho com ele.

Como citado na resposta da pergunta 1, utilizei um método de força-bruta, onde eu realizei testes com todas as quantidades possíveis de atributos ranqueadas pelo _&quot;SelectKBest&quot;_ e, para cada quantidade de atributos selecionados, foi feito o ajuste automático dos parâmetros de cada classificador (pelo menos dos que possuem parâmetros) usando o _&quot;GridSearchCV&quot;_ que resultou em excelentes combinações de parâmetros.

Para o classificador _&quot;Decision Tree&quot;_ foram usadas as seguintes combinações de valores dos seguintes parâmetros:

- **criterion** _[&#39;gini&#39;, &#39;entropy&#39;] - função para medir a qualidade de uma divisão, onde &quot;gini&quot; corresponde à_ [_impureza Gini_](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) _e &quot;entropy&quot; corresponde ao_ [_ganho de informação_](https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain)_._
- **min\_samples\_split** _[10,15,20,25] - a quantidade mínima de amostras requeridas para dividir um nó interno_


#### 5) What is validation, and what&#39;s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: &quot;discuss validation&quot;, &quot;validation strategy&quot;]

**Resposta:** Validação significa separar parte do seu conjunto de dados para treinamento do classificador e outra para avaliar a performance do classificador. O grande desafio é fazer uma boa divisão para que o classificador não fique ou muito enviesado (&quot;bias&quot;) - ignora os dados independente do treinamento - ou com muita variância (&quot;variance&quot;) - fica extremamente perceptivo aos dados replicando apenas o que já viu.

 A validação foi realizada usando o _&quot;train\_test\_split&quot;_ do _sklearn_ onde 30% dos dados foram reservados para testes e 70% para treinamento. Após a obtenção dos melhores parâmetros do classificador com o _&quot;GridSearchCV&quot;_ e a quantidade de atributos selecionadas com o _&quot;SelectKBest&quot;_, os resultados foram avaliados com validação-cruzada usando o algoritmo &quot;_StratifiedShuffleSplit&quot;_ do _sklearn_ com 1000 &quot;folds&quot;, da mesma forma realizada no método _&quot;test\_classifier&quot;_ do arquivo _&quot;tester.py&quot;_ disponibilizado no projeto.

Acredito que essa escolha foi apropriada porque o &quot;_StratifiedShuffleSplit&quot;_ é um validador-cruzado que testa diversas combinações de dados mantendo um percentual similar de POI&#39;s do conjunto completo em cada teste, então deve gerar métricas mais consistentes, e aproveitando para já avaliar quais classificadores iriam atingir as métricas mínimas requeridas no projeto (0.3 para precisão e abrangência) usando esse mesmo algoritmos e quantidade de &quot;folds&quot;.


#### 6) Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm&#39;s performance.** **[relevant rubric item: &quot;usage of evaluation metrics&quot;]

**Resposta:** As métricas escolhidas foram precisão (&quot;precision&quot;), abrangência (&quot;recall&quot;) e acurácia (&quot;accuracy&quot;). Em média, o classificador final atinge:

- Acurácia: 0.92
- Precisão: 0.75
- Abrangência: 0.64

O que significa de modo geral que o classificador tem ótima performance para identificar a quantidade POI&#39;s no conjunto, mas com maior foco em afirmar com confiança quais os funcionários que são de fato um POI, mesmo com o efeito colateral de talvez deixar de registrar alguns POI&#39;s legítimos (falsos negativos). Se fosse o inverso ( abrangência maior do que a precisão), então o foco seria de tentar identificar os POI&#39;s no conjunto o máximo possível, mesmo que alguns inocentes fossem eventualmente classificados como POI (falsos positivos).

Talvez para um caso de investigação de fraudes, essa segunda opção fosse mais interessante, pois é melhor encontrar os culpados primeiro e depois tentar &quot;limpar o nome&quot; dos inocentes do que ter a possibilidade de alguns acusados não serem identificados. Infelizmente os resultados que deram boas abrangências não tiveram boa performance nas outras métricas, então optei pelo resultado atual que estava mais balanceado.

#### Apêndice:

Em anexo ao projeto estou encaminhando relatórios de 4 execuções para escolha dos melhores atributos e classificadores:

- **classifier\_evaluation\_NO\_FILTERED\_NEW\_FEATURES.txt** - Lista o resultado de todos os classificadores para cada quantidade possível de atributos selecionada pelo _&quot;SelectKBest&quot;_ (incluindo os 2 novos atributos computados)
- **classifier\_evaluation\_NO\_FILTERED\_OLD\_FEATURES.txt** - Mesma coisa que o anterior, mas considerando apenas os atributos originais do conjunto de dados
- **classifier\_evaluation\_FILTERED\_NEW\_FEATURES.txt** - Mesma coisa que o primeiro, mas lista apenas os classificadores que obtiveram métricas de _&quot;precision&quot;_ e _&quot;recall&quot;_ com um valor mínimo de 0.3
- **classifier\_evaluation\_ FILTERED\_OLD\_FEATURES.txt** - Mesma coisa que o anterior, mas considerando apenas os atributos originais do conjunto de dados
- **Dados.xlsx** - Todos os dados das métricas que aparecem nos arquivos acima organizados em planilhas e com outras métricas adicionais (acurácia, F1 e F2)

**OBS:** O programa &quot;poi\_id.py&quot; permite que os relatórios acima sejam gerados com base em escolhas durante a sua execução:

- Usar ou não os novos atributos computados
- Definir o limite mínimo (_threshold_) de _&quot;precision&quot;_ e _&quot;recall&quot;_ que classificadores devem atingir para serem listados no relatório
- Definir a faixa de valor que deverá ser utilizada para o processo de força-bruta de seleção de atributos como o &quot;SelectKBest&quot;
