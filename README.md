# Udacity Nanodegree Fundamentos de Data Science 2 - Projeto 4 (Machine Learning)
Autor: M�rcio Souza de Oliveira

**Question�rio de avalia��o dos resultados do projeto**

**1) Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  ****[relevant rubric items: &quot;data exploration&quot;, &quot;outlier investigation&quot;]**

**Resposta:** O objetivo desse projeto � usar t�cnicas de Machine Learning para treinar algoritmos (classificadores) com dados reais de fraudes que ocorreram na empresa americana [ENRON](https://en.wikipedia.org/wiki/Enron) para podermos prever o envolvimento de poss�veis suspeitos em casos similares em outras empresas.

O conjunto de dados possui informa��es financeiras de cada funcion�rio da ENRON como sal�rio, b�nus, valores em a��es, pagamentos realizados e recebidos, quantidades de e-mails enviados e recebidos, conte�do dos e-mails, etc. Al�m disso, temos a informa��o extra dos funcion�rios que foram classificados como &quot;POI&quot; (_person of interest_), ou seja, daqueles que realmente foram acusados de estarem envolvidos com as fraudes, que � uma informa��o relevante para treinar os algoritmos.

No total temos 146 registros de funcion�rios com 21 atributos cada onde todos foram todos utilizados, com exce��o do atributo _&quot;email\_address&quot;_ que era o �nico n�o num�rico e que n�o era relevante para os testes. No projeto tamb�m foram disponibilizados os conte�dos de diversos e-mails que talvez pudessem ser relevantes para criar novos atributos, mas n�o foram utilizados. Desse conjunto percebi que apenas 18 registros foram marcados como &quot;POI&quot; e temos bastantes atributos com dados dispon�veis, como � o caso dos &quot;_loan\_advances&quot;_ que s� aparece em 3 registros ou _&quot;restricted\_stock\_deffered&quot;_ que s� aparece em 17 registros

Tr�s registros foram removidos: o primeiro era um totalizador (_TOTAL_) criado na planilha original e foi importado incorretamente como um dado, portanto destacava-se como _&quot;outlier&quot;_, o segundo (_LOCKHART EUGENE E_) n�o possu�a dados em nenhum atributo e o �ltimo (_THE TRAVEL AGENCY IN THE PARK_) n�o parecia ser um funcion�rio.

Por fim, tamb�m reparei que 2 registros foram importados incorretamente da planilha original (os dados estavam deslocados de suas colunas corretas) e fiz o acerto manual dos mesmos (_BELFER ROBERT_ e _BHATNAGAR SANJAY_).

**2) What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: &quot;create new features&quot;, &quot;intelligently select features&quot;, &quot;properly scale features&quot;]**

**Resposta:** Os atributos escolhidos foram selecionados usando o algoritmo _&quot;SelectKBest&quot;._ Para chegar na quantidade de atributos ideais, usei uma t�cnica de for�a bruta que come�ava com apenas 1 atributo para ser selecionado pelo _&quot;SelectKBest&quot;_ e em seguida testava os classificadores com apenas esse atributo, em seguida utilizei 2 atributos selecionados pelo _&quot;SelectKBest&quot;_ e repeti os testes. E isso foi se repetindo at� chegar no total dos 21 atributos poss�veis (incluindo os 2 novos, explicados mais abaixo) para serem testados com os classificadores. Ap�s an�lise de todos resultados, o n�mero ideal de atributos para o melhor classificador se resumiu a 11 com os seguintes scores do _&quot;SelectKBest&quot;_:

- Valor total em a��es (22.51)
- Op��es de a��es exercidas (22.35)
- B�nus (20.79)
- Sal�rio (18.29)
- Receita diferida (11.42)
- Incentivo de longo prazo (9.92)
- Pagamentos totais (9.28)
- A��es restritas (8.83)
- Recibos compartilhados com POI (8.59)
- Adiantamentos de empr�stimos (7.18)
- % dos e-mails totais destinados para um POI (16.41)

N�o houve necessidade de escalonamento dos atributos uma vez que essa t�cnica n�o afeta os algoritmos selecionados para os testes.

Dois atributos foram computados e adicionados ao conjunto original:

- % dos e-mails totais destinados para um POI
- % dos e-mails totais recebidos de um POI

Como e-mails s�o um meio comum de comunica��o em empresas, a quantidade total poderia n�o ser relevante para identificar um suspeito, ent�o optei por trabalhar apenas com o percentual das mensagens que foram trocadas com um POI para tentar identificar uma poss�vel afinidade do suspeito nas fraudes. Para funcion�rios que j� estavam marcados como POI n�o computei esses dados para n�o enviesar o classificador.

O % dos e-mails totais destinados para um POI acabou se mostrando como uma boa escolha ficando em d�cimo primeiro lugar dentre os atributos selecionados para o melhor classificador, como listado acima. Curiosamente, o atributo de adiantamentos de empr�stimos (_&quot;loan advances&quot;_) que s� foi informado em 3 registro, como j� comentado na resposta da pergunta 1, ficou em d�cimo lugar.

**OBS:** Mais detalhes sobre os relat�rios gerados com os testes dos classificadores est�o nos ap�ndices desse documento.

**3) What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: &quot;pick an algorithm&quot;]**

**Resposta:** O classificador selecionado foi o _&quot;Decision Tree&quot;_ com os seguintes par�metros ajustados:

- _criterion=&#39;entropy&#39;_
- _min\_samples\_split=15_

Tamb�m foram testados _&quot;Gaussian Naive Bayes&quot;_ e _&quot;Logistic Regression&quot;_.

Quando usei um dos atributos novos, o melhor resultado obtido foram com 11 no total selecionados pelo _&quot;SelectKBest&quot;_:

| **Classificador** | **Acur�cia (****&quot;Accuracy&quot;****)** | **Precis�o (****&quot;Precision&quot;****)** | **Abrang�ncia (****&quot;Recall&quot;****)** |
| --- | --- | --- | --- |
| **Gaussian Naive Bayes (padr�o)** | 0.8223 | 0.3258 | 0.3115 |
| **DecisionTree (ajustado)** | **0.9299** | **0.7881** | **0.6490** |
| **LogisticRegression (ajustado)** | 0.5996 | 0.1865 | 0.5960 |

Fazendo a mesma an�lise usando apenas os atributos originais, o melhor resultado obtido foi com apenas 5 atributos selecionados pelo _&quot;SelectKBest&quot;_ e o Gaussian Naive Bayes mostrou melhor performance:

| **Classificador** | **Acur�cia (****&quot;Accuracy&quot;****)** | **Precis�o (****&quot;Precision&quot;****)** | **Abrang�ncia (****&quot;Recall&quot;****)** |
| --- | --- | --- | --- |
| **Gaussian Naive Bayes (padr�o)** | **0.8470** | **0.4541** | **0.3510** |
| **DecisionTree (ajustado)** | 0.8201 | 0.3401 | 0.2755 |
| **LogisticRegression (ajustado)** | 0.6516 | 0.2093 | 0.5180 |



**4) What does it mean to tune the parameters of an algorithm, and what can happen if you don&#39;t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  ****[relevant rubric items: &quot;discuss parameter tuning&quot;, &quot;tune the algorithm&quot;]**

**Resposta:** Ajustar os par�metros de um classificador significa encontrar uma combina��o de par�metros que resultar� na melhor performance do classificador de acordo com as m�tricas que estamos avaliando. Quando realizei testes iniciais com os par�metros padr�o de cada classificador, os resultados para algumas m�tricas, como precis�o e abrang�ncia, ficaram bem baixos, sequer atingindo o valor m�nimo de 0.3 exigidos nesse projeto para essas mesmas m�tricas. Portanto � algo muito importante fazer o ajuste dos par�metros.

Os ajustes foram feitos tanto manualmente quanto automaticamente. A parte manual � a escolha dos atributos dos dados e a quantidade dos mesmos que ser�o utilizados para o ajuste dos par�metros de cada classificador. Isso tamb�m � essencial para algoritmos como o Gaussian Naive Bayes que n�o possui par�metros para serem ajustados, ent�o selecionar bons atributos � essencial para atingir o melhor desempenho com ele.

Como citado na resposta da pergunta 1, utilizei um m�todo de for�a-bruta, onde eu realizei testes com todas as quantidades poss�veis de atributos ranqueadas pelo _&quot;SelectKBest&quot;_ e, para cada quantidade de atributos selecionados, foi feito o ajuste autom�tico dos par�metros de cada classificador (pelo menos dos que possuem par�metros) usando o _&quot;GridSearchCV&quot;_ que resultou em excelentes combina��es de par�metros.

Para o classificador _&quot;Decision Tree&quot;_ foram usadas as seguintes combina��es de valores dos seguintes par�metros:

- **criterion** _[&#39;gini&#39;, &#39;entropy&#39;] � fun��o para medir a qualidade de uma divis�o, onde &quot;gini&quot; corresponde �_ [_impureza Gini_](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) _e &quot;entropy&quot; corresponde ao_ [_ganho de informa��o_](https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain)_._
- **min\_samples\_split** _[10,15,20,25] � a quantidade m�nima de amostras requeridas para dividir um n� interno_

J� para o classificador _&quot;Logistic Regression&quot;_ foram usadas as seguintes combina��es de valores dos seguinte par�metros:

- **C** _[0.05, 0.5, 1, 10, 10\*\*2, 10\*\*3, 10\*\*5, 10\*\*10, 10\*\*15] � Controla o limite de troca entre uma fronteira de decis�o suave e outra que classifica todos os pontos de treinamento corretamente._
- **tol** _[10\*\*-1, 10\*\*-2, 10\*\*-4, 10\*\*-5, 10\*\*-6, 10\*\*-10, 10\*\*-15] � toler�ncia para o crit�rio de interrup��o_



**5) What is validation, and what&#39;s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: &quot;discuss validation&quot;, &quot;validation strategy&quot;]**

**Resposta:** Valida��o significa separar parte do seu conjunto de dados para treinamento do classificador e outra para avaliar a performance do classificador. O grande desafio � fazer uma boa divis�o para que o classificador n�o fique ou muito enviesado (&quot;bias&quot;) � ignora os dados independente do treinamento � ou com muita vari�ncia (&quot;variance&quot;) � fica extremamente perceptivo aos dados replicando apenas o que j� viu.

 A valida��o foi realizada usando o _&quot;train\_test\_split&quot;_ do _sklearn_ onde 30% dos dados foram reservados para testes e 70% para treinamento. Ap�s a obten��o dos melhores par�metros do classificador com o _&quot;GridSearchCV&quot;_ e a quantidade de atributos selecionadas com o _&quot;SelectKBest&quot;_, os resultados foram avaliados com valida��o-cruzada usando o algoritmo &quot;_StratifiedShuffleSplit&quot;_ do _sklearn_ com 1000 &quot;folds&quot;, da mesma forma realizada no m�todo _&quot;test\_classifier&quot;_ do arquivo _&quot;tester.py&quot;_ disponibilizado no projeto.

Acredito que essa escolha foi apropriada porque o &quot;_StratifiedShuffleSplit&quot;_ � um validador-cruzado que testa diversas combina��es de dados mantendo um percentual similar de POI&#39;s do conjunto completo em cada teste, ent�o deve gerar m�tricas mais consistentes, e aproveitando para j� avaliar quais classificadores iriam atingir as m�tricas m�nimas requeridas no projeto (0.3 para precis�o e abrang�ncia) usando esse mesmo algoritmos e quantidade de &quot;folds&quot;.



**6) Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm&#39;s performance.** **[relevant rubric item: &quot;usage of evaluation metrics&quot;]**

**Resposta:** As m�tricas escolhidas foram precis�o (&quot;precision&quot;), abrang�ncia (&quot;recall&quot;) e acur�cia (&quot;accuracy&quot;). Em m�dia, o classificador final atinge:

- Acur�cia: 0.92
- Precis�o: 0.75
- Abrang�ncia: 0.64

O que significa de modo geral que o classificador tem �tima performance para identificar a quantidade POI&#39;s no conjunto, mas com maior foco em afirmar com confian�a quais os funcion�rios que s�o de fato um POI, mesmo com o efeito colateral de talvez deixar de registrar alguns POI&#39;s leg�timos (falsos negativos). Se fosse o inverso ( abrang�ncia maior do que a precis�o), ent�o o foco seria de tentar identificar os POI&#39;s no conjunto o m�ximo poss�vel, mesmo que alguns inocentes fossem eventualmente classificados como POI (falsos positivos).

Talvez para um caso de investiga��o de fraudes, essa segunda op��o fosse mais interessante, pois � melhor encontrar os culpados primeiro e depois tentar &quot;limpar o nome&quot; dos inocentes do que ter a possibilidade de alguns acusados n�o serem identificados. Infelizmente os resultados que deram boas abrang�ncias n�o tiveram boa performance nas outras m�tricas, ent�o optei pelo resultado atual que estava mais balanceado.

**Ap�ndice:**

Em anexo ao projeto estou encaminhando relat�rios de 4 execu��es para escolha dos melhores atributos e classificadores:

- **classifier\_evaluation\_NO\_FILTERED\_NEW\_FEATURES.txt** � Lista o resultado de todos os classificadores para cada quantidade poss�vel de atributos selecionada pelo _&quot;SelectKBest&quot;_ (incluindo os 2 novos atributos computados)
- **classifier\_evaluation\_NO\_FILTERED\_OLD\_FEATURES.txt** � Mesma coisa que o anterior, mas considerando apenas os atributos originais do conjunto de dados
- **classifier\_evaluation\_FILTERED\_NEW\_FEATURES.txt** � Mesma coisa que o primeiro, mas lista apenas os classificadores que obtiveram m�tricas de _&quot;precision&quot;_ e _&quot;recall&quot;_ com um valor m�nimo de 0.3
- **classifier\_evaluation\_ FILTERED\_OLD\_FEATURES.txt** � Mesma coisa que o anterior, mas considerando apenas os atributos originais do conjunto de dados

**OBS:** O programa &quot;poi\_id.py&quot; permite que os relat�rios acima sejam gerados com base em escolhas durante a sua execu��o:

- Usar ou n�o os novos atributos computados
- Definir o limite m�nimo (_threshold_) de _&quot;precision&quot;_ e _&quot;recall&quot;_ que classificadores devem atingir para serem listados no relat�rio
- Definir a faixa de valor que dever� ser utilizada para o processo de for�a-bruta de sele��o de atributos como o &quot;SelectKBest&quot;