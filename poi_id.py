#!/usr/local/bin/python
# coding: latin-1

import os
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from tester import dump_classifier_and_data

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def getData():
    with open("final_project_dataset.pkl", "r") as data_file:
        return pickle.load(data_file)

        
### Conta o total de POI's no dataset
def countPOI(dataset):
    poi_count = 0
    for key, value in dataset.items():
        if value["poi"]:
            poi_count += 1
    return poi_count
        
 
### Removendo dados que não possuem nenhum valor ou são dados agredados de planilha
def removeOutliers(dataset):
    for key in ["TOTAL", "THE TRAVEL AGENCY IN THE PARK", "LOCKHART EUGENE E"]:
        print key, "- Ex: salary = {}, bonus = {}".format(dataset[key]["salary"], dataset[key]["bonus"]), "\n"
        dataset.pop(key, 0)

        
### Checando dados com PDF e tentando encontrar possíveis erros de sincronia 
### como valores que deveriam ser negativos e estão positivos
def CheckDataSyncProblems(dataset):
    for name, info in sorted(dataset.items()):
        for dkey, dvalue in info.items():
            if dkey == "restricted_stock_deferred" and dvalue != "NaN" and dvalue > 0:
                print name, "- Ex: restricted_stock_deferred = {}".format(info["restricted_stock_deferred"]), "\n"
                break

                
### Corrigindo valores para BELFER ROBERT e BHATNAGAR SANJAY
def FixDataSyncProblems(dataset):
    dataset["BELFER ROBERT"] = dataset.fromkeys(dataset["BELFER ROBERT"], "NaN")
    dataset["BELFER ROBERT"]["deferred_income"] = -102500
    dataset["BELFER ROBERT"]["expenses"] = 3285
    dataset["BELFER ROBERT"]["director_fees"] = 102500
    dataset["BELFER ROBERT"]["total_payments"] = 3285
    dataset["BELFER ROBERT"]["restricted_stock"] = 44093
    dataset["BELFER ROBERT"]["restricted_stock_deferred"] = -44093
    dataset["BELFER ROBERT"]["poi"] = False

    dataset["BHATNAGAR SANJAY"] = dataset.fromkeys(dataset["BHATNAGAR SANJAY"], "NaN")
    dataset["BHATNAGAR SANJAY"]["expenses"] = 137864
    dataset["BHATNAGAR SANJAY"]["total_payments"] = 137864
    dataset["BHATNAGAR SANJAY"]["exercised_stock_options"] = 15456290
    dataset["BHATNAGAR SANJAY"]["restricted_stock"] = 2604490
    dataset["BHATNAGAR SANJAY"]["restricted_stock_deferred"] = -2604490
    dataset["BHATNAGAR SANJAY"]["total_stock_value"] = 15456290
    dataset["BHATNAGAR SANJAY"]["from_messages"] = 29
    dataset["BHATNAGAR SANJAY"]["to_messages"] = 523
    dataset["BHATNAGAR SANJAY"]["shared_receipt_with_poi"] = 463
    dataset["BHATNAGAR SANJAY"]["from_this_person_to_poi"] = 1
    dataset["BHATNAGAR SANJAY"]["from_poi_to_this_person"] = 0
    dataset["BHATNAGAR SANJAY"]["poi"] = False

### Método utilizado para decidir se novos features computados deverão ser adicionados no conjunto de análise
def setupNewFeatures():
    os.system("cls")
    choice = raw_input("Gostaria de adicionar novos features, como a fração de emails enviados\ne recebidos de POI's? (S / N)\n\n")
    if choice in ["s","n","S","N"]:
        return choice
    else:
        return setupNewFeatures()    


### Calcula a fração de determinadas mensagens em relação ao total
def computeFraction(poi_messages, all_messages):
    fraction = 0.
    if poi_messages != "NaN" and all_messages != "NaN":
        fraction = float(poi_messages) / all_messages
    return fraction    

    
### Adicionando novos features no dataset que seriam as frações de e-mail enviados para um POI
### e recebidas de um POI
def AddNewFeatures(dataset, features_list):
    new_features = ["fraction_from_poi_email", "fraction_to_poi_email"]
    numerator_features = ["from_poi_to_this_person", "from_this_person_to_poi"]
    denominator_features = ["to_messages", "from_messages"]

    for name in dataset:
        data_point = dataset[name]

        for i, feature in enumerate(new_features):
            if data_point["poi"]:
                data_point[feature] = 'NaN'
            else:
                poi_messages = data_point[numerator_features[i]]
                all_messages = data_point[denominator_features[i]]
                fraction_messages = computeFraction(poi_messages, all_messages)
                data_point[feature] = fraction_messages

    return features_list + new_features

    
def TestNewFeatures(dataset, name, new_features):
    numerator_features = ["from_poi_to_this_person", "from_this_person_to_poi"]
    denominator_features = ["to_messages", "from_messages"]

    print name, "\n- {} = {:.4f} ({} / {})\n- {} = {:.4f} ({} / {})\n".format(new_features[0], 
                                                                  dataset[name][new_features[0]],
                                                                  dataset[name][numerator_features[0]],
                                                                  dataset[name][denominator_features[0]],
                                                                  new_features[1], 
                                                                  dataset[name][new_features[1]],
                                                                  dataset[name][numerator_features[1]],
                                                                  dataset[name][denominator_features[1]])

                                                                  
### Esse método utiliza o Pandas para avaliar a quantidade de nulos em cada feature
def describeData(dataset):
    df_list = []
    for key, value in dataset.items():
        df_list.append(value)
    
    df = pd.DataFrame(df_list, columns = dataset.items()[0][1].keys())

    for i in df.columns:
        df[i][df[i].apply(lambda i: True if str(i) == "NaN" else False)]=None
    
    df = df.convert_objects(convert_numeric=True)
    df.info()

    
### Essa é uma versão modificada do método test_classifier disponibiliza em tester.py
### A diferença é que ela retorna os valores numa tupla, em vez de printá-los na tela
### e adiciona nesta tupla as métricas de precision e recall
def my_test_classifier(clf, dataset, feature_list, folds = 1000):
    from sklearn.cross_validation import StratifiedShuffleSplit
    PERF_FORMAT_STRING = "\
    \tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
    Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
    RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
    \tFalse negatives: {:4d}\tTrue negatives: {:4d}"

    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break

    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        results = (clf, 
                   PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5),
                   RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives),
                   precision, 
                   recall)
    except:
        results = (clf, "Got a divide by zero when trying out:", "Precision or recall may be undefined due to a lack of true positive predicitons.", 0, 0)
    
    return results

    
### Esse método selectiona as n melhores features usando o K-Best
### A feature "poi" não é considerada nos scores e é recolocada na primeira posição
def SelectFeatures(features_list, labels, features, num_features):
    from sklearn.feature_selection import SelectKBest
    clf = SelectKBest(k = num_features)
    clf = clf.fit(features, labels)

    feature_weights = {}
    for i, feature in enumerate(clf.scores_):
        feature_weights[features_list[1:][i]] = feature
    
    best_features = sorted(feature_weights.items(), key = lambda k: k[1], reverse = True)[:num_features]
    new_features = []
    scores = []
    for f, s in best_features:
        new_features.append(f)
        scores.append(s)
    
    return (["poi"] + new_features, [0.00] + scores)


### Esse método testa diversos parâmetros de 3 algoritmos pré-selecionados 
### (Naive Bayes, Regressão Logística e Árvore de Decisão) e retorna os melhores
### parâmetros de cada classificador e resultados dos testes com estes parâmetros
def EvaluateClassifiers(dataset, features_list):
    from sklearn.cross_validation import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.grid_search import GridSearchCV
    
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    
    classifiers = [{"classifier": GaussianNB(), 
                    "params": {"priors": [None]}}, 
                   {"classifier": LogisticRegression(),
                    "params": {
                        "C": [0.05, 0.5, 1, 10, 10**2, 10**3, 10**5, 10**10, 10**15],
                        "tol":[10**-1, 10**-2, 10**-4, 10**-5, 10**-6, 10**-10, 10**-15],
                        "class_weight":["balanced"]
                    }},
                   {"classifier": DecisionTreeClassifier(),
                    "params": {
                        "criterion": ["gini", "entropy"],
                        "min_samples_split": [10,15,20,25]
                    }}]

    results = []
    for c in classifiers:
        clf = GridSearchCV(c["classifier"], c["params"])
        clf.fit(features_train, labels_train)
        best_estimator = clf.best_estimator_
        best_params = clf.best_params_
        test_results = my_test_classifier(clf.best_estimator_, dataset, features_list)
        results.append((best_estimator, best_params, test_results))
    
    return results


### Esse método gera um relatório em txt para um range de diversas quantidades de features 
### selecionadas com o K-Best e para cada quantidade são testados os algoritmos para obter 
### os melhores parâmetros.
###
### É possível definir um threshold mínimo para que o relatório só grave os testes com as
### métricas de precision e recall acima desse threshold
def DoEvaluation(dataset, features_list, min_threshold, min_range, max_range):
    evaluation_file = "classifier_evaluation.txt"
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    print "Processando a avaliação dos classificadores (pode demorar alguns minutos)...\n\n"
    
    try:
        with open(evaluation_file, "w") as file:
            file.write("Min. Threshold: {}\n".format(min_threshold))
            file.write("Range: ({}, {})\n\n".format(min_range, max_range))
            for num_features in range(min_range, max_range):
                test_features, scores = SelectFeatures(features_list, labels, features, num_features)
                file.write("************************************\n")
                file.write("Testando com K = {} features\n\n".format(num_features))
                for i, feature in enumerate(test_features):
                    file.write("- ({:.2f}) {}\n".format(scores[i], feature))
                file.write("************************************\n\n")
            
                results = EvaluateClassifiers(dataset, test_features)
            
                for r in results:
                    if r[2][3] >= min_threshold and r[2][4] >= min_threshold:
                        file.write(">>> Best Estimator:\n\n{}\n\n".format(r[0]))
                        file.write(">>> Best Parameters:\n\n{}\n\n".format(r[1]))
                        file.write(">>> Test Results:\n\n{}\n{}\n\n".format(r[2][1], r[2][2]))
                        file.write("************************************\n\n")
            
            print "Arquivo '{}' gravado com sucesso!".format(evaluation_file)
    except IOError as err:
        print "\nErro de I/O: {}".format(err)


### Esse método solicita ao usuário os parâmetros para execução da avaliação dos classificadores:
###- min_threshold: relatório será filtrado por um valor mínimo de precision e recall (entre 0 e 1)
###- min_range: quantidade mínima de features (mínimo 1)
###- max_range: quantidade máxima de features (máximo, tamanho total das features disponíveis) 
def setupEvaluation(features_list):
    os.system("cls")
    level = raw_input("Selecione os parâmetros numéricos para avaliação dos classificadores no formato:\n\n" + 
        "min_threshold, min_range, max_range\n\n" + 
        "onde:\n" + 
        "- min_threshold: relatório será filtrado por um valor mínimo de precision e recall (entre 0 e 1)\n" + 
        "- min_range: quantidade mínima de features (mínimo 1)\n" + 
        "- max_range: quantidade máxima de features (máximo " + str(len(features_list)) + ") \n\n" + 
        "Ou digite 'p' para pular esta etapa.\n\n" + 
        "Para cada valor K no range informado, serão selecionados os K melhores features do dataset (com o K-Best)\n" + 
        "que serão utilizados na escolha do melhor classificador para esses features que também serão testados com \n" + 
        "diversos parâmetros específicos de cada classificador.\n\n" + 
        "No final do processo será gravado um arquivo TXT com os resultados da avaliação\n\n")
    if level in ["p","P"]:
        return [level, 0, 0]
    elif "," in level:
        params = level.strip().split(",") 
        try:
            params[0] = float(params[0])
            params[1] = int(params[1])
            params[2] = int(params[2])
            if (len(params) == 3 and params[0] >= 0 and params[0] <= 1 and params[1] >= 1 
                and params[1] < params[2] and params[2] <= len(features_list)):
                return params
            else:
                return setupEvaluation(features_list)
        except ValueError:
            return setupEvaluation(features_list)
    return setupEvaluation(features_list)
        

### Esse método utiliza a quantidade de features previamente computada (6) que gerou os melhores resultados de testes
### conforme relatório gerado em "DoEvaluation()" com parâmetros de threshold = 0.3 e range de features (1, 22)
def DumpBestClassifier(dataset, features_list):
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    final_features,_ = SelectFeatures(features_list, labels, features, 11)

    data = featureFormat(dataset, final_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=15)
    clf.fit(features_train, labels_train)
    test_classifier(clf, dataset, final_features)
    dump_classifier_and_data(clf, dataset, final_features)

        
def main():
    ### 1) Selecionando todos os features numéricos
    features_list = ["poi",
                    "bonus", 
                    "deferral_payments", 
                    "deferred_income", 
                    "director_fees", 
                    "exercised_stock_options", 
                    "expenses", 
                    "from_messages",
                    "from_poi_to_this_person",
                    "from_this_person_to_poi",
                    "loan_advances", 
                    "long_term_incentive", 
                    "other", 
                    "restricted_stock", 
                    "restricted_stock_deferred", 
                    "salary",
                    "shared_receipt_with_poi", 
                    "to_messages", 
                    "total_payments", 
                    "total_stock_value"]

    ### 2) Obtendo dados e fazendo um teste
    data_dict = getData()
    os.system("cls")
    print "Obtendo dados. Teste com BECK SALLY W:\n\n{}".format(data_dict["BECK SALLY W"])
    print ("\n\nQuantidade total de registros: {}\nQuantidade total de features: {}\n".format(len(data_dict), len(data_dict["BECK SALLY W"])) + 
            "Quantidade de POI's: {}".format(countPOI(data_dict)))
    raw_input("\n\nPresione ENTER para continuar\n\n")
    os.system("cls")
    
    ### 3) Removendo outliers
    print "Removendo registros que não possuem valor ou são dados agrupados na planilha\nde origem...\n\n"
    removeOutliers(data_dict)
    print "\n\nQuantidade total de registros: {}".format(len(data_dict))
    raw_input("\n\nPresione ENTER para continuar\n\n")
    os.system("cls")

    ### 4) Corrigindo registros
    print "Verificando dados que possuem erro de sincronia com o PDF de origem.\nPor exemplo, valores que deveriam ser negativos e estão positivos\n\n"
    CheckDataSyncProblems(data_dict)
    FixDataSyncProblems(data_dict)
    print "\nVerificando dados após a correção\n\n"
    print "BELFER ROBERT", "- Ex: restricted_stock_deferred = {}".format(data_dict["BELFER ROBERT"]["restricted_stock_deferred"]), "\n"
    print "BHATNAGAR SANJAY", "- Ex: restricted_stock_deferred = {}".format(data_dict["BHATNAGAR SANJAY"]["restricted_stock_deferred"]), "\n"
    raw_input("\n\nPresione ENTER para continuar\n\n")
    os.system("cls")

    ### 5) Adicionando novos features
    use_features = setupNewFeatures()
    os.system("cls")
    if use_features in ["s","S"]:
        features_list = AddNewFeatures(data_dict, features_list)
        print "Testando novos features adicionados:\n\n"
        TestNewFeatures(data_dict, "BECK SALLY W", features_list[-2:])
        raw_input("\n\nPresione ENTER para continuar\n\n")
        os.system("cls")

    ### 6) Avaliando a disponibilidade dos dados em cada feature (contagem dos nulos)
    print "Avaliando a disponibilidade dos dados em cada feature (contagem dos nulos)...\n\n"
    describeData(data_dict)
    raw_input("\n\nPresione ENTER para continuar\n\n")
    os.system("cls")

    ### 7) Avaliando a performance dos classificadores (opcional)
    min_threshold, min_range, max_range = setupEvaluation(features_list)
    os.system("cls")
    if min_threshold not in ["p","P"]:
        DoEvaluation(data_dict, features_list, min_threshold, min_range, max_range)    
        raw_input("\n\nPresione ENTER para continuar\n\n")
        os.system("cls")
        
    ### 8) Fazendo o dump com os resultados do melhor classificador encontrado
    str_result = ("Fazendo o dump com os resultados do melhor classificador encontrado...\n\n" + 
                  "Resultados com 11 features")
    if len(features_list) == 22:
        str_result = str_result + " (usando uma das features novas):\n\n"
    else:
        str_result = str_result + " (usando apenas features originais):\n\n"
    print (str_result)
    DumpBestClassifier(data_dict, features_list)
    raw_input("\n\nPara verificar novamente a performance do classificador selecionado, rode\n" + 
              "o utilitário `tester.py`.Para checar a performance de outros classificadores\n" + 
              "consulte o log gerado\n\n\nPROGRAMA ENCERRADO\n\n\nPresione ENTER para finalizar.\n")
    os.system("cls")
    
main()

