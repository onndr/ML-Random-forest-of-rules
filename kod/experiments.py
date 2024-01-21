from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from algorithm import Selector, ComplexRule, RuleSet, RandomForest
from tools import get_mushrooms_data, get_titanic_data, get_students_data, quality_measures, dump_exp_results, \
    count_statistics
from constants import RULE_RANKING_METHODS, RuleRankingMethodsEnum, DefaultHyperparamsValuesEnum


# Eksperymenty
#
# Zmienne algorytmów
#   Sposoby oceny kompleksów, np. pokrycie, dokładność, dominacja klasy z uwzględnieniem pokrycia.
#   Sposoby wyboru ziarna np. losowy, pierwszy przykład, losowy z klasy dominującej.
#
# Hiperparametry
#   Użyjemy przeszukiwania losowego wartości hiperparametrów.
#
#   Rozmiar zbioru trenującego (dla całego algorytmu), np. 80%, 90%, 95%
#   Rozmiar zbiorów trenujących dla poszczególnych zbiorów reguł, np. 500 przykładów
#   Maksymalna ilość zbiorów reguł, np. 100, 500, 1400
#   Maksymalna ilość reguł w zbiorze reguł, np. 2, 10, 250
#
# Miary jakości
#   Tablica pomyłek (confusion matrix)
#   Dokładność
#   Precyzja
#   F1-score


# domyślne zbiory danych
sets = {
    "mushrooms_set": get_mushrooms_data(),
    "titanic_set": get_titanic_data(),
    "students_set": get_students_data()
}

# domyślne wartości hiperparametrów
iters = DefaultHyperparamsValuesEnum.ITERATIONS.value       # liczba iteracji eksperymentu
B = DefaultHyperparamsValuesEnum.B.value                    # maksymalna liczba zbiorów reguł
M = DefaultHyperparamsValuesEnum.M.value                    # wielkość podzbioru trenującego dla każdego zbioru reguł
T = DefaultHyperparamsValuesEnum.T.value                    # liczba drzew w lesie
m = DefaultHyperparamsValuesEnum.m.value                    # liczba atrybutów w podzbiorze dla każdego drzewa
test_size = DefaultHyperparamsValuesEnum.TEST_SIZE.value    # rozmiar zbioru testowego (w procentach)


def exp_var_rule_ranking(iters, sets, B, M, T, m, test_size):
    model = RandomForest()
    results = {
        k: {
            r.value: {
                "confusion_matrix": [],
                "accuracy": [],
                "precision": [],
                "f1_score": []
            } for r in RULE_RANKING_METHODS
        } for k in sets.keys()
    }
    results["hyperparams"] = {
        "iters": iters,
        "B": B,
        "M": M,
        "T": T,
        "m": m,
        "test_size": test_size
    }

    for k, v in sets.items():
        X, y, attributes_values, classes = v
        for r in RULE_RANKING_METHODS:
            for i in range(iters):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                model.train(X_train, y_train, attributes_values, B, M, T, m, r)
                y_pred = [model.predict(x) for x in X_test]
                print("y_test: ", y_test)
                print("y_pred: ", y_pred)
                cm, acc, prec, f1 = quality_measures(y_test, y_pred, classes)
                results[k][r.value]["confusion_matrix"].append(cm.tolist())
                results[k][r.value]["accuracy"].append(acc)
                results[k][r.value]["precision"].append(prec)
                results[k][r.value]["f1_score"].append(f1)
                print("DONE. Set: ", k, " Rule ranking method: ", r, " Iteration: ", i)
            # count average, std deviation, best, worst for each measure
            results[k][r.value]["statistics"] = count_statistics(
                results[k][r.value]["confusion_matrix"],
                results[k][r.value]["accuracy"],
                results[k][r.value]["precision"],
                results[k][r.value]["f1_score"]
            )

    # Należy pracować na zagregowanych wynikach z min. 25 uruchomień.
    # Dla takich algorytmów podaje się średnią, odchylenia standardowe, najlepszy i najgorszy wynik. Należy o tym napisać już w dokumentacji wstępnej.



    dump_exp_results("[RandomForest]_rule_ranking_methods.json", results)


exp_var_rule_ranking(iters, sets, B, M, T, m, test_size)



def exp_hyperparam_training_set_size_whole_algorithm():
    pass


def exp_hyperparam_training_set_size_per_rule_set():
    pass


def exp_hyperparam_max_rule_sets_number():
    pass


def exp_hyperparam_max_rules_per_ruleset_number():
    pass


def compare_models():
    # i guess we first find the best params for our model and then compare
    custom_forest = RandomForest()
    classic_forest = RandomForestClassifier()
    single_ruleset = RuleSet()
