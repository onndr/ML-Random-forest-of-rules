from sklearn.ensemble import RandomForestClassifier
from algorithm import Selector, ComplexRule, RuleSet, RandomForest
from tools import get_mushrooms_data, get_titanic_data, get_students_data, quality_measures, dump_exp_results
from constants import RULE_RANKING_METHODS


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
X, y, attributes_values = get_mushrooms_data()
# domyślne wartości hiperparametrów
B = 30      # maksymalna liczba zbiorów reguł
M = 100     # wielkość podzbioru trenującego dla każdego zbioru reguł


def exp_var_rule_ranking():
    model = RandomForest()
    results = {
        r: {
            "confusion_matrix": [],
            "accuracy": [],
            "precision": [],
            "f1_score": []
        } for r in RULE_RANKING_METHODS
    }
    for r in RULE_RANKING_METHODS:
        model.train(X, y, attributes_values, B, M)
        y_pred = model.predict(X)
        cm, acc, prec, f1 = quality_measures(y, y_pred)
        results[r]["confusion_matrix"].append(cm.tolist())
        results[r]["accuracy"].append(acc)
        results[r]["precision"].append(prec)
        results[r]["f1_score"].append(f1)
        dump_exp_results("rule_ranking_method_" + r + ".json", results)


def exp_var_seed_choice():
    pass


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
