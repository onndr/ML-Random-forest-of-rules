import enum

MUSHROOMS_DATA_PATH = "../dane/mushrooms/agaricus-lepiota.data"
MUSHROOMS_COLUMN_NAMES = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
              'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
              'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring',
              'veil-type', 'veil-color', 'ring-number', 'ring-type',
              'spore-print-color', 'population', 'habitat']

STUDENTS_DATA_PATH = "../dane/students/data.csv"

TITANIC_DATA_PATH = "../dane/titanic/train.csv"

EXPERIMENTS_RESULTS_FOLDER_PATH = "../wyniki_eksperymentów/"


class RuleRankingMethodsEnum(enum.Enum):
    COVERAGE = "coverage"
    ACCURACY = "accuracy"
    CLASS_DOMINANCE = "class_dominance"


RULE_RANKING_METHODS = [
    RuleRankingMethodsEnum.COVERAGE,
    RuleRankingMethodsEnum.ACCURACY,
    RuleRankingMethodsEnum.CLASS_DOMINANCE
]


class DefaultHyperparamsValuesEnum(enum.Enum):
    # maksymalna liczba zbiorów reguł
    B = 100
    # wielkość podzbioru trenującego dla każdego zbioru reguł
    M = 100
    # maksymalna ilość kompleksów, które pozostają po specjalizacji (najlepiej ocenione kompleksy)
    m = 5
    # maksymalna ilość reguł w jednym zbiorze reguł
    T = 10
    # metoda oceniania reguł w trakcie indukcji
    RULE_RANKING_METHOD = RuleRankingMethodsEnum.COVERAGE
    # liczba iteracji eksperymentu
    ITERATIONS = 25
    # rozmiar zbioru testowego (w procentach)
    TEST_SIZE = 0.2
