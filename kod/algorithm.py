import copy
import math
import random
import enum
from typing import List, Dict
from functools import partial
from constants import RuleRankingMethodsEnum, DefaultHyperparamsValuesEnum

# import pandas


class GeneralizationStatus(enum.Enum):
    MORE_GENERAL = 1
    LESS_GENERAL = 2
    EQUAL = 3
    INCOMPARABLE = 4


class Selector:
    def __init__(self, attribute_name: str, possible_values: set):
        self.attribute_name = attribute_name
        self.possible_values = possible_values
        self.current_values = set()

    def add_value(self, value):
        # adds new value to selector
        if value in self.possible_values:
            self.current_values.add(value)

    def remove_value(self, value):
        # removes value from selector
        if '?' in self.current_values:
            self.current_values = set(self.possible_values.copy())

        self.current_values.discard(value)

    def set_cover_all(self):
        # sets selector to cover all possible values
        self.current_values = set('?')

    def set_cover_nothing(self):
        # sets selector to cover nothing
        self.current_values = set()

    def does_cover(self, value):
        # checks if selector covers value
        if '?' in self.current_values or value in self.current_values:
            return True
        return False

    def is_more_general(self, other):
        # checks if selector is more general than other
        if self.current_values == other.current_values:
            return GeneralizationStatus.EQUAL
        if other.current_values.issubset(self.current_values) or '?' in self.current_values:
            return GeneralizationStatus.MORE_GENERAL
        if self.current_values.issubset(other.current_values) or '?' in other.current_values:
            return GeneralizationStatus.LESS_GENERAL
        return GeneralizationStatus.INCOMPARABLE


class ComplexRule:
    def __init__(self, selectors: dict, predicted_class):
        self.selectors = selectors
        self.predicted_class = predicted_class
        self.amount_of_selectors = len(selectors)

    def set_cover_all(self):
        # sets all selectors to cover all possible values
        for s in self.selectors.values():
            s.set_cover_all()

    def set_cover_nothing(self):
        # sets all selectors to cover nothing
        for s in self.selectors:
            s.set_cover_nothing()

    def set_selector(self, selector: Selector):
        # sets selector to given selector
        self.selectors[selector.attribute_name] = selector

    def add_to_selector(self, attribute_name, value):
        # adds value to selector
        self.selectors[attribute_name].add_value(value)

    def remove_from_selector(self, attribute_name, value):
        # removes value from selector
        self.selectors[attribute_name].remove_value(value)

    def does_cover(self, example: dict):
        # checks if rule covers example
        for key, value in example.items():
            if key in self.selectors.keys() and not self.selectors[key].does_cover(value):
                return False
        return True

    def is_more_general(self, other):
        # checks if rule is more general than other
        status = set()
        for attr_name, selector in self.selectors.items():
            status.add(selector.is_more_general(other.selectors[attr_name]))

        if GeneralizationStatus.INCOMPARABLE in status:
            return GeneralizationStatus.INCOMPARABLE

        if GeneralizationStatus.MORE_GENERAL in status and GeneralizationStatus.LESS_GENERAL in status:
            return GeneralizationStatus.INCOMPARABLE

        if GeneralizationStatus.MORE_GENERAL in status:
            return GeneralizationStatus.MORE_GENERAL

        if GeneralizationStatus.LESS_GENERAL in status:
            return GeneralizationStatus.LESS_GENERAL

        if GeneralizationStatus.EQUAL in status:
            return GeneralizationStatus.EQUAL

        return GeneralizationStatus.INCOMPARABLE

    def coverage(self, examples):
        # returns number of examples covered by rule
        return sum(self.does_cover(example) for example in examples)

    def specialize(self, negative_seed, positive_seed):
        # returns list of rules specialized by negative seed
        specialized_rules = []

        for selector_name, selector in self.selectors.items():
            # if negative seed and positive seed have the same value for given selector, skip it
            if negative_seed[selector_name] == positive_seed[selector_name]:
                continue
            else:
                # if negative seed is covered by selector, specialize rule by removing negative seed value from selector
                if selector.does_cover(negative_seed[selector_name]):
                    specialized_rule = copy.deepcopy(self)
                    specialized_rule.remove_from_selector(selector.attribute_name,
                                                          negative_seed[selector.attribute_name])
                    # add specialized rule to list of specialized rules
                    specialized_rules.append(specialized_rule)

        return specialized_rules


class RuleSet:
    def __init__(self, rules: list = None) -> None:
        self.rules = rules if rules else []

    def coverage(self, rule, positive_examples, negative_examples):
        # returns number of examples covered by rule (according to lecture)
        # true positives - false positives + true negatives
        return (
                sum(rule.does_cover(example[0]) for example in positive_examples)
                + len(negative_examples)
                - (2 * sum(rule.does_cover(example[0]) for example in negative_examples))
        )

    def accuracy(self, rule, positive_examples, negative_examples, y_class_amount):
        # returns accuracy of rule (according to lecture)
        # true positives + 1 / true positives + false positives + amount of classes
        return (
                (sum(rule.does_cover(example[0]) for example in positive_examples) + 1)
                / (sum(rule.does_cover(example[0]) for example in positive_examples + negative_examples)
                   + y_class_amount)
        )

    def class_dominance(self, rule, positive_examples, negative_examples):
        # returns class dominance with coverage of rule (according to lecture)
        # true positives * log(true positives / true positives + false positives)
        true_positives = sum(rule.does_cover(example[0]) for example in positive_examples)
        if true_positives == 0:
            return 0
        return (
                true_positives *
                math.log(
                    true_positives / (
                                true_positives + sum(rule.does_cover(example[0]) for example in negative_examples))
                )
        )

    def append_rule(self, rule: ComplexRule):
        # appends rule to ruleset
        self.rules.append(rule)

    def predict(self, example: dict):
        # predicts class of example
        for rule in self.rules:
            if rule.does_cover(example):
                return rule.predicted_class
        return self.rules[-1].predicted_class

    def train(self, X: list[dict], y: list, attributes_names: list, attributes_values: dict, max_rules: int = 10,
              max_rules_before_reducing: int = 10,
              rule_ranking_function=RuleRankingMethodsEnum.COVERAGE):
        # trains ruleset

        # choose rule ranking function
        match rule_ranking_function:
            case RuleRankingMethodsEnum.COVERAGE:
                rule_ranking_function = self.coverage
            case RuleRankingMethodsEnum.ACCURACY:
                y_class_amount = len(set(y))
                rule_ranking_function = partial(self.accuracy, y_class_amount=y_class_amount)
            case RuleRankingMethodsEnum.CLASS_DOMINANCE:
                rule_ranking_function = self.class_dominance
            case _:
                raise ValueError("Unknown rule ranking function")

        # create list of all not covered examples
        all_not_covered_examples = []
        for i, row in enumerate(X):
            all_not_covered_examples.append((row, y[i]))

        current_best_rule = None
        current_best_rule_coverage = None

        # main loop of rules creation
        while len(self.rules) < max_rules and all_not_covered_examples:
            current_best_rule = None
            current_best_rule_coverage = 0
            # choose positive seed (first is not covered example, next ones are random (due to shuffling in the end))
            positive_seed = (all_not_covered_examples[0])

            positive_examples = []
            negative_examples = []
            # split all not covered examples into positive and negative examples
            for row in all_not_covered_examples:
                if row[1] == positive_seed[1]:
                    positive_examples.append(row)
                else:
                    negative_examples.append(row)
            # create full selector for each attribute
            selectors = {}
            for attributes_name in attributes_names:
                selectors[attributes_name] = Selector(attributes_name, attributes_values[attributes_name])
            current_rules = [ComplexRule(selectors, all_not_covered_examples[0][1])]
            current_rules[0].set_cover_all()

            current_seed_index = 0

            # loop of creating one rule
            while negative_examples and current_seed_index < len(negative_examples):
                # assign negative seed (going through all negative examples)
                negative_seed = negative_examples[current_seed_index]
                current_seed_index += 1

                # specialize current rules by negative seed
                new_rules = []
                for rule in current_rules:
                    if rule.does_cover(negative_seed[0]):
                        new_rules += rule.specialize(negative_seed[0], positive_seed[0])
                    else:
                        new_rules.append(rule)
                current_rules = new_rules.copy()
                # remove less general rules
                for i, rule in enumerate(current_rules):
                    if rule not in new_rules:
                        continue
                    for j, rule2 in enumerate(current_rules[i + 1:]):
                        if rule2 not in new_rules:
                            continue
                        which_is_general = rule.is_more_general(rule2)
                        match which_is_general:
                            case GeneralizationStatus.INCOMPARABLE:
                                pass
                            case GeneralizationStatus.MORE_GENERAL:
                                new_rules.remove(rule2)
                                continue
                            case GeneralizationStatus.LESS_GENERAL:
                                new_rules.remove(rule)
                                break
                            case GeneralizationStatus.EQUAL:
                                new_rules.remove(rule2)
                                continue
                current_rules = new_rules.copy()
                # remove rules if there are more than the parameter provided
                if len(current_rules) > max_rules_before_reducing:
                    current_rules.sort(
                        key=lambda c_rule: rule_ranking_function(c_rule, positive_examples, negative_examples),
                        reverse=True
                    )
                    current_rules = current_rules[:max_rules_before_reducing]

                # if there are no rules there were conflict examples, remove this example from set (dealt with later)
                if not current_rules:
                    break

                # set current best rule to the best rule from current rules (if it is better than current best rule)
                if rule_ranking_function(current_rules[0], positive_examples,
                                         negative_examples) > current_best_rule_coverage:
                    current_best_rule = current_rules[0]
                    current_best_rule_coverage = rule_ranking_function(current_rules[0], positive_examples,
                                                                       negative_examples)
            # if there are no rules there were conflict examples, remove this example from set (dealt now)
            if not current_rules:
                # if it was early then go again with the same positive seed
                if not current_best_rule:
                    negative_examples.pop(current_seed_index - 1)
                    all_not_covered_examples = negative_examples + positive_examples
                    continue

                # if it was not early then add current best rule to ruleset and go again with different positive seed

                # save best rule
                self.rules.append(current_best_rule)

                # delete covered positive_examples from all not covered examples
                all_not_covered_examples = negative_examples
                for i, example in enumerate(positive_examples):
                    if not current_best_rule.does_cover(example[0]):
                        all_not_covered_examples.append(example)
                random.shuffle(all_not_covered_examples)
                continue
            # if every went right then add current best rule to ruleset
            self.rules.append(current_rules[0])
            all_not_covered_examples = negative_examples
            for i, example in enumerate(positive_examples):
                if not current_rules[0].does_cover(example[0]):
                    all_not_covered_examples.append(example)
            random.shuffle(all_not_covered_examples)


class RandomForest:
    def __init__(self):
        self.rulesets = []

    def train(self, X, y, attributes_values, amount_of_rulesets, training_set_size, max_rules_in_ruleset,
              max_rules_before_reducing, rule_ranking_function):
        # trains random forest

        self.rulesets = []
        max_attributes = len(attributes_values)
        num_attributes = math.floor(math.sqrt(max_attributes))

        attributes = list(attributes_values.keys())

        xy = list(zip(X, y))

        for _ in range(amount_of_rulesets):
            # create training subset
            xy_subset = random.sample(xy, training_set_size)
            X_subset, y_subset = zip(*xy_subset)

            # choose random attributes
            attribute_names_subset = random.choices(attributes, k=num_attributes)
            X_subset = [{k: d[k] for k in attribute_names_subset} for d in X_subset]

            ruleset = RuleSet()
            ruleset.train(X_subset, y_subset, attribute_names_subset, attributes_values, max_rules_in_ruleset,
                          max_rules_before_reducing, rule_ranking_function)
            self.rulesets.append(ruleset)

    def predict(self, example: dict):
        # predicts class of example
        predictions = {}
        for ruleset in self.rulesets:
            pred = ruleset.predict(example)
            if pred not in predictions:
                predictions[pred] = 1
            else:
                predictions[pred] += 1
        return max(predictions, key=predictions.get)


if __name__ == "__main__":
    # example of ruleset training
    X = [
        {
            "outlook": "sunny",
            "temperature": "hot",
            "humidity": "high",
            "wind": "normal"
        },
        {
            "outlook": "sunny",
            "temperature": "hot",
            "humidity": "high",
            "wind": "high"
        },
        {
            "outlook": "overcast",
            "temperature": "hot",
            "humidity": "high",
            "wind": "normal"
        },
        {
            "outlook": "rainy",
            "temperature": "mild",
            "humidity": "high",
            "wind": "normal"
        },
        {
            "outlook": "rainy",
            "temperature": "cold",
            "humidity": "normal",
            "wind": "normal"
        },
        {
            "outlook": "rainy",
            "temperature": "cold",
            "humidity": "normal",
            "wind": "high"
        },
        {
            "outlook": "overcast",
            "temperature": "cold",
            "humidity": "normal",
            "wind": "high"
        },
        {
            "outlook": "sunny",
            "temperature": "mild",
            "humidity": "high",
            "wind": "normal"
        },
        {
            "outlook": "sunny",
            "temperature": "cold",
            "humidity": "normal",
            "wind": "normal"
        },
        {
            "outlook": "rainy",
            "temperature": "mild",
            "humidity": "normal",
            "wind": "normal"
        },
        {
            "outlook": "sunny",
            "temperature": "mild",
            "humidity": "normal",
            "wind": "high"
        },
        {
            "outlook": "overcast",
            "temperature": "mild",
            "humidity": "high",
            "wind": "high"
        },
        {
            "outlook": "overcast",
            "temperature": "hot",
            "humidity": "normal",
            "wind": "normal"
        },
        {
            "outlook": "rainy",
            "temperature": "mild",
            "humidity": "high",
            "wind": "high"
        },

    ]
    y = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
    attributes_names = ["outlook", "temperature", "humidity", "wind"]
    attribute_values = {
        "outlook": ["sunny", "overcast", "rainy"],
        "temperature": ["hot", "mild", "cold"],
        "humidity": ["high", "normal"],
        "wind": ["normal", "high"]
    }
    ruleSet = RuleSet()
    ruleSet.train(X, y, attributes_names, attribute_values, 2, 2, RuleRankingMethodsEnum.COVERAGE)
    pass
