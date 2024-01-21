import copy
import math
import random
import enum
from typing import List, Dict

import pandas


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
        if value in self.possible_values:
            self.current_values.add(value)

    def remove_value(self, value):
        if '?' in self.current_values:
            self.current_values = set(self.possible_values.copy())

        self.current_values.discard(value)

    def have_value(self, value):
        if (value in self.current_values or '?' in self.current_values):
            return True
        return False

    def set_cover_all(self):
        self.current_values = set('?')

    def set_cover_nothing(self):
        self.current_values = set()

    def does_cover(self, value):
        if '?' in self.current_values or value in self.current_values:
            return True
        return False

    def is_more_general(self, other):
        if self.current_values == other.current_values:
            return GeneralizationStatus.EQUAL
        if other.current_values.issubset(self.current_values):
            return GeneralizationStatus.MORE_GENERAL
        if self.current_values.issubset(other.current_values):
            return GeneralizationStatus.LESS_GENERAL
        return GeneralizationStatus.INCOMPARABLE


class ComplexRule:
    def __init__(self, selectors: dict, predicted_class):
        self.selectors = selectors
        self.predicted_class = predicted_class
        self.amount_of_selectors = len(selectors)

    def set_cover_all(self):
        for s in self.selectors.values():
            s.set_cover_all()

    def set_cover_nothing(self):
        for s in self.selectors:
            s.set_cover_nothing()

    def set_selector(self, selector: Selector):
        self.selectors[selector.attribute_name] = selector

    def add_to_selector(self, attribute_name, value):
        self.selectors[attribute_name].add_value(value)

    def remove_from_selector(self, attribute_name, value):
        self.selectors[attribute_name].remove_value(value)

    def does_cover(self, example: dict):
        for key, value in example.items():
            if key in self.selectors.keys() and not self.selectors[key].does_cover(value):
                return False
        return True

    def is_more_general(self, other):
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
        return sum(self.does_cover(example) for example in examples)

    def specialize(self, negative_seed, positive_seed):
        # TODO check this method
        specialized_rules = []

        for selector_name, selector in self.selectors.items():
            if negative_seed[selector_name] == positive_seed[selector_name]:
                continue
            else:
                if selector.have_value(negative_seed[selector_name]):
                    specialized_rule = copy.deepcopy(self)
                    specialized_rule.remove_from_selector(selector.attribute_name,
                                                          negative_seed[selector.attribute_name])
                    specialized_rules.append(specialized_rule)

        return specialized_rules


class RuleSet:
    def __init__(self, rules: list = None) -> None:
        self.rules = rules if rules else []

    def coverage(self, rule, positive_examples, negative_examples):
        return (
                sum(rule.does_cover(example[0]) for example in positive_examples)
                - len(negative_examples)
                + sum(rule.does_cover(example[0]) for example in negative_examples)
        )

    def accuracy(self, rule, positive_examples, negative_examples, y_class_amount):
        return (
                (sum(rule.does_cover(example[0]) for example in positive_examples) + 1)
                / (sum(rule.does_cover(example[0]) for example in positive_examples + negative_examples)
                   + y_class_amount)
        )

    def class_dominance(self, rule, positive_examples, negative_examples):
        true_positives = sum(rule.does_cover(example[0]) for example in positive_examples)
        return (
                true_positives *
                math.log(
                    true_positives / (true_positives + sum(rule.does_cover(example[0]) for example in negative_examples))
                )
        )

    def append_rule(self, rule: ComplexRule):
        self.rules.append(rule)

    def predict(self, example: dict):
        for rule in self.rules:
            if rule.does_cover(example):
                return rule.predicted_class
        return self.rules[-1].predicted_class

    def train(self, X: List[Dict], y: List, attributes_names: List, attributes_values: Dict, T: int = 10, m: int = 10,
              rule_ranking_function="coverage"):
        match rule_ranking_function:
            case "coverage":
                rule_ranking_function = self.coverage
            case "accuracy":
                y_class_amount = len(set(y))
                rule_ranking_function = self.accuracy(y_class_amount=y_class_amount)
            case "class_dominance":
                rule_ranking_function = self.class_dominance
            case _:
                raise ValueError("Unknown rule ranking function")

        all_not_covered_examples = []
        for i, row in enumerate(X):
            all_not_covered_examples.append((row, y[i]))
        current_best_rule = None
        current_best_rule_coverage = None
        while len(self.rules) < T:
            current_best_rule = None
            current_best_rule_coverage = 0
            positive_seed = (all_not_covered_examples[0])

            positive_examples = []
            negative_examples = []
            for row in all_not_covered_examples:
                if row[1] == positive_seed[1]:
                    positive_examples.append(row)
                else:
                    negative_examples.append(row)

            selectors = {}
            for attributes_name in attributes_names:
                selectors[attributes_name] = Selector(attributes_name, attributes_values[attributes_name])
            current_rules = [ComplexRule(selectors, all_not_covered_examples[0][1])]
            current_rules[0].set_cover_all()

            current_seed_index = 0

            while negative_examples and current_seed_index < len(negative_examples):
                negative_seed = negative_examples[current_seed_index]
                new_rules = []
                for rule in current_rules:
                    if rule.does_cover(negative_seed[0]):
                        new_rules += rule.specialize(negative_seed[0], positive_seed[0])
                    else:
                        new_rules.append(rule)
                current_rules = new_rules.copy()
                for i, rule in enumerate(current_rules):
                    if rule not in new_rules:
                        continue
                    for j, rule2 in enumerate(current_rules[i + 1:]):
                        if rule2 not in new_rules:
                            continue
                        which_is_general = rule.is_more_general(rule2)  # I DON'T LIKE DELETING FROM LIST WHILE
                        # ITERATING OVER IT SO I DON'T DO IT
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
                current_seed_index += 1
                if len(current_rules) > m:
                    current_rules.sort(
                        key=lambda c_rule: rule_ranking_function(c_rule, positive_examples, negative_examples),
                        reverse=True
                    )
                    current_rules = current_rules[:m]
                if rule_ranking_function(current_rules[0], positive_examples,
                                         negative_examples) > current_best_rule_coverage:
                    current_best_rule = current_rules[0]
                    current_best_rule_coverage = rule_ranking_function(current_rules[0], positive_examples,
                                                                       negative_examples)
                current_seed_index += 1

            self.rules.append(current_best_rule)  # TODO check if this saves correctly in next iteration
            all_not_covered_examples = negative_examples
            for i, example in enumerate(positive_examples):
                if not current_best_rule.does_cover(example[0]):
                    all_not_covered_examples.append(example)
            all_not_covered_examples = positive_examples + negative_examples
            random.shuffle(all_not_covered_examples)

        pass
        # @TODO: implement, current implementation is not finished

        # # Separate the training data into positive and negative examples
        # positive_examples = [example for i, example in enumerate(X) if y[i] == 1]
        # negative_examples = [example for i, example in enumerate(X) if y[i] == 0]

        # # Initialize the rule counter
        # t = 0

        # # Repeat until there are no more positive examples or the rule limit is reached
        # while positive_examples and t < T:
        #     # Select the first positive example as the seed
        #     seed = positive_examples[0]

        #     # Initialize the rule set with a universal rule
        #     G = [ComplexRule(attribute_subset)]

        #     # Repeat until there are no more negative examples covered by the rule set
        #     while any(rule.does_cover(example) for rule in G for example in negative_examples):
        #         # Select the first negative example covered by the rule set as the negative seed
        #         negative_seed = next(example for example in negative_examples if any(rule.does_cover(example) for rule in G))

        #         # Specialize each rule in the rule set using the negative seed and the positive seed
        #         G = [specialized_rule for rule in G for specialized_rule in rule.specialize(negative_seed, seed)]

        #         # If the rule set is too large, keep only the best rules according to some criterion (e.g., coverage)
        #         if len(G) > 2:
        #             G.sort(key=lambda rule: rule.coverage(positive_examples), reverse=True)
        #             G = G[:2]

        #     # Select the best rule from the rule set according to some criterion (e.g., coverage)
        #     best_rule = max(G, key=lambda rule: rule.coverage(positive_examples))

        #     # Add the best rule to the list of rules
        #     self.rules.append(best_rule)

        #     # Remove the positive examples covered by the best rule
        #     positive_examples = [example for example in positive_examples if not best_rule.does_cover(example)]

        #     # Increment the rule counter
        #     t += 1


class RandomForest:
    def __init__(self):
        self.rulesets = []

    def train(self, X: '''pandas.Dataframe''', y: pandas.DataFrame, attributes_values, B, M):
        max_attributes = len(attributes_values)
        num_attributes = math.floor(math.sqrt(max_attributes))

        attributes = attributes_values.keys()

        xy = list(zip(X, y))

        for _ in range(B):
            xy_subset = random.sample(xy, M)
            X_subset, y_subset = zip(*xy_subset)

            attribute_subset = random.choices(attributes, k=num_attributes)
            X_subset = X_subset[attribute_subset]

            ruleset = RuleSet()
            ruleset.train(X_subset, y_subset, attribute_subset, attributes_values)
            self.rulesets.append(ruleset)

    def predict(self, example: dict):
        predictions = {}
        for ruleset in self.rulesets:
            pred = ruleset.predict(example)
            if pred not in predictions:
                predictions[pred] = 1
            else:
                predictions[pred] += 1
        return max(predictions, key=predictions.get)


if __name__ == "__main__":
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
    ruleSet.train(X, y, attributes_names, attribute_values, 2, 2, "coverage")
    pass