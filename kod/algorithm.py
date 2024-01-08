import math
import random

import pandas


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
            self.current_values = self.possible_values.copy()

        self.current_values.discard('?')
        self.current_values.discard(value)

    def set_cover_all(self):
        self.current_values.add('?')

    def set_cover_no(self):
        self.current_values = set()

    def does_cover(self, value):
        if '?' in self.current_values or value in self.current_values:
            return True
        return False

    def is_more_general(self, other):
        if '?' in self.current_values:
            return True
        if '?' in other.current_values:
            return False
        if self.current_values == other.current_values:
            return None
        if self.current_values.issubset(other.current_values):
            return True
        return False

class ComplexRule:
    def __init__(self, selectors: dict, predicted_class):
        self.selectors = selectors
        self.predicted_class = predicted_class
        self.n_selectors = len(selectors)

    def set_cover_all(self):
        for s in self.selectors:
            s.set_cover_all()

    def set_cover_no(self):
        for s in self.selectors:
            s.set_cover_no()

    def set_selector(self, selector: Selector):
        self.selectors[selector.attribute_name] = selector

    def add_to_selector(self, attribute_name, value):
        self.selectors[attribute_name].add_value(value)

    def remove_from_selector(self, attribute_name, value):
        self.selectors[attribute_name].remove(value)

    def does_cover(self, example: dict):
        for key, value in example.items():
            if key in self.selectors.keys() and not self.selectors[key].does_cover(value):
                return False
        return True

    def is_more_general(self, other):
        status = set()
        for attr_name, selector in self.selectors:
            status.add(selector.is_more_general(other.selectors[attr_name]))

        if True in status and False in status:
            return None

        return status.pop()

    def coverage(self, examples):
        return sum(self.does_cover(example) for example in examples)

    def specialize(self, negative_seed, positive_seed):
        # Initialize the specialized rules
        specialized_rules = []

        # Iterate over the selectors
        for selector in self.selectors.values():
            # If the selector covers the negative seed, specialize the rule by removing the negative seed from the selector
            if selector.does_cover(negative_seed[selector.attribute_name]):
                specialized_rule = ComplexRule(self.selectors, self.predicted_class)
                specialized_rule.remove_from_selector(selector.attribute_name, negative_seed[selector.attribute_name])
                specialized_rules.append(specialized_rule)

        return specialized_rules


class RuleSet:
    def __init__(self, rules: list = None) -> None:
        self.rules = rules if rules else []

    def append_rule(self, rule: ComplexRule):
        self.rules.append(rule)

    def predict(self, example: dict):
        for rule in self.rules:
            if rule.does_cover(example):
                return rule.predicted_class
        return self.rules[-1].predicted_class

    def train(self, X, y, attribute_subset, attributes_values, T=10):
        #@TODO: implement, current implementation is not finished

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

    def train(self, X: pandas.Dataframe, y: pandas.DataFrame, attributes_values, B, M):
        max_attributes = len(attributes_values)
        num_attributes = math.floor(math.sqrt(max_attributes))

        attributes = attributes_values.keys()

        xy = list(zip(X, y))

        for _ in range(B):
            # Select a random subset of the zipped list
            xy_subset = random.sample(xy, M)

            # Unzip the subset back into X and y
            X_subset, y_subset = zip(*xy_subset)

            # Select a random subset of the attributes
            attribute_subset = random.choices(attributes, k=num_attributes)
            X_subset = X_subset[attribute_subset]

            # Train a new rule set
            ruleset = RuleSet()
            ruleset.train(X_subset, y_subset, attribute_subset, attributes_values)

            # Add the trained rule set to the forest
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
