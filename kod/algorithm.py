class Selector:
    def __init__(self, attribute_name: str, possible_values: set):
        self.attribute_name = attribute_name
        self.possible_values = possible_values
        self.current_values = set()

    def add_value(self, value):
        if value in self.possible_values:
            self.current_values.add(value)

    def remove_value(self, value):
        if value in self.current_values:
            self.current_values.remove(value)
            self.current_values.discard('?')

    def set_cover_all(self):
        self.current_values.add('?')

    def set_cover_no(self):
        self.current_values = {}

    def does_cover(self, value):
        if '?' in self.current_values or value in self.current_values:
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
            status.add(selector.is_more_general(other.celectors[attr_name]))

        if True in status and False in status:
            return None

        return status.pop()


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


class RandomForest:
    def __init__(self, rulesets: list = None) -> None:
        self.rulesets = rulesets if rulesets else []

    def add_ruleset(self, ruleset: RuleSet):
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


def aq():
    pass

