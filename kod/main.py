# from tools import read_data
# from constants import DATA_PATH, COLUMN_NAMES
# from algorithm import Selector, ComplexRule, RuleSet, RandomForest

def main():
    pass
    # print("Number of attributes: ", len(COLUMN_NAMES))
    #
    # mushrooms_df = read_data(DATA_PATH, COLUMN_NAMES)
    # print("Dataframe head: ", mushrooms_df.head())
    # print("Dataframe describe: ", mushrooms_df.describe())
    #
    # attributes_values = {}
    # for column in COLUMN_NAMES:
    #     attributes_values[column] = mushrooms_df[column].unique().tolist()
    # print("Attributes values: ", attributes_values)
    # y = mushrooms_df['class']
    # y[y == 'e'] = 1
    # y[y == 'p'] = 0
    # X = mushrooms_df.drop(['class'], axis=1)
    #
    # print("X: ", X)
    # print("y: ", y)
    #
    # rf = RandomForest()
    # rf.train(X, y, attributes_values, 10, 10)



if __name__ == "__main__":
    main()
