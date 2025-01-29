import itertools
import random

import joblib

if __name__ == '__main__':
    # 从2024年数据中，全部合约配对，从中抽取500配对，保存到pairs.pkl
    data = joblib.load('data2024.pkl')
    symbol_list = data.keys()
    print(len(symbol_list))
    combinations = list(itertools.combinations(symbol_list, 2))
    random_combinations = random.sample(combinations, 500)
    print(random_combinations)
    joblib.dump(random_combinations, 'pairs.pkl')
