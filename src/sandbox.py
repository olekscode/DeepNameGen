import constants
from namegen import NameGenerator

import pickle
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    with open(str(constants.TRAIN_VALID_TEST_DIR / 'test_pairs.pkl'), 'rb') as f:
        test_pairs = pickle.load(f)

    namegen = NameGenerator()
    suggestions = []

    for row in tqdm(test_pairs):
        source_tokens = ' '.join(row[0])
        real_name = ' '.join(row[1])

        name, attention = namegen.get_name_and_attention_for(source_tokens)
        suggestions.append([source_tokens, attention, name, real_name])

    suggestions = pd.DataFrame(suggestions, columns=['Source', 'Attention', 'GeneratedName', 'RealName'])
    suggestions.to_csv('/Users/oleks/Desktop/suggestions.csv', sep='\t', index=False)
