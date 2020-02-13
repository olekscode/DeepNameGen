import metrics
from constants import SOS_TOKEN, EOS_TOKEN

from collections import OrderedDict

import pandas as pd


class Evaluator:
    def __init__(self, methods, tensor_encoder):
        self.methods = methods
        self.tensor_encoder = tensor_encoder

        self.metric_functions = {
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1': metrics.f1_score,
            'ExactMatch': metrics.exact_match
        }


    def evaluate(self, model):
        df = pd.DataFrame()
        df['Source'] = self.methods['source']
        df['RealName'] = self.methods['name']
        df['GeneratedName'] = self.methods['source'].apply(model)

        df = self.tensor_encoder.decode(df, 'Source', ['RealName', 'GeneratedName'])

        for metric, func in self.metric_functions.items():
            df[metric] = [
                func(row['RealName'], row['GeneratedName'])
                for i, row in df.iterrows()]

        df['Source'] = df['Source'].apply(lambda sentence: ' '.join(sentence))
        df['RealName'] = df['RealName'].apply(lambda sentence: ' '.join(sentence))
        df['GeneratedName'] = df['GeneratedName'].apply(lambda sentence: ' '.join(sentence))

        return df
