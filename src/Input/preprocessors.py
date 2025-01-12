from sklearn.base import BaseEstimator, TransformerMixin


class TemporalVariableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()
        for feature in self.variables:
            X[feature] = self.reference_variable - X[feature]
        return X
