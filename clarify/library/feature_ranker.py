from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
import numpy as np


def correlation_matrix(df: DataFrame, feature_columns: list, output_column: str) -> np.ndarray:
    """
    generates the Pearson correlation coefficients matrix for feature ranking
    """

    vec_assembler = VectorAssembler(inputCols=feature_columns + [output_column], outputCol="all_columns")
    df_vectorized = vec_assembler.transform(df)
    cor_matrix = Correlation.corr(df_vectorized, "all_columns").collect()[0][0].toArray()
    return cor_matrix


def feature_ranker(df: DataFrame, feature_columns: list, output_column: str, must_include_features: list = None):
    """
    ranks features based on their correlation with the output and their inter-correlations
    """

    cor_matrix = abs(correlation_matrix(df, feature_columns, output_column))

    if must_include_features:
        must_include_feature_ids = [i for i in range(len(feature_columns)) \
                                    if feature_columns[i] in must_include_features]
        must_include_feature_scores = cor_matrix[must_include_feature_ids, -1]
        must_include_features_ids = [must_include_feature_ids[i] for i in np.argsort(must_include_feature_scores)[::-1]]
        ranked_feature_ids = must_include_feature_ids
        remaining_feature_ids = [i for i in range(len(feature_columns)) if i not in must_include_features_ids]
        remaining_feature_scores = []
        for feature_id in remaining_feature_ids:
            remaining_feature_scores.append(cor_matrix[feature_id, -1] - max([cor_matrix[feature_id, i] for i in
                                                                              ranked_feature_ids]))
    else:
        ranked_feature_ids = []
        remaining_feature_ids = range(len(feature_columns))
        remaining_feature_scores = cor_matrix[remaining_feature_ids, -1]

    while len(ranked_feature_ids) < len(feature_columns):
        best_feature_arg = np.argmax(remaining_feature_scores)
        best_feature_id = remaining_feature_ids[best_feature_arg]
        remaining_feature_ids = [remaining_feature_ids[i] for i in range(len(remaining_feature_ids)) if i !=
                                 best_feature_arg]
        ranked_feature_ids.append(best_feature_id)
        remaining_feature_scores = []
        for feature_id in remaining_feature_ids:
            remaining_feature_scores.append(cor_matrix[feature_id, -1] - max([cor_matrix[feature_id, i] for i in
                                                                              ranked_feature_ids]))
    ranked_features = [feature_columns[i] for i in ranked_feature_ids]
    return ranked_features
