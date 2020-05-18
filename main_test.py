from sklearn.datasets import load_boston
from pyspark.sql import SparkSession
from feature_ranker_modules import correlation_matrix, feature_ranker
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor, GeneralizedLinearRegression
from feature_selector_modules import feature_selector

spark = SparkSession.Builder().appName("test").getOrCreate()

# %% Loading the Boston Dataset as a sample dataset and creating the spark dataframes

boston = load_boston()
feature_names = boston.feature_names.tolist()
output_name = 'outcome'
boston_columns = feature_names + [output_name]
X = boston.data.tolist()
y = boston.target.tolist()
Xy = [(i + [j]) for (i, j) in zip(X, y)]
boston_df = spark.createDataFrame(Xy, boston_columns)

# %% Ranking features

ranked_features = feature_ranker(df=boston_df,
                                 feature_columns=feature_names,
                                 output_column=output_name)
print(ranked_features)

# %% Feature selection

scores = feature_selector(df=boston_df,
                          ranked_features=ranked_features,
                          output_column=output_name,
                          estimator_obj=RandomForestRegressor(),
                          feature_inclusion_increments=1,
                          train_test_split_ratio=[0.66, 0.33],
                          cv=1,
                          evaluation_metric='r2')
print(scores)

