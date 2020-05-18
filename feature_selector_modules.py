from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler


def feature_selector(df: DataFrame, ranked_features: list, output_column: str, estimator_obj=RandomForestRegressor,
                     feature_inclusion_increments: int = 1, train_test_split_ratio: list = None, cv: int = -1,
                     evaluation_metric: str = 'r2'):
    """
    Trains the estimator at multiple steps, with features progressively added to the input list based on their ranks

    :param df: the input dataset with features and output as columns
    :param ranked_features: the output of the feature ranking algorithm or a manually selected ranking scheme
    :param output_column: the name of the output column in the dataset
    :param estimator_obj: the training model object
    :param train_test_split_ratio: the default for train_test_split_ratio is [0.66, 0.33]
    :param cv: if left as default (c = -1), changes nothing. If selected as a value > 1, it enforces cross validation
                and overrides the train-test-splitting
    :param feature_inclusion_increments:
    :param evaluation_metric: evaluation metric to return for predictions on test set - "rmse": root mean
            squared error - "mse": mean squared error - "r2" (default): coefficient of determination -
            "mae": mean absolute error
    """

    if train_test_split_ratio is None:
        train_test_split_ratio = [0.66, 0.33]

    feature_count_list = list(range(1, len(ranked_features), feature_inclusion_increments)) + [len(ranked_features)]

    estimator_features_col = 'features'
    while estimator_features_col in df.columns:
        estimator_features_col += '_'
    estimator_prediction_col = 'prediction'
    while estimator_prediction_col in df.columns:
        estimator_prediction_col += '_'
    estimator_obj.setFeaturesCol(estimator_features_col)
    estimator_obj.setPredictionCol(estimator_prediction_col)
    estimator_obj.setLabelCol(output_column)

    evaluator = RegressionEvaluator(
        labelCol=output_column, predictionCol=estimator_prediction_col, metricName=evaluation_metric)

    scores = []
    if cv <= 1:
        df_train, df_test = df.randomSplit(train_test_split_ratio)
        for feature_count in feature_count_list:
            input_features = ranked_features[0: feature_count]
            assembler = VectorAssembler(
                inputCols=input_features,
                outputCol=estimator_features_col)
            df_train = assembler.transform(df_train)
            fit_model = estimator_obj.fit(df_train)
            df_test = assembler.transform(df_test)
            df_test = fit_model.transform(df_test)
            score = evaluator.evaluate(df_test)
            scores.append((feature_count, score))
            df_train = df_train.drop(estimator_features_col)
            df_test = df_test.drop(estimator_features_col, estimator_prediction_col)
    else:
        for feature_count in feature_count_list:
            input_features = ranked_features[0: feature_count]
            assembler = VectorAssembler(
                inputCols=input_features,
                outputCol=estimator_features_col)
            df = assembler.transform(df)
            grid = ParamGridBuilder().addGrid(estimator_obj.featuresCol, [estimator_obj.getFeaturesCol()]).build()
            crossval = CrossValidator(estimator=estimator_obj,
                                      evaluator=evaluator,
                                      numFolds=cv,
                                      estimatorParamMaps=grid)
            fit_crossval = crossval.fit(df)
            scores.append((feature_count, fit_crossval.avgMetrics[0]))
            df = df.drop(estimator_features_col)

    return scores
