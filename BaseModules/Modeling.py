# -*- coding:utf-8 -*-

# The functions of this module include:
#    1.create the following classification algorithms' estimators:
#       (1). logistic regression
#       (2). gradient boosting machine
#       (3). naive bayes
#       (4). random forest
#       (5). deep learning
#       (6). stacking
#       (7). k means
#    2. get a specific estimator from above
#    3. create a grid search estimator
#    4. conduct training process
#    5. select the best model of grid search
#    6. conduct prediction process
#    7. conduct test process
#

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch

algorithm_aka = {
    'DL': 'deep_learning',
    'LR': 'logistic_regression',
    'GBM': 'gradient_boosting_machine',
    'NB': 'naive_bayes',
    'RF': 'random_forest',
    'Stacking': 'stacking',
    'KM': 'k_means'
}


def deep_learning(xval=None, sample_size=None, nfolds=None, hparams=None, for_stacking=None):
    """
    create a deep learning algorithm estimator
    :param xval: if for cross-validation
    :param sample_size: training set sample amount
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :param for_stacking: if it is used for stacking
    :return: a constructed deep learning estimator, a parameters' dict for grid search
    """

    if sample_size <= 10000:
        default_nfolds = 3
        hidden_opts = [[30, 30], [20, 20], [10, 10]]
        input_dropout_ratio_opts = [0, 0.05, 0.1]
        l1_opts = [0, 1e-4, 1e-6]
        l2_opts = [0, 1e-4, 1e-6]

    elif 10000 < sample_size <= 100000:
        default_nfolds = 3
        hidden_opts = [[20, 20], [30, 30]]
        input_dropout_ratio_opts = [0, 0.05]
        l1_opts = [0, 1e-6]
        l2_opts = [0, 1e-6]

    else:
        if sample_size > 500000:
            default_nfolds = 1
        else:
            default_nfolds = 2
        hidden_opts = [[20, 20], [10, 10]]
        input_dropout_ratio_opts = [0, 0.05]
        l1_opts = [1e-6]
        l2_opts = [1e-6]

    default_hparams = dict({'hidden': hidden_opts,
                            'input_dropout_ratio': input_dropout_ratio_opts,
                            'l1': l1_opts,
                            'l2': l2_opts})

    if nfolds is None:
        nfolds = default_nfolds
    if hparams is None:
        hparams = default_hparams

    if xval:
        if for_stacking:
            dl_estimator = H2ODeepLearningEstimator(nfolds=nfolds, fold_assignment="Modulo",
                                                    seed=1, keep_cross_validation_predictions=True,
                                                    shuffle_training_data=True)
        else:
            dl_estimator = H2ODeepLearningEstimator(nfolds=nfolds, shuffle_training_data=True)
    else:
        dl_estimator = H2ODeepLearningEstimator(shuffle_training_data=True)

    return dl_estimator, hparams


def logistic_regression(xval=None, sample_size=None, nfolds=None, hparams=None, for_stacking=None):
    """
    create a logistic regression algorithm estimator
    
    Note:
     1. standardize: True(default)
     3. missing_values_handling: mean_imputation(default)

    :param xval: if for cross-validation
    :param sample_size: training set sample amount
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :param for_stacking: if it is used for stacking
    :return: a constructed logistic regression estimator, a parameters' dict for grid search
    """
    if sample_size <= 10000:
        if sample_size <= 5000:
            default_nfolds = 3
        else:
            default_nfolds = 5
        alpha_opts = [0, 0.25, 0.5, 0.75, 1]
        lambda_opts = [1, 0.5, 0.1, 0.01, 0]

    elif 10000 < sample_size <= 100000:
        default_nfolds = 3
        alpha_opts = [0, 0.5, 1]
        lambda_opts = [1, 0.5, 0.1, 0.01, 0]

    else:
        default_nfolds = 2
        alpha_opts = [0, 0.5, 1]
        lambda_opts = [1, 0.5, 0.1, 0]

    default_hparams = dict({'alpha': alpha_opts, 'lambda': lambda_opts})

    if nfolds is None:
        nfolds = default_nfolds
    if hparams is None:
        hparams = default_hparams

    if xval:
        if for_stacking:
            lr_estimator = H2OGeneralizedLinearEstimator(family="binomial",
                                                         remove_collinear_columns=True,
                                                         max_iterations=50,
                                                         nfolds=nfolds,
                                                         fold_assignment="Modulo",
                                                         seed=1,
                                                         keep_cross_validation_predictions=True)
        else:
            lr_estimator = H2OGeneralizedLinearEstimator(family="binomial",
                                                         remove_collinear_columns=True,
                                                         max_iterations=50,
                                                         nfolds=nfolds)
    else:
        lr_estimator = H2OGeneralizedLinearEstimator(family="binomial",
                                                     remove_collinear_columns=True,
                                                     max_iterations=50)

    return lr_estimator, hparams


def gradient_boosting_machine(xval=None, sample_size=None, nfolds=None, hparams=None, for_stacking=None):
    """
    create a gradient boosting machine algorithm estimator
    :param xval: if for cross-validation
    :param sample_size: training set sample amount
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :param for_stacking: if it is used for stacking
    :return: a constructed gradient boosting machine estimator, a parameters' dict for grid search
    """

    if sample_size <= 10000:
        default_nfolds = 3
        ntrees_opts = [50, 70]
        max_depth_opts = [5, 7]
        learn_rate_opts = [0.1, 0.01]
        min_rows_opts = [5, 10]
        sample_rate_opts = [1]
        col_sample_rate_per_tree_opts = [0.8]
        col_sample_rate_opts = [1]

    elif 10000 < sample_size <= 100000:
        default_nfolds = 3
        ntrees_opts = [50]
        max_depth_opts = [5, 7]
        learn_rate_opts = [0.1]
        min_rows_opts = [5, 10]
        sample_rate_opts = [1]
        col_sample_rate_per_tree_opts = [0.8]
        col_sample_rate_opts = [1]

    else:
        if sample_size > 500000:
            default_nfolds = 1
        else:
            default_nfolds = 2
        ntrees_opts = [50]
        max_depth_opts = [5, 7]
        learn_rate_opts = [0.1]
        min_rows_opts = [5, 10]
        sample_rate_opts = [0.8]
        col_sample_rate_per_tree_opts = [0.8]
        col_sample_rate_opts = [1]

    default_hparams = dict({'ntrees': ntrees_opts,
                            'max_depth': max_depth_opts,
                            'learn_rate': learn_rate_opts,
                            'min_rows': min_rows_opts,
                            'sample_rate': sample_rate_opts,
                            'col_sample_rate_per_tree': col_sample_rate_per_tree_opts,
                            'col_sample_rate': col_sample_rate_opts})

    if nfolds is None:
        nfolds = default_nfolds
    if hparams is None:
        hparams = default_hparams

    if xval:
        if for_stacking:
            gbm_estimator = H2OGradientBoostingEstimator(nfolds=nfolds,
                                                         fold_assignment="Modulo",
                                                         seed=1,
                                                         keep_cross_validation_predictions=True)
        else:
            gbm_estimator = H2OGradientBoostingEstimator(nfolds=nfolds)
    else:
        gbm_estimator = H2OGradientBoostingEstimator()

    return gbm_estimator, hparams


def naive_bayes(xval=None, sample_size=None, nfolds=None, hparams=None, for_stacking=None):
    """
    create a naive bayes algorithm estimator
    :param xval: if for cross-validation
    :param sample_size: training set sample amount
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :param for_stacking: if it is used for stacking
    :return: a constructed naive bayes estimator, a parameters' dict for grid search
    """
    if sample_size <= 50000:
        if sample_size <= 10000:
            default_nfolds = 3
        else:
            default_nfolds = 5
        laplace_opts = [0.1, 1, 5, 10]
        min_sdev_opts = [0.001, 0.005, 0.1]
        eps_sdev_opts = [0, 0.001, 0.01]

    elif 50000 < sample_size <= 500000:
        default_nfolds = 3
        laplace_opts = [0.1, 1, 5]
        min_sdev_opts = [0.001, 0.1]
        eps_sdev_opts = [0, 0.01]

    else:
        default_nfolds = 2
        laplace_opts = [0.1, 5]
        min_sdev_opts = [0.001, 0.005]
        eps_sdev_opts = [0, 0.01]

    default_hparams = dict({'laplace': laplace_opts,
                            'min_sdev': min_sdev_opts,
                            'eps_sdev': eps_sdev_opts})

    if nfolds is None:
        nfolds = default_nfolds
    if hparams is None:
        hparams = default_hparams

    if xval:
        if for_stacking:
            nb_estimator = H2ONaiveBayesEstimator(nfolds=nfolds,
                                                  fold_assignment="Modulo",
                                                  seed=1,
                                                  keep_cross_validation_predictions=True)
        else:
            nb_estimator = H2ONaiveBayesEstimator(nfolds=nfolds)
    else:
        nb_estimator = H2ONaiveBayesEstimator()

    return nb_estimator, hparams


def random_forest(xval=None, sample_size=None, nfolds=None, hparams=None, for_stacking=None):
    """
    create a random forest algorithm estimator
    :param xval: if for cross-validation
    :param sample_size: training set sample amount
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :param for_stacking: if it is used for stacking
    :return: a constructed random forest estimator, a parameters' dict for grid search
    """

    if sample_size <= 10000:
        default_nfolds = 3
        ntrees_opts = [50, 70]
        max_depth_opts = [5, 15]
        sample_rate_opts = [0.6, 0.8]
        min_rows_opts = [5, 10]
        col_sample_rate_per_tree_opts = [1]

    elif 10000 < sample_size <= 100000:
        default_nfolds = 3
        ntrees_opts = [50, 70]
        max_depth_opts = [5, 10]
        sample_rate_opts = [0.6]
        min_rows_opts = [5, 10]
        col_sample_rate_per_tree_opts = [1]

    else:
        default_nfolds = 2
        ntrees_opts = [50]
        max_depth_opts = [5, 10]
        sample_rate_opts = [0.6]
        min_rows_opts = [5, 10]
        col_sample_rate_per_tree_opts = [1]

    default_hparams = dict({'ntrees': ntrees_opts,
                            'max_depth': max_depth_opts,
                            'sample_rate': sample_rate_opts,
                            'min_rows': min_rows_opts,
                            'col_sample_rate_per_tree': col_sample_rate_per_tree_opts})

    if nfolds is None:
        nfolds = default_nfolds
    if hparams is None:
        hparams = default_hparams

    if xval:
        if for_stacking:
            rf_estimator = H2ORandomForestEstimator(nfolds=nfolds,
                                                    fold_assignment="Modulo",
                                                    seed=1,
                                                    keep_cross_validation_predictions=True)
        else:
            rf_estimator = H2ORandomForestEstimator(nfolds=nfolds, seed=1)
    else:
        rf_estimator = H2ORandomForestEstimator()

    return rf_estimator, hparams


def stacking(trained_base_model_lst):
    """
    create a stacking algorithm estimator
    :param trained_base_model_lst: a list of base algorithm models' id
    :return: a constructed stacking estimator
    """
    return H2OStackedEnsembleEstimator(model_id="stacking_model", base_models=trained_base_model_lst)


def k_means(xval=None, sample_size=None, nfolds=None, hparams=None):
    """
    create a k-means algorithm estimator
    :param xval: if for cross-validation
    :param sample_size: training set sample amount
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :return: a constructed k-means estimator, a parameters' dict for grid search 
    """

    if sample_size <= 10000:
        if sample_size < 5000:
            default_nfolds = 3
        else:
            default_nfolds = 5
        k_opts = [3, 5, 10]
        max_iterations_opts = [5, 10, 20]
        standardize_opts = [0.1, 0.6, 0.8]

    elif 10000 < sample_size <= 100000:
        default_nfolds = 3
        k_opts = [3, 5, 10]
        max_iterations_opts = [5, 10, 20]
        standardize_opts = [0.1, 0.6]

    else:
        default_nfolds = 2
        k_opts = [3, 5, 10]
        max_iterations_opts = [5, 10]
        standardize_opts = [0.1, 0.6]

    default_hparams = dict({'k': k_opts,
                            'max_iterations': max_iterations_opts,
                            'standardize': standardize_opts})

    if nfolds is None:
        nfolds = default_nfolds
    if hparams is None:
        hparams = default_hparams

    if xval:
        km_estimator = H2OKMeansEstimator(nfolds=nfolds)
    else:
        km_estimator = H2OKMeansEstimator()

    return km_estimator, hparams


def grid_search(estimator, hparams=None, search_criteria=None):
    """
    create a grid search estimator
    :param estimator: a specific estimator for grid search
    :param hparams: a hyper parameters dict for grid search
    :param search_criteria: criteria for grid search 
    :return: a constructed grid search estimator
    """
    print estimator
    print hparams
    return H2OGridSearch(model=estimator, hyper_params=hparams, search_criteria=search_criteria)


def get_gridsearch_best(trained_gs_model, metric='auc', decreasing=True):
    """
    get the best model after grid search
    :param trained_gs_model: a trained grid search model
    :param metric: a metric based on which to choose thebest model
    :param decreasing: sort grid search model by the metric score decresing or incresing
    :return: the best trained model
    """
    return trained_gs_model.get_grid(sort_by=metric, decreasing=decreasing)[0]


def training(model, x=None, y=None, train_data=None, valid_data=None):
    """
    conduct training process
    :param model: a model for training
    :param x: x columns' name (names of predictors)
    :param y: y column name (name of the response)
    :param train_data: train data frame
    :param valid_data: validation data frame
    :return: a trained model
    """
    print '\n\n--train--**\n\n'
    model.train(x=x, y=y, training_frame=train_data, validation_frame=valid_data)
    return model


def get_algorithm_estimator(algo_aka, sample_size=None, xval=None, nfolds=None, hparams=None,
                            for_stacking=None, model_lst=None):
    """
    get a specific algorithm function name
    :param algo_aka: an algorithm's name for short
    :param sample_size: training set sample amount
    :param xval: if for cross-validation
    :param nfolds: k value for k-fold cross-validation
    :param hparams: hyper parameters for grid search
    :param for_stacking: if it is for stacking
    :return: a specific constructed estimator
    """
    if algo_aka == 'Stacking':
        return eval(algorithm_aka[algo_aka])(model_lst)
    elif algo_aka == 'KM':
        return eval(algorithm_aka[algo_aka])(xval=xval, sample_size=sample_size, nfolds=nfolds, hparams=hparams)
    else:
        return eval(algorithm_aka[algo_aka])(xval=xval, sample_size=sample_size, nfolds=nfolds,
                                             hparams=hparams, for_stacking=for_stacking)


def save_model(model, path, model_type=None):
    """
    save model
    :param model: a model to be saved
    :param path: dir to save model
    :return: saveed path ( dir + saved filename)
    """
    if model_type == 'mojo':
        return model.save_mojo(path=path, force=True)
    else:
        return h2o.save_model(model, path)


# =====================
# prediction
# =====================


def load_model(path):
    """
    load model
    :param path: model path
    :return: a loaded model
    """
    return h2o.load_model(path)


def prediction(trained_model, test_data):
    """
    conduct predicting process
    :param trained_model: a trained model
    :param test_data: test data for prediction
    :return: a data frame of prediction result
    """
    return trained_model.predict(test_data)


# =====================
# test
# =====================


def test_perfomance(trained_model, test_data):
    """
    conduct testing process
    :param trained_model: a trained model
    :param test_data: test data for prediction
    :return: an object of class H2OModelMetrics
    """
    return trained_model.model_performance(test_data)
