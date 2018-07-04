import datetime
from BaseModules.Modeling import *


def learn_train_single_classification_model(train_df, x, y, learn_conf_dict):
    """
    
    :return: 
    """
    algo = learn_conf_dict['train_conf']['algorithm']
    cv_k = learn_conf_dict['train_conf']['cv_k']
    hparams = learn_conf_dict['train_conf']['hparams']

    hparams = hparams if isinstance(hparams, dict) and len(hparams.keys()) > 0 else None
    xval = True
    valid = False
    valid_df = None
    if isinstance(cv_k, int) and cv_k < 2:
        xval = False
        valid = True
        train_df, valid_df = train_df.split_frame(ratios=[0.7])

    sample_size = train_df.shape[0]

    print '\n===== train %s algorithm =====' % algo
    start_time = datetime.datetime.now()
    print start_time

    estimator, hparams = get_algorithm_estimator(algo, sample_size=sample_size, xval=xval,
                                                 nfolds=cv_k, hparams=hparams)

    gs_model = grid_search(estimator, hparams)
    trained_gs_model = training(gs_model, x=x, y=y, train_data=train_df, valid_data=valid_df)
    trained_model = get_gridsearch_best(trained_gs_model, metric='auc')

    end_time = datetime.datetime.now()
    print end_time
    print 'running time: ' + str((end_time - start_time).seconds)
    print trained_model
    return trained_model, xval, valid


def learn_train_stacking(train_df, x, y, learn_conf_dict):
    """
    
    :return: 
    """
    cv_k = learn_conf_dict['train_conf']['cv_k']
    hparams = learn_conf_dict['train_conf']['hparams']

    xval = True

    train_df, valid_df = train_df.split_frame(ratios=[0.7])
    sample_size = train_df.shape[0]

    print '\n===== train stacking algorithm ====='
    start_time = datetime.datetime.now()
    print start_time

    model_id_lst = list()
    for algo in hparams.keys():
        sub_hparams = hparams[algo]
        estimator, sub_hparams = get_algorithm_estimator(algo_aka=algo, sample_size=sample_size,
                                                         xval=xval, nfolds=cv_k, hparams=sub_hparams,
                                                         for_stacking=True)
        gs_model = grid_search(estimator, sub_hparams)
        trained_gs_model = training(gs_model, x=x, y=y, train_data=train_df)
        trained_model = get_gridsearch_best(trained_gs_model, metric='auc')
        model_id_lst.append(trained_model.model_id)

    stacking_estimator = get_algorithm_estimator(algo_aka='Stacking',
                                                 model_lst=model_id_lst)
    print "=========================================="
    print stacking_estimator
    print "=========================================="
    stacking_trained_model = training(stacking_estimator, x=x, y=y, train_data=train_df, valid_data=valid_df)

    end_time = datetime.datetime.now()
    print end_time
    print 'running time: ' + str((end_time - start_time).seconds)
    print stacking_trained_model
    return stacking_trained_model


def learn_train_kmeans(train_df, x, learn_conf_dict):
    """
    
    :return: 
    """
    cv_k = learn_conf_dict['train_conf']['cv_k']
    hparams = learn_conf_dict['train_conf']['hparams']
    hparams = hparams if isinstance(hparams, dict) and len(hparams.keys()) > 0 else None
    xval = True
    valid = False
    valid_df = None
    if isinstance(cv_k, int) and cv_k < 2:
        xval = False
        valid = True
        train_df, valid_df = train_df.split_frame(ratios=[0.7])

    sample_size = train_df.shape[0]

    print '\n===== train kmeans algorithm ====='
    start_time = datetime.datetime.now()
    print start_time

    estimator, hparams = get_algorithm_estimator(algo_aka='KM', sample_size=sample_size, xval=xval,
                                                 nfolds=cv_k, hparams=hparams)
    gs_model = grid_search(estimator, hparams)
    trained_gs_model = training(gs_model, x=x, train_data=train_df, valid_data=valid_df)
    trained_model = get_gridsearch_best(trained_gs_model, metric='betweenss', decreasing=True)

    end_time = datetime.datetime.now()
    print end_time
    print 'running time: ' + str(end_time - start_time)
    print trained_model
    return trained_model, xval, valid
