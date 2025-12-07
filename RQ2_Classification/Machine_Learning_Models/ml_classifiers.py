import datetime
import warnings
from collections import Counter
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_recall_curve
import glo

from imbens.ensemble import SelfPacedEnsembleClassifier, SMOTEBoostClassifier, SMOTEBaggingClassifier, KmeansSMOTEBoostClassifier, BalancedRandomForestClassifier



SCORING = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'AUC': make_scorer(average_precision_score)}
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def calculate_scores(logger, method_name, method_info):
    tmp_key = method_info + "_" + method_name
    temp_predict = glo.get_val(tmp_key)
    res = "\n fold " + str(temp_predict[3]) + "\tAccuracyMean: " + str(temp_predict[2] / temp_predict[3]) \
          + "\tPrecisionMean: " + str(temp_predict[0] / temp_predict[3]) + "\t NegativePrecisionMean: " + str(
        temp_predict[1] / temp_predict[3]) \
          + "\tRecallMean: " + str(temp_predict[4] / temp_predict[3]) + "\t NegativeRecallMean: " + str(temp_predict[5] / temp_predict[3]) \
          + "\tNegativeF1Mean: " + str(temp_predict[6] / temp_predict[3]) + "\t F1Mean: " + str(temp_predict[7] / temp_predict[3]) \
          + "\tPR-AUCMean: " + str(temp_predict[8] / temp_predict[3]) + "\tROC-AUCMean: " + str(temp_predict[9] / temp_predict[3])
    print(res)
    logger.info(res)


def save_fold_results_to_json(filename="results_per_fold.json"):
    """保存每一轮的详细结果到JSON文件"""
    fold_results_key = "fold_results"
    fold_results = glo.get_val(fold_results_key)
    
    if fold_results is not None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(fold_results, f, indent=4, ensure_ascii=False)
        print(f"\n结果已保存到 {filename}")
    else:
        print("\n没有找到fold结果数据")


def evaluate_method(model, method_name, logger, test_x, test_y, method_info):
    logger.info("test data distribution: " + str(sorted(Counter(test_y).items())))


    # # 随机森林：0.55
    # # 决策树：0.57

    # # 假设 test_x 是你的测试集
    # y_proba = model.predict_proba(test_x)[:, 1]  # 取正类（类别1）的概率

    # # 手动设置阈值
    # threshold = 0.55  # 举例：你想更敏感，降低阈值以提高召回率

    # # 手动判断：概率 >= 阈值 → 预测为 1，否则为 0
    # y_pred = (y_proba >= threshold).astype(int)


    y_pred = model.predict(test_x)
    y_score = model.predict_proba(test_x)[:, 1]

    acc_res = accuracy_score(test_y, y_pred)
    tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel()
    recall_res = recall_score(test_y, y_pred, average=None)
    precision_res = precision_score(test_y, y_pred, average=None)
    f1_res = f1_score(test_y, y_pred, average=None)

    # PR-AUC
    aucScore = average_precision_score(y_true=test_y, y_score=y_score)
    
    # ROC-AUC
    rocAucScore = roc_auc_score(y_true=test_y, y_score=y_score)

    tmp_key = method_info + "_" + method_name
    if glo.get_val(tmp_key) is None:
        glo.set_val(tmp_key, [precision_res[1], precision_res[0], acc_res, 1, recall_res[1], recall_res[0], f1_res[0],
                                      f1_res[1], aucScore, rocAucScore])
        current_fold = 1
    else:
        temp_predict = glo.get_val(tmp_key)
        temp_predict[0] += precision_res[1]
        temp_predict[1] += precision_res[0]
        temp_predict[2] += acc_res
        temp_predict[3] += 1
        temp_predict[4] += recall_res[1]
        temp_predict[5] += recall_res[0]
        temp_predict[6] += f1_res[0]
        temp_predict[7] += f1_res[1]
        temp_predict[8] += aucScore
        temp_predict[9] += rocAucScore
        glo.set_val(tmp_key, temp_predict)
        current_fold = temp_predict[3]
    
    # 保存每一轮的详细结果
    fold_results_key = "fold_results"
    fold_results = glo.get_val(fold_results_key)
    if fold_results is None:
        fold_results = {}
        glo.set_val(fold_results_key, fold_results)
    
    if method_info not in fold_results:
        fold_results[method_info] = {}
    if method_name not in fold_results[method_info]:
        fold_results[method_info][method_name] = []
    
    # 记录当前fold的结果
    fold_result = {
        "fold": current_fold,
        "Precision": float(precision_res[1]),
        "Recall": float(recall_res[1]),
        "F1": float(f1_res[1]),
        "Accuracy": float(acc_res),
        "PR-AUC": float(aucScore),
        "ROC-AUC": float(rocAucScore),
        "Negative_Precision": float(precision_res[0]),
        "Negative_Recall": float(recall_res[0]),
        "Negative_F1": float(f1_res[0])
    }
    
    fold_results[method_info][method_name].append(fold_result)
    glo.set_val(fold_results_key, fold_results)


# Random Forest
def rf_classifier(train_x, train_y, logger):
    '''
    parameters = {
        'n_estimators': list(range(10, 110, 10))
    }
    model_g = RandomForestClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("rf_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    '''
    # 调参
    parameters = {
        'n_estimators': [10, 20, 30, 50, 100],
        'estimator__n_estimators': list(range(10, 110, 10)),      # RandomForest树的数量
    }
    model_g = RandomForestClassifier(random_state=3407)
    clf = SelfPacedEnsembleClassifier(random_state=3407, estimator=model_g)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)
    grid = GridSearchCV(clf, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("rf_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    model_g = RandomForestClassifier(random_state=3408, class_weight='balanced')
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=20)

    # model_g = RandomForestClassifier(random_state=3408)
    # clf = SMOTEBoostClassifier(random_state=3408, estimator=model_g)
    # clf = SMOTEBaggingClassifier(random_state=3408, estimator=model_g)
    # clf = KmeansSMOTEBoostClassifier(random_state=3408, estimator=model_g)

    # clf = BalancedRandomForestClassifier(random_state=3408, class_weight='balanced')
    # clf = RandomForestClassifier(random_state=3408)
    clf.fit(train_x, train_y)

    return clf


# Logistic Regression
def lr_classifier(train_x, train_y, logger):

    # 参数搜索
    '''
    parameters = {
        'C': np.linspace(0.0001, 20, 20),
        'solver': ["newton-cg", "lbfgs", "liblinear", "sag"],
        'dual': [False],
        'verbose': [False],
        'max_iter': [500]
    }
    model_g = LogisticRegression(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("lr_para: %s", grid.best_params_)
    best_model = grid.best_estimator_

    return best_model
    '''

    model_g = LogisticRegression(random_state=3408, class_weight='balanced')
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=30)
    clf.fit(train_x, train_y)

    return clf


# KNN
def knn_classifier(train_x, train_y, logger):
    # parameters = {
    #     'n_neighbors': np.arange(1, 11, 1),
    #     'algorithm': ['auto']
    # }
    # model_g = KNeighborsClassifier()
    # # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    # fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    # grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # logger.info("knn_para: %s" % (grid.best_params_))
    # best_model = grid.best_estimator_
    # return best_model

    model_g = KNeighborsClassifier()
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=10)
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)

    return clf


# Decision Tree
def dt_classifier(train_x, train_y, logger):

    '''
    parameters = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [5, 10, 15]
    }
    model_g = DecisionTreeClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("dt_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    model_g = DecisionTreeClassifier(random_state=3408, class_weight='balanced')
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=10)
    clf.fit(train_x, train_y)

    return clf


# Support Vector Machine
def svm_classifier(train_x, train_y, logger):
    '''
    parameters = {
        'C': [0.1, 1, 10, 50],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    model_g = SVC(random_state=3407, probability=True)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("svm_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    # model_g = SVC(random_state=3408, class_weight='balanced', probability=True)
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=10)
    clf = SVC(random_state=3408, class_weight='balanced', probability=True)
    clf.fit(train_x, train_y)

    return clf


# Gradient Boosting
def gb_classifier(train_x, train_y, logger):
    '''
    parameters = {"loss": ["log_loss"],
                 "learning_rate": [0.1],
                 "min_samples_split": np.linspace(0.1, 0.4, 3),
                 "min_samples_leaf": np.linspace(0.1, 0.3, 3),
                 "max_depth": [3, 8],
                 "max_features": ["sqrt"],
                 "criterion": ["friedman_mse"],
                 "subsample": [0.95],
                 "n_estimators": [50, 100, 150]
                 }
    model_g = GradientBoostingClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("gb_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    model_g = GradientBoostingClassifier(random_state=3408)
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    clf.fit(train_x, train_y)

    return clf


# Naive Bayes
def nb_classifier(train_x, train_y, logger):
    model_g = MultinomialNB()
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    clf.fit(train_x, train_y)
    
    return clf


# MLP (NN)
def mlp_classifier(train_x, train_y, logger):
    '''
    parameters = {
        "hidden_layer_sizes": [(50,), (50, 25), (25,)],
        "activation": ["relu", "logistic"],
        "solver": ['adam']
    }
    model_g = MLPClassifier(random_state=3407)
    # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    grid.fit(train_x, train_y)
    logger.info("mlp_para: %s" % (grid.best_params_))
    best_model = grid.best_estimator_
    return best_model
    '''

    model_g = MLPClassifier(random_state=3408)
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=10)
    clf.fit(train_x, train_y)

    return clf


# XGBoost
def xgb_classifier(train_x, train_y, logger):
    # parameters = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [3, 6],
    #     'learning_rate': [0.1, 0.2],
    #     'subsample': [0.9],
    #     'colsample_bytree': [0.9]
    # }
    # model_g = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=3407)
    # # fold = KFold(n_splits=10, random_state=5, shuffle=True)
    # fold = StratifiedKFold(n_splits=10, random_state=5, shuffle=True)
    # grid = GridSearchCV(model_g, parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # logger.info("xgb_para: %s" % (grid.best_params_))
    # best_model = grid.best_estimator_
    # return best_model


    model_g = XGBClassifier(random_state=3408, class_weight='balanced')
    clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g)
    # clf = SelfPacedEnsembleClassifier(random_state=3408, estimator=model_g, n_estimators=60)
    # clf = XGBClassifier(random_state=3408, class_weight='balanced')
    clf.fit(train_x, train_y)

    return clf


def methods_container(train_x, train_y, test_x, test_y, logger, class_type):

    # Standardization
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # Normalization
    scaler = MinMaxScaler()
    train_x_scaled1 = scaler.fit_transform(train_x_scaled)
    test_x_scaled1 = scaler.transform(test_x_scaled)

    # logger.info("==================Logistic Regression==============")
    # lr_model = lr_classifier(train_x_scaled, train_y, logger)
    # evaluate_method(lr_model, "lr", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "lr", class_type)

    logger.info("==================Random Forest==============")
    rf_model = rf_classifier(train_x_scaled, train_y, logger)
    evaluate_method(rf_model, "rf", logger, test_x_scaled, test_y, class_type)
    calculate_scores(logger, "rf", class_type)

    # logger.info("==================KNN================")
    # knn_model = knn_classifier(train_x_scaled1, train_y, logger)
    # evaluate_method(knn_model, "knn", logger, test_x_scaled1, test_y, class_type)
    # # knn_model = knn_classifier(train_x_scaled, train_y, logger)
    # # evaluate_method(knn_model, "knn", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "knn", class_type)
    
    # logger.info("==================Decision Tree================")
    # dt_model = dt_classifier(train_x_scaled, train_y, logger)
    # evaluate_method(dt_model, "dt", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "dt", class_type)
    
    # logger.info("==================Support Vector Machine================")
    # svm_model = svm_classifier(train_x_scaled, train_y, logger)
    # evaluate_method(svm_model, "svm", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "svm", class_type)

    # logger.info("==================Gradient Boosting================")
    # gb_model = gb_classifier(train_x_scaled, train_y, logger)
    # evaluate_method(gb_model, "gb", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "gb", class_type)

    # logger.info("==================Naive Bayes================")
    # nb_model = nb_classifier(train_x_scaled1, train_y, logger)
    # evaluate_method(nb_model, "nb", logger, test_x_scaled1, test_y, class_type)
    # # nb_model = nb_classifier(train_x_scaled, train_y, logger)
    # # evaluate_method(nb_model, "nb", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "nb", class_type)

    # logger.info("==================MLP================")
    # mlp_model = mlp_classifier(train_x_scaled, train_y, logger)
    # evaluate_method(mlp_model, "mlp", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "mlp", class_type)

    # logger.info("==================XGBoost================")
    # xgb_model = xgb_classifier(train_x_scaled, train_y, logger)
    # evaluate_method(xgb_model, "xgb", logger, test_x_scaled, test_y, class_type)
    # calculate_scores(logger, "xgb", class_type)
    
    # # 注意：在所有fold训练完成后，调用以下函数保存每轮结果到JSON
    # save_fold_results_to_json("results_per_fold.json")