"""
# offline score
# 0.9513713333333333+0.5295316666666666 = 1.480903
"""
import os
import gc
import sys
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
import sklearn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(BASE_DIR)


print(sklearn.__version__)  # '0.23.1'
print(np.__version__)  # '1.18.1'
print(pd.__version__)  # '1.0.4'


def save_sub(df_raw_sub):
    df_raw_sub.to_csv(os.path.join(BASE_DIR,'submission.csv'), index=False, encoding='utf-8')


def separate_age_gender_prob(prob):
    prob_age = prob[:, :10] + prob[:, 10:]
    prob_gender = np.concatenate((prob[:, :10].sum(
        axis=1, keepdims=True), prob[:, 10:].sum(axis=1, keepdims=True)),
                                 axis=1)
    # pred_age = np.argmax(prob_age, axis=1) + 1
    # pred_gender = np.argmax(prob_gender, axis=1) + 1
    prob_age = np.float32(prob_age)
    prob_gender = np.float32(prob_gender)
    return prob_age, prob_gender


def get_pi_col(i):
    """
    df_prob_all[get_pi_col(1)].corr()
    """
    col2calc = []
    for col in collist:
        col2calc.append(col + '_p' + str(i))
    return col2calc


def get_df(all_probs):
    all_col_nm = []
    for col in collist:
        for i in range(1, all_probs[0].shape[1] + 1):
            all_col_nm.append(col + '_p' +
                              str(i))  # 所有模型 所有的 列名 age 每个模型10列，gender2列
    all_concated = np.concatenate(all_probs, axis=1)
    df_prob = pd.DataFrame(all_concated, columns=all_col_nm)
    print(df_prob.shape)
    df_prob = df_prob.astype('float32')
    return df_prob


if __name__ == "__main__":

#     BASE_DIR = "../oof/BANJITINO_PROB/"
    BASE_DIR = os.path.join(BASE_DIR,"oof/BANJITINO_PROB/")
    ZQ_BASE_DIR = os.path.join(BASE_DIR, "zqoof/")
    AZ_BASE_DIR = os.path.join(BASE_DIR, "high_score_models/")
    BK_BASE_DIR = os.path.join(BASE_DIR, "beike_best_model/")

    file_zoos = [
        ZQ_BASE_DIR + "1475a/model20.npy",
        BK_BASE_DIR + "merge_bkmodel3_2_14765/model20.npy",
        ZQ_BASE_DIR + "14743/model20.npy", ZQ_BASE_DIR + "1475c/model20.npy",
        ZQ_BASE_DIR + "1475c/model20.npy", AZ_BASE_DIR + "model_dnn_1475.npy",
        AZ_BASE_DIR + "model_lstm_1475.npy", BK_BASE_DIR +
        "merge_bkmodel4_4_1475/model20.npy", ZQ_BASE_DIR + "1475d/model20.npy",
        ZQ_BASE_DIR + "14734/model20.npy", ZQ_BASE_DIR + "14727/model20.npy",
        AZ_BASE_DIR + "model_transformer_14745.npy",
        ZQ_BASE_DIR + "1474b/model20.npy", ZQ_BASE_DIR + "1474a/model20.npy",
        ZQ_BASE_DIR + "14722b/model20.npy", ZQ_BASE_DIR + "14720/model20.npy",
        BK_BASE_DIR + "merge_bkmodel1_1_14731/model20.npy",
        BK_BASE_DIR + "merge_bkmodel2_1_147296/model20.npy",
        ZQ_BASE_DIR + "14722/model20.npy", ZQ_BASE_DIR + "1474c/model20.npy",
        ZQ_BASE_DIR + "1474d/model20.npy"
    ]

    all_probs = []
    for file in file_zoos:
        np_prob = np.load(file)
        print(np.mean(np.sum(np_prob, axis=1)))
        all_probs.append(np_prob)
    len_model = len(all_probs)
    collist = []
    for i in range(1, len_model + 1):
        collist.append(f'model_{i}')
    # 找个base user_id!
    sub_base = pd.DataFrame([i for i in range(1, 4000001)],
                            columns=['user_id'])
    sub_base = sub_base.sort_values('user_id').reset_index(drop=True)

    all_probs_age = []
    all_probs_gender = []
    for i_prob in all_probs:
        prob_age, prob_gender = separate_age_gender_prob(i_prob)
        all_probs_age.append(prob_age)
        all_probs_gender.append(prob_gender)

    del all_probs
    gc.collect()

    df_prob_age = get_df(all_probs_age)
    df_prob_gender = get_df(all_probs_gender)

    datalabel = pd.read_pickle(os.path.join(BASE_DIR, "datalabel.pkl"))
    traindata = datalabel.loc[~datalabel['age'].isna()].reset_index(drop=True)
    testdata = datalabel.loc[datalabel['age'].isna()].copy().reset_index(
        drop=True)
    traindata['age'] = traindata['age'].astype(np.int8)
    traindata['gender'] = traindata['gender'].astype(np.int8)
    y_age = traindata['age'].values
    y_gender = traindata['gender'].values

    del datalabel
    gc.collect()

    # stack age
    skf = StratifiedKFold(n_splits=5, random_state=1111, shuffle=True)
    folds = list(skf.split(traindata, traindata['age']))

    train_val = df_prob_age.iloc[:3000000]
    test = df_prob_age.iloc[3000000:]

    age_model_lst = []
    age_score_lst = []
    age_oof = np.zeros((3000000, 10), dtype='float32')
    for count, (train_idx, valid_idx) in enumerate(folds):
        X_train = train_val.iloc[train_idx].values
        X_val = train_val.iloc[valid_idx].values
        y_train = y_age[train_idx]
        y_val = y_age[valid_idx]
        print(f'Training Fold {count}...')
        model = RidgeClassifier(alpha=0.5)
        #     model = LogisticRegression(n_jobs=30)
        model.fit(X_train, y_train)
        try:
            val_pred_prob = model.predict_proba(X_val)
            age_oof[valid_idx] = val_pred_prob
            val_pred = np.argmax(val_pred_prob,
                                 axis=1)  # model.predict(X_val) #
        except:
            print(model.coef_)
            val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        print(f"Fold-{count}: Acc =", acc)
        age_score_lst.append(acc)
        age_model_lst.append(model)
    print(np.mean(age_score_lst))

    test = df_prob_age.iloc[3000000:]
    age_pred = pd.DataFrame()
    age_pred['user_id'] = testdata['user_id'].copy()
    for i, mi in enumerate(age_model_lst, 1):
        age_pred[f'pred_{i}'] = mi.predict(test)

    age_pred['predicted_age_mean'] = age_pred[[
        'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'
    ]].mean(axis=1)
    age_pred['predicted_age_std'] = age_pred[[
        'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'
    ]].std(axis=1)
    age_pred.loc[age_pred["predicted_age_std"] != 0, 'mode'] = age_pred.loc[
        age_pred["predicted_age_std"] != 0,
        ['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].mode(axis=1)[0]
    age_pred['predicted_age'] = np.where(age_pred['mode'].isnull(),
                                         age_pred['predicted_age_mean'],
                                         age_pred['mode']) + 1
    df_pred = age_pred[['user_id', 'predicted_age']]

    # gender
    skf = StratifiedKFold(n_splits=5, random_state=1111, shuffle=True)
    folds = list(skf.split(traindata, traindata['gender']))
    train_val = df_prob_gender.iloc[:3000000]
    test = df_prob_gender.iloc[3000000:]
    gender_model_lst = []
    gender_score_lst = []
    gender_oof = np.zeros((3000000, 2), dtype='float32')
    for count, (train_idx, valid_idx) in enumerate(folds):
        X_train = train_val.iloc[train_idx].values
        y_train = y_gender[train_idx]
        X_val = train_val.iloc[valid_idx].values
        y_val = y_gender[valid_idx]
        print(f'Training Fold {count}...')
        model = RidgeClassifier(alpha=0.5)
        #     model = LogisticRegression(n_jobs=30)
        model.fit(X_train, y_train)
        try:
            val_pred_prob = model.predict_proba(X_val)
            gender_oof[valid_idx] = val_pred_prob
            val_pred = np.argmax(val_pred_prob,
                                 axis=1)  # model.predict(X_val) #
        except:
            print(model.coef_)
            val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        print(f"Fold-{count}: Acc =", acc)
        gender_score_lst.append(acc)
        gender_model_lst.append(model)

    print(np.mean(gender_score_lst))

    test = df_prob_gender.iloc[3000000:]
    gender_pred = pd.DataFrame()
    gender_pred['user_id'] = testdata['user_id'].copy()
    for i, mi in enumerate(gender_model_lst, 1):
        gender_pred[f'pred_{i}'] = mi.predict(test)
    gender_pred['gender_pred'] = gender_pred[[
        'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'
    ]].mean(axis=1)
    gender_pred['predicted_gender'] = gender_pred['gender_pred'].apply(
        lambda x: 1 if x >= 0.5 else 0)
    df_pred["predicted_gender"] = gender_pred['predicted_gender'] + 1
    df_pred['predicted_age'] = df_pred['predicted_age'].astype('int8')
    df_pred['predicted_gender'] = df_pred['predicted_gender'].astype('int8')
    save_sub(df_pred)

    # offline score
    # 0.9513713333333333+0.5295316666666666 = 1.480903
