import csv
import math
from random import randrange

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error as MAE, mean_absolute_error
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

#We used this method to get some idea for some best parameters for our models
#Train set consists of 80 percent of original training dataset and Test set is 20 percent of the original training dataset
#Which makes iq parameters somewhat more unreliable than it already is so we may need to tweak it

if __name__ == '__main__':
    train = pd.read_csv("dengue_features_train.csv")
    total_cases = pd.read_csv("dengue_labels_train.csv")["total_cases"]
    test = pd.read_csv("dengue_features_test.csv")
    total = pd.read_csv("dengue_labels_train.csv")

    train.fillna(method='ffill', inplace=True)
    test.fillna(method='ffill', inplace=True)

    change_to_celsius = ["reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k",
                         "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k"]
    for i in change_to_celsius:
        train[i] = train[i] - 273.15
        test[i] = test[i] - 273.15

    train_SJ = train.loc[train['city'] == 'sj']
    train_IQ = train.loc[train['city'] == 'iq']
    test_SJ = test.loc[test['city'] == 'sj']
    test_IQ = test.loc[test['city'] == 'iq']
    total_CASESJ = total.loc[total['city'] == 'sj']
    total_CASEIQ = total.loc[total['city'] == 'iq']

    train_SJ = train_SJ.drop(columns=['city', 'year', 'week_start_date'])
    train_IQ = train_IQ.drop(columns=['city', 'year', 'week_start_date'])
    test_SJ = test_SJ.drop(columns=['city', 'year', 'week_start_date'])
    test_IQ = test_IQ.drop(columns=['city', 'year', 'week_start_date'])
    total_CASESJ = total_CASESJ.drop(columns=['city', 'year', 'weekofyear'])
    total_CASEIQ = total_CASEIQ.drop(columns=['city', 'year', 'weekofyear'])

    total_CASEIQ = total_CASEIQ.assign(Lag_by_1_Week=total_CASEIQ['total_cases'].shift(-1))
    total_CASEIQ = total_CASEIQ.apply(lambda x: x.fillna(method='ffill'))
    total_CASEIQ['Lag_by_1_Week'] = total_CASEIQ['Lag_by_1_Week'].astype(int)

    lag_by_1_week_IQ = total_CASEIQ['Lag_by_1_Week']
    total_CASEIQ = total_CASEIQ.drop(columns='Lag_by_1_Week')

    total_CASESJ = total_CASESJ.assign(Lag_by_1_Week=total_CASESJ['total_cases'].shift(-1))
    total_CASESJ = total_CASESJ.apply(lambda x: x.fillna(method='ffill'))
    total_CASESJ['Lag_by_1_Week'] = total_CASESJ['Lag_by_1_Week'].astype(int)

    lag_by_1_week_SJ = total_CASESJ['Lag_by_1_Week']
    total_CASESJ = total_CASESJ.drop(columns='Lag_by_1_Week')

    total_CASEIQ = total_CASEIQ.assign(Lag_by_2_Week=total_CASEIQ['total_cases'].shift(-2))
    total_CASEIQ = total_CASEIQ.apply(lambda x: x.fillna(method='ffill'))
    total_CASEIQ['Lag_by_2_Week'] = total_CASEIQ['Lag_by_2_Week'].astype(int)

    lag_by_2_week_IQ = total_CASEIQ['Lag_by_2_Week']
    total_CASEIQ = total_CASEIQ.drop(columns='Lag_by_2_Week')

    total_CASESJ = total_CASESJ.assign(Lag_by_2_Week=total_CASESJ['total_cases'].shift(-2))
    total_CASESJ = total_CASESJ.apply(lambda x: x.fillna(method='ffill'))
    total_CASESJ['Lag_by_2_Week'] = total_CASESJ['Lag_by_2_Week'].astype(int)

    lag_by_2_week_SJ = total_CASESJ['Lag_by_2_Week']
    total_CASESJ = total_CASESJ.drop(columns='Lag_by_2_Week')

    minmax_scale = MinMaxScaler().fit(train_SJ)
    train_SJ2 = minmax_scale.transform(train_SJ)
    test_SJ2 = minmax_scale.transform(test_SJ)

    train_SJ = pd.DataFrame(train_SJ2, columns=train_SJ.columns)
    test_SJ = pd.DataFrame(test_SJ2, columns=test_SJ.columns)

    train_SJ_copy = train_SJ.copy()
    train_SJ_copy["total_cases"] = total_CASESJ.iloc[:, 0]
    corrmatSJ = train_SJ_copy.corr()
    top_corr_featuresSJ = corrmatSJ.index
    df_corrSJ = train_SJ_copy[top_corr_featuresSJ].corr()

    for feature in df_corrSJ:
        if feature == 'total_cases' or feature == 'weekofyear' or feature == "reanalysis_specific_humidity_g_per_kg" or feature == "reanalysis_dew_point_temp_k" or feature == "station_avg_temp_c" or feature == "station_min_temp_c":
            continue
        if math.isnan(df_corrSJ[feature]['total_cases']) or math.fabs(df_corrSJ[feature]['total_cases']) < 2:
            del train_SJ[feature]
            del test_SJ[feature]

    minmax_scale2 = MinMaxScaler().fit(train_IQ)
    train_IQ2 = minmax_scale2.transform(train_IQ)
    test_IQ2 = minmax_scale2.transform(test_IQ)

    train_IQ = pd.DataFrame(train_IQ2, columns=train_IQ.columns)
    test_IQ = pd.DataFrame(test_IQ2, columns=test_IQ.columns)

    total_CASEIQ.index = range(520)

    train_IQ_copy = train_IQ.copy()
    train_IQ_copy['total_cases'] = total_CASEIQ
    # train_IQ_copy['total_cases'] = train_IQ_copy['total_cases'].astype(int)
    corrmatIQ = train_IQ_copy.corr()
    top_corr_featuresIQ = corrmatIQ.index
    df_corrIQ = train_IQ_copy[top_corr_featuresIQ].corr()

    for feature in df_corrIQ:
        if feature == 'total_cases' or feature == 'weekofyear' or feature == "reanalysis_specific_humidity_g_per_kg" or feature == "reanalysis_dew_point_temp_k" or feature == "station_avg_temp_c" or feature == "station_min_temp_c":
            continue
        if math.isnan(df_corrIQ[feature]['total_cases']) or math.fabs(df_corrIQ[feature]['total_cases']) < 2:
            del train_IQ[feature]
            del test_IQ[feature]
    columnsSJ = []
    for index in list(train_SJ):
        if train_SJ[index].var() > 1000:
            columnsSJ.append(index)
    columnsIQ = []
    for index in list(train_IQ):
        if train_IQ[index].var() > 1000:
            columnsIQ.append(index)
    for element in columnsSJ:
        del train_SJ[element]
        del test_SJ[element]
    for element in columnsIQ:
        del train_IQ[element]
        del test_IQ[element]

    paramgrid = {
        "max_depth": range(1, 10),
        "criterion": ["mae"]
    }
    paramgrid2 = {
        "max_depth": range(1, 10),
        "criterion": ["mae"]
    }

    rtreeForSJ = EvolutionaryAlgorithmSearchCV(
        estimator=RandomForestRegressor(),
        params=paramgrid,
        scoring="neg_mean_absolute_error",
        cv=StratifiedKFold(n_splits=4),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        gene_crossover_prob=0.5,
        tournament_size=3,
        generations_number=5,
        n_jobs=1
    )
    rtreeForIQ = EvolutionaryAlgorithmSearchCV(
        estimator=RandomForestRegressor(),
        params=paramgrid2,
        scoring="neg_mean_absolute_error",
        cv=StratifiedKFold(n_splits=4),
        verbose=1,
        population_size=50,
        gene_mutation_prob=0.10,
        gene_crossover_prob=0.5,
        tournament_size=3,
        generations_number=5,
        n_jobs=1
    )
    # rtreeForSJ = RandomForestRegressor(n_estimators=1000, max_depth=5, criterion="mae")
    # rtreeForIQ = RandomForestRegressor(n_estimators=1000, max_depth=5, criterion="mae")

    train_SJ_train, train_SJ_test = train_test_split(train_SJ, test_size=0.2, random_state=24)
    total_CASESJ_train, total_CASESJ_test = train_test_split(total_CASESJ, test_size=0.2, random_state=24)

    train_IQ_train, train_IQ_test = train_test_split(train_IQ, test_size=0.2, random_state=24)
    total_CASEIQ_train, total_CASEIQ_test = train_test_split(total_CASEIQ, test_size=0.2, random_state=24)

    rtreeForSJ.fit(train_SJ, total_CASESJ)
    rtreeForIQ.fit(train_IQ, total_CASEIQ)

    predictionsSJ = rtreeForSJ.predict(train_SJ_test)
    predictionsIQ = rtreeForIQ.predict(train_IQ_test)
    sjscore = mean_absolute_error(total_CASESJ_test, predictionsSJ)
    iqscore = mean_absolute_error(total_CASEIQ_test, predictionsIQ)
    print(sjscore)
    print(iqscore)
    # print(len(predictionsSJ)+len(predictionsIQ))
    # print(len(predictionsIQ))
    # finalArr = []
    # for k in predictionsSJ:
    #     finalArr.append(k)
    # for t in predictionsIQ:
    #     finalArr.append(t)
    # submission = pd.read_csv('submission_format.csv')
    # # print(finalArr[2])
    # # for i in range(len(submission['total_cases'])):
    # #     submission['total_cases'][i]=finalArr[i]
    # # submission.drop(columns=['city','year','weekofyear'])
    # # submission.to_csv("output.csv")
    # for i in range(len(finalArr)):
    #     finalArr[i] = int(finalArr[i])
    # del submission['total_cases']
    # submission['total_cases'] = finalArr
    # submission.to_csv("output.csv", index=False)
