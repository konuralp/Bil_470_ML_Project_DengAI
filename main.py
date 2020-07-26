import math
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


train = pd.read_csv("dengue_features_train.csv")
total_cases = pd.read_csv("dengue_labels_train.csv")["total_cases"]
test = pd.read_csv("dengue_features_test.csv")
total= pd.read_csv("dengue_labels_train.csv")

train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)

change_to_celsius = ["reanalysis_air_temp_k", "reanalysis_avg_temp_k","reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k","reanalysis_min_air_temp_k"]
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

train_SJ = pd.DataFrame(train_SJ2, columns = train_SJ.columns)
test_SJ = pd.DataFrame(test_SJ2, columns = test_SJ.columns)

train_SJ_copy = train_SJ.copy()
train_SJ_copyauto = train_SJ.copy()
test_SJ_copyauto = test_SJ.copy()
train_SJ_copy["total_cases"] = total_CASESJ.iloc[:, 0]
corrmatSJ = train_SJ_copy.corr()
top_corr_featuresSJ = corrmatSJ.index
df_corrSJ = train_SJ_copy[top_corr_featuresSJ].corr()

for feature in df_corrSJ:
    if feature == 'total_cases' or feature == 'weekofyear' or feature =="reanalysis_specific_humidity_g_per_kg" or feature=="reanalysis_dew_point_temp_k" or feature=="station_avg_temp_c" or feature=="station_min_temp_c":
        continue
    if math.isnan(df_corrSJ[feature]['total_cases']) or math.fabs(df_corrSJ[feature]['total_cases']) < 2:
        del train_SJ[feature]
        del test_SJ[feature]

for feature in df_corrSJ:
    if feature == 'total_cases':
        continue
    if math.isnan(df_corrSJ[feature]['total_cases']) or math.fabs(df_corrSJ[feature]['total_cases']) < 0.1:
        del train_SJ_copyauto[feature]
        del test_SJ_copyauto[feature]


minmax_scale2 = MinMaxScaler().fit(train_IQ)
train_IQ2 = minmax_scale2.transform(train_IQ)
test_IQ2 = minmax_scale2.transform(test_IQ)

train_IQ = pd.DataFrame(train_IQ2, columns = train_IQ.columns)
test_IQ = pd.DataFrame(test_IQ2, columns = test_IQ.columns)

total_CASEIQ.index = range(520)

train_IQ_copy = train_IQ.copy()
train_IQ_copyauto = train_IQ.copy()
test_IQ_copyauto = test_IQ.copy()
train_IQ_copy['total_cases'] = total_CASEIQ
#train_IQ_copy['total_cases'] = train_IQ_copy['total_cases'].astype(int)
corrmatIQ = train_IQ_copy.corr()
top_corr_featuresIQ = corrmatIQ.index
df_corrIQ = train_IQ_copy[top_corr_featuresIQ].corr()


for feature in df_corrIQ:
    if feature == 'total_cases' or feature == 'weekofyear' or feature =="reanalysis_specific_humidity_g_per_kg" or feature=="reanalysis_dew_point_temp_k" or feature=="station_avg_temp_c" or feature=="station_min_temp_c":
        continue
    if math.isnan(df_corrIQ[feature]['total_cases']) or math.fabs(df_corrIQ[feature]['total_cases']) < 2:
        del train_IQ[feature]
        del test_IQ[feature]

for feature in df_corrIQ:
    if feature == 'total_cases':
        continue
    if math.isnan(df_corrIQ[feature]['total_cases']) or math.fabs(df_corrIQ[feature]['total_cases']) < 0.2:
        del train_IQ_copyauto[feature]
        del test_IQ_copyauto[feature]

columnsSJ=[]
for index in list(train_SJ):
    if train_SJ[index].var()>1000:
        columnsSJ.append(index)
columnsIQ=[]
for index in list(train_IQ):
    if train_IQ[index].var()>1000:
        columnsIQ.append(index)
for element in columnsSJ:
    del train_SJ[element]
    del test_SJ[element]
for element in columnsIQ:
    del train_IQ[element]
    del test_IQ[element]
bestScoreSJ = 45.
bestScoreIQ = 45.
bestsjTree = None
bestiqTree = None


rtreeForSJ = RandomForestRegressor(max_depth=6, criterion="mae",oob_score=True)
rtreeForIQ = RandomForestRegressor(max_depth=6, criterion="mae",oob_score=True)
rtreeForSJ.fit(train_SJ, total_CASESJ)
rtreeForIQ.fit(train_IQ, total_CASEIQ)
predictionsSJ = rtreeForSJ.predict(test_SJ)
predictionsIQ = rtreeForIQ.predict(test_IQ)

rtreeForSJ2 = RandomForestRegressor(max_depth=6, criterion="mae",oob_score=True)
rtreeForIQ2 = RandomForestRegressor(max_depth=6, criterion="mae",oob_score=True)
rtreeForSJ2.fit(train_SJ_copyauto, total_CASESJ)
rtreeForIQ2.fit(train_IQ_copyauto, total_CASEIQ)
predictionsSJ2 = rtreeForSJ2.predict(test_SJ_copyauto)
predictionsIQ2 = rtreeForIQ2.predict(test_IQ_copyauto)

print(rtreeForIQ.oob_score_, rtreeForSJ.oob_score_)
print()
print(rtreeForIQ2.oob_score_, rtreeForSJ2.oob_score_)

#if rtreeForIQ.oob_score_ >= rtreeForIQ2.oob_score_:
#    predictionsIQ = predictionsIQ
#
#else:
#    predictionsIQ = predictionsIQ2
#
#if rtreeForSJ.oob_score_ >= rtreeForSJ2.oob_score_:
#    predictionsSJ = predictionsSJ
#
#else:
#    predictionsSJ = predictionsSJ2

# Manuel feature selection was seen to be more consistent, but we obtained our best result with the fusion of manual
# selection for Iquitos and auto feature selection with San Juan.(To be fair, we know which features auto feature
#   selection selects and we could have selected them by ourselves afterwards manually as well.)

finalArr=[]
for k in predictionsSJ:
    finalArr.append(k)
for t in predictionsIQ:
    finalArr.append(t)
submission=pd.read_csv('submission_format.csv')
for i in range(len(submission['total_cases'])):
    submission['total_cases'][i]=finalArr[i]
for i in range(len(finalArr)):
    finalArr[i]=int(finalArr[i])
del submission['total_cases']
submission['total_cases'] = finalArr
submission.to_csv("outputBR.csv", index=False)

