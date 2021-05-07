import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sweetviz as sv
# Preprocessing
from sklearn.preprocessing import LabelEncoder
# Modeling
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Model selection
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score

from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Import dataset
dataset = pd.read_csv("dataset/bank-additional-full.csv", sep = ";")

# Recode and change target variable name
dataset['y'] = dataset['y'].apply(lambda x: 0 if x =='no' else 1)
dataset.rename(columns = {"y" : "deposit"}, inplace = True)

dataset.info()

# recorde pdays variable
dataset['pdays'] = dataset['pdays'].apply(lambda x: 0 if x ==999 else x)

# One hot encoding

# First: we create two data sets for numeric and non-numeric data
numerical = dataset.select_dtypes(exclude=['object'])
categorical = dataset.select_dtypes(include=['object'])

# Second: One-hot encode the non-numeric columns

onehot = pd.get_dummies(categorical)

# third: Third: Union the one-hot encoded columns to the numeric ones

df = pd.concat([numerical, onehot], axis=1)


X = df.loc[ : , df.columns != 'deposit']
y = df[['deposit']]

# Create training, evaluation and test sets
X_train, test_X, y_train, test_y = train_test_split(X, y, test_size=.3, random_state=123)
X_eval, X_test, y_eval, y_test = train_test_split(test_X, test_y, test_size=.5, random_state=123)

X_y_train = pd.concat([X_train.reset_index(drop = True), y_train.reset_index(drop = True)], axis = 1)


count_no_deposit, count_deposit = X_y_train['deposit'].value_counts()
no_deposit = X_y_train[X_y_train['deposit'] == 0]
deposit = X_y_train[X_y_train['deposit'] == 1]

no_deposit_under = no_deposit.sample(count_deposit)

train_balanced = pd.concat([no_deposit_under.reset_index(drop = True), deposit.reset_index(drop = True)], axis = 0)

round(train_balanced['deposit'].value_counts()*100/len(train_balanced['deposit']), 2)

X_train = train_balanced.loc[ : , train_balanced.columns != 'deposit']
y_train = train_balanced[['deposit']]



'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector = SelectKBest(f_classif, k=10)
selector.fit(X_train, y_train)

X_train.columns[selector.get_support(indices=True)]

# 1st way to get the list
vector_names = list(X_train.columns[selector.get_support(indices=True)])
vector_names

X_train = X_train[vector_names]
'''

# Logistic regression

###########################
#MODEL 1
############################
clf_logistic = LogisticRegression(max_iter = 100000).fit(X_train, np.ravel(y_train))

preds = clf_logistic.predict_proba(X_eval)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval
pred_comparison = pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1)
pred_comparison.head(10)
preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > 0.5 else 0)
# Print the confusion matrix
matrix = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix)

######## Optimal cut-off
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit']).reset_index(drop = True)

numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)

cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))


cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

########## SET CUT-OFF POINT
preds = clf_logistic.predict_proba(X_eval)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)
target_names = ['No-deposit', 'Deposit']
print(classification_report(y_eval, preds_df['prob_accept_deposit'], target_names=target_names))


matrix_2 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_2)

accuracy_log_reg_1 = round((matrix_2[0][0]+matrix_2[1][1])/sum(sum(matrix_2)), 3)
print(accuracy_log_reg_1)

recall_deposit_log_reg_1 = round(matrix_2[1][1]/(matrix_2[1][1]+matrix_2[1][0]), 2)
print(recall_deposit_log_reg_1)

#AUC
prob_deposit_log_reg_1 = preds[:, 1]
auc_log_reg_1 = round(roc_auc_score(y_eval, prob_deposit_log_reg_1), 3)
print(auc_log_reg_1)


############################
# MODEL 2
############################

clf_logistic2 = LogisticRegression(solver='sag', max_iter = 10000, penalty = 'l2').fit(X_train, np.ravel(y_train))
preds = clf_logistic2.predict_proba(X_eval)

preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval

# oPTIMAL CUT OFF SEARCH

numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)


cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))

cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

# set threshold

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)

matrix_3 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_3)


accuracy_log_reg_2 = round((matrix_3[0][0]+matrix_3[1][1])/sum(sum(matrix_3)), 3)
print(accuracy_log_reg_2)

recall_deposit_log_reg_2 = round(matrix_3[1][1]/(matrix_3[1][1]+matrix_3[1][0]), 3)
print(recall_deposit_log_reg_2)

prob_deposit_log_reg_2 = preds[:, 1]
auc_log_reg_2 = round(roc_auc_score(y_eval, prob_deposit_log_reg_2), 3)
print(auc_log_reg_2)

############################
# MODEL 3
############################

# For selecting the variables we will use de RFE method. The Recursive Feature Elimination (RFE) method is a feature selection approach. It works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.

from sklearn.feature_selection import RFE

logreg = LogisticRegression()
rfe = RFE(logreg, 12)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)

print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))

# The variables having showing True are the ones we are interested in

col = X_train.columns[rfe.support_]

sum(rfe.support_)

X_train_reduced = X_train[col]
X_eval_reduced = X_eval[col]
###################### Modeling

clf_logistic3 = LogisticRegression(max_iter = 100000).fit(X_train_reduced, np.ravel(y_train))

preds = clf_logistic3.predict_proba(X_eval_reduced)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval


######## Optimal cut-off

numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)

cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))


cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

########## SET CUT-OFF POINT
preds = clf_logistic3.predict_proba(X_eval_reduced)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)
target_names = ['No-deposit', 'Deposit']
print(classification_report(y_eval, preds_df['prob_accept_deposit'], target_names=target_names))


matrix_4 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_4)

accuracy_log_reg_3 = round((matrix_4[0][0]+matrix_4[1][1])/sum(sum(matrix_4)), 3)
print(accuracy_log_reg_3)

recall_deposit_log_reg_3 = round(matrix_4[1][1]/(matrix_4[1][1]+matrix_4[1][0]), 2)
print(recall_deposit_log_reg_3)

#AUC
prob_deposit_log_reg_3 = preds[:, 1]
auc_log_reg_3 = round(roc_auc_score(y_eval, prob_deposit_log_reg_3), 3)
print(auc_log_reg_3)


data = {'Model': ['Logistic Regression Model 1', 'Regularized Logistic Regression Model', 'Reduced Logistic Regression Model'], 
        'Accuracy': [accuracy_log_reg_1, accuracy_log_reg_2, accuracy_log_reg_3],
        'Recall': [recall_deposit_log_reg_1, recall_deposit_log_reg_2, recall_deposit_log_reg_3],
        'AUC': [auc_log_reg_1, auc_log_reg_2, auc_log_reg_3]
        } 
comparison = pd.DataFrame(data) 
print(comparison)

############################
# MODEL 4
############################

# Train a model
clf_gbt = xgb.XGBClassifier(use_label_encoder=False).fit(X_train, np.ravel(y_train))

# Predict with a model
preds = clf_gbt.predict_proba(X_eval)

# Create dataframes with predictions
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval
pred_comparison = pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1)
pred_comparison.head(10)

numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)


cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))


cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

# Predict with a model

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)

target_names = ['No-deposit', 'Deposit']
print(classification_report(y_eval, preds_df['prob_accept_deposit'], target_names=target_names))


matrix_5 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_5)

accuracy_XGB_1 = round((matrix_5[0][0]+matrix_5[1][1])/sum(sum(matrix_5)), 3)
print(accuracy_XGB_1)

recall_XGB_1 = round(matrix_5[1][1]/(matrix_5[1][1]+matrix_5[1][0]), 2)
print(recall_XGB_1)

prob_deposit_xgb_1 = preds[:, 1]
auc_XGB_1 = round(roc_auc_score(y_eval, prob_deposit_xgb_1), 3)
print(auc_XGB_1)


############################
# MODEL 5
############################

# Create and train the model on the training data
clf_gbt2 = xgb.XGBClassifier(use_label_encoder=False).fit(X_train,np.ravel(y_train))

# Print the column importances from the model

var_importance = clf_gbt2.get_booster().get_score(importance_type = 'weight')

# Visualisation of best variables
xgb.plot_importance(clf_gbt2, importance_type = 'weight')
plt.show()
plt.close()

# Filter the X_train dataset with best variables
col_names = pd.DataFrame(var_importance, index = [1]).columns
  
col_names

X_train_reduced = X_train[col_names]
X_eval_reduced = X_eval[col_names]
# Create and train the model on the training data
clf_gbt2 = xgb.XGBClassifier(use_label_encoder=False).fit(X_train_reduced,np.ravel(y_train))

## OIptimal CUT OFF SEARCH
preds = clf_gbt2.predict_proba(X_eval_reduced)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit']).reset_index(drop = True)
true_df = y_eval

numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)


cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))


cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)


# Predict with a model
preds = clf_gbt2.predict_proba(X_eval_reduced)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)
target_names = ['No-deposit', 'Deposit']
print(classification_report(y_eval, preds_df['prob_accept_deposit'], target_names=target_names))


matrix_6 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_6)

accuracy_XGB_2 = round((matrix_6[0][0]+matrix_6[1][1])/sum(sum(matrix_6)), 3)
print(accuracy_XGB_2)

recall_XGB_2 = round(matrix_6[1][1]/(matrix_6[1][1]+matrix_6[1][0]), 2)
print(recall_XGB_2)

prob_deposit_xgb_2 = preds[:, 1]
auc_XGB_2 = round(roc_auc_score(y_eval, prob_deposit_xgb_2), 3)
print(auc_XGB_2)



############################
# MODEL 6
############################

clf_gbt3 = xgb.XGBClassifier(learning_rate = 0.01, max_depth = 7, n_estimators = 300)

cv_scores = cross_val_score(clf_gbt3, X_train, np.ravel(y_train), cv = 10)
print(cv_scores)

print("Average accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

preds = cross_val_predict(clf_gbt3, X_eval, np.ravel(y_eval), cv=10, method = 'predict_proba')
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit']).reset_index(drop = True)
true_df = y_eval

numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)

cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))

cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

preds = cross_val_predict(clf_gbt3, X_eval, np.ravel(y_eval), cv=10, method = 'predict_proba')
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit']).reset_index(drop = True)
true_df = y_eval

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)
target_names = ['No-deposit', 'Deposit']
print(classification_report(true_df, preds_df['prob_accept_deposit'], target_names=target_names))

matrix_7 = confusion_matrix(true_df,preds_df['prob_accept_deposit'])
print(matrix_7)

accuracy_XGB_3 = round((matrix_7[0][0]+matrix_7[1][1])/sum(sum(matrix_7)), 3)
print(accuracy_XGB_3)

recall_XGB_3 = round(matrix_7[1][1]/(matrix_7[1][1]+matrix_7[1][0]), 2)
print(recall_XGB_3)

prob_deposit_xgb_3 = preds[:, 1]
auc_XGB_3 = round(roc_auc_score(true_df, prob_deposit_xgb_3), 3)
print(auc_XGB_3)



data = {'Model': ['Gradient Boosting Trees Model 1', 'Reduced Gradient Boosting Trees Model', 'Cross Validated Gradient Boosting Trees Model'], 
        'Accuracy': [accuracy_XGB_1, accuracy_XGB_2, accuracy_XGB_3],
        'Recall': [recall_XGB_1, recall_XGB_2, recall_XGB_3],
        'AUC': [auc_XGB_1, auc_XGB_2, auc_XGB_3]
        } 
comparison = pd.DataFrame(data) 
print(comparison)
                                      

############################
# MODEL 7
############################


random_forest = RandomForestClassifier().fit(X_train, np.ravel(y_train))

preds = random_forest.predict_proba(X_eval)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval
pred_comparison = pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1)
pred_comparison.head(10)



numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)


cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))


cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)

target_names = ['No-deposit', 'Deposit']
print(classification_report(y_eval, preds_df['prob_accept_deposit'], target_names=target_names))

matrix_8 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_8)

accuracy_random_forest = round((matrix_8[0][0]+matrix_8[1][1])/sum(sum(matrix_8)), 3)
print(accuracy_random_forest)

recall_random_forest = round(matrix_8[1][1]/(matrix_8[1][1]+matrix_8[1][0]), 2)
print(recall_random_forest)

prob_deposit_random_forest = preds[:, 1]
auc_random_forest = round(roc_auc_score(y_eval, prob_deposit_random_forest), 3)
print(auc_random_forest)

############################
# MODEL 8
############################

# Potential grid search (Computational power extensive)

#model = RandomForestClassifier()
#n_estimators = [100, 200, 300, 400, 500]
#min_samples_split = list(range(0,200,10))
#param_grid = dict(min_samples_split=min_samples_split, n_estimators=n_estimators)
#fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring="recall", cv=kfold)
#grid_result = grid_search.fit(X_train, y_train)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))
	
# best_min_samples_split = grid_result.best_params_['min_samples_split']
# best_n_estimators = grid_result.best_params_['n_estimators']
# print("Best recall: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

random_forest_2 = RandomForestClassifier(n_estimators=350,min_samples_split=70,min_samples_leaf=7).fit(X_train, np.ravel(y_train))
preds = random_forest_2.predict_proba(X_eval)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_eval
pred_comparison = pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1)
pred_comparison.head(10)



numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
preds_df.head(5)


cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))


cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)

target_names = ['No-deposit', 'Deposit']
print(classification_report(y_eval, preds_df['prob_accept_deposit'], target_names=target_names))

matrix_8 = confusion_matrix(y_eval,preds_df['prob_accept_deposit'])
print(matrix_8)

accuracy_random_forest = round((matrix_8[0][0]+matrix_8[1][1])/sum(sum(matrix_8)), 3)
print(accuracy_random_forest)

recall_random_forest = round(matrix_8[1][1]/(matrix_8[1][1]+matrix_8[1][0]), 2)
print(recall_random_forest)

prob_deposit_random_forest = preds[:, 1]
auc_random_forest = round(roc_auc_score(y_eval, prob_deposit_random_forest), 3)
print(auc_random_forest)



############################
# MODEL SELECTION
############################


data = {'Model': ['Logistic Regression Model 1', 
                  'Regularized Logistic Regression Model', 
                  'Reduced Logistic Regression Model', 
                  'Gradient Boosting Trees Model 1', 
                  'Reduced Gradient Boosting Trees Model', 
                  'Cross Validated Gradient Boosting Trees Model',
                  'Random Forest',
                  'Tuned Random Forest'], 
        'Accuracy': [accuracy_log_reg_1, 
                     accuracy_log_reg_2, 
                     accuracy_log_reg_3, 
                     accuracy_XGB_1, 
                     accuracy_XGB_2, 
                     accuracy_XGB_3,
                     accuracy_random_forest,
                     accuracy_random_forest_2],
        'Recall': [recall_deposit_log_reg_1, 
                   recall_deposit_log_reg_2, 
                   recall_deposit_log_reg_3, 
                   recall_XGB_1, 
                   recall_XGB_2, 
                   recall_XGB_3,
                   recall_random_forest,
                   recall_random_forest_2],
        'AUC': [auc_log_reg_1, 
                auc_log_reg_2, 
                auc_log_reg_3, 
                auc_XGB_1, 
                auc_XGB_2, 
                auc_XGB_3,
                auc_random_forest,
                auc_random_forest_2]
        } 


comparison = pd.DataFrame(data) 
print(comparison.sort_values(["Recall", "AUC"], ascending = False))

comparison = pd.DataFrame(data) 
print(comparison)


comparison = pd.DataFrame(data) 
print(comparison.sort_values(["AUC", "Recall"], ascending = False))



################################################
# ROC LOG REG
################################################

# ROC chart components
fallout_lr_1, sensitivity_lr_1, thresholds_lr_1 = roc_curve(y_eval, prob_deposit_log_reg_1)
fallout_lr_2, sensitivity_lr_2, thresholds_lr_2 = roc_curve(y_eval, prob_deposit_log_reg_2)
fallout_lr_3, sensitivity_lr_3, thresholds_lr_3 = roc_curve(y_eval, prob_deposit_log_reg_3)


# ROC Chart with both
plt.plot(fallout_lr_1, sensitivity_lr_1, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_lr_2, sensitivity_lr_2, color = 'red', label='%s' % 'Regularized Logistic Regression Model')
plt.plot(fallout_lr_3, sensitivity_lr_3, color = 'green', label='%s' % 'Reduced Logistic Regression Model')



plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for all LR models on the Probability of Deposit")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
plt.close()


################################################
# ROC XGBoost
################################################

# ROC chart components
fallout_xgb_1, sensitivity_xgb_1, thresholds_xgb_1 = roc_curve(y_eval, prob_deposit_xgb_1)
fallout_xgb_2, sensitivity_xgb_2, thresholds_xgb_2 = roc_curve(y_eval, prob_deposit_xgb_2)
fallout_xgb_3, sensitivity_xgb_3, thresholds_xgb_3 = roc_curve(y_eval, prob_deposit_xgb_3)


# ROC Chart with both
plt.plot(fallout_xgb_1, sensitivity_xgb_1, color = 'blue', label='%s' % 'XGBoost Model')
plt.plot(fallout_xgb_2, sensitivity_xgb_2, color = 'red', label='%s' % 'Reduced XGBoost Model')
plt.plot(fallout_xgb_3, sensitivity_xgb_3, color = 'green', label='%s' % 'Cross Validated XGBoost Model')



plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for all XGB models on the Probability of Deposit")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
plt.close()


################################################
# Random Forest
################################################

# ROC chart components
fallout_random_forest, sensitivity_random_forest, thresholds_random_forest = roc_curve(y_eval, prob_deposit_random_forest)
fallout_random_forest_2, sensitivity_random_forest_2, thresholds_random_forest_2 = roc_curve(y_eval, prob_deposit_random_forest_2)

# ROC Chart with both
plt.plot(fallout_random_forest, sensitivity_random_forest, color = 'blue', label='%s' % 'Random Forest Model')
plt.plot(fallout_random_forest_2, sensitivity_random_forest_2, color = 'red', label='%s' % 'Tuned Random Forest Model')


plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for Random Forest Model on the Probability of Deposit")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
plt.close()

################################################
# All Models
################################################


# ROC Chart with both
plt.plot(fallout_lr_1, sensitivity_lr_1, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_lr_2, sensitivity_lr_2, color = 'red', label='%s' % 'Regularized Logistic Regression Model')
plt.plot(fallout_lr_3, sensitivity_lr_3, color = 'green', label='%s' % 'Reduced Logistic Regression Model')
plt.plot(fallout_xgb_1, sensitivity_xgb_1, color = 'yellow', label='%s' % 'XGBoost Model')
plt.plot(fallout_xgb_2, sensitivity_xgb_2, color = 'blueviolet', label='%s' % 'Reduced XGBoost Model')
plt.plot(fallout_xgb_3, sensitivity_xgb_3, color = 'orange', label='%s' % 'Cross Validated XGBoost Model')
plt.plot(fallout_random_forest, sensitivity_random_forest, color = 'orchid', label='%s' % 'Random Forest Model')
plt.plot(fallout_random_forest_2, sensitivity_random_forest_2, color = 'black', label='%s' % 'Random Forest Model')

plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for Random Forest Model on the Probability of Deposit")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
plt.close()


################################################
# MODEL ASSESSMENT
################################################


# test

preds = clf_gbt.predict_proba(X_test)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_test
numbers  = [float(x)/1000 for x in range(1000)]
for i in numbers:
    preds_df[i]= preds_df.prob_accept_deposit.map(lambda x: 1 if x > i else 0)
    
cutoff_df = pd.DataFrame( columns = ['prob','accs','def_recalls','nondef_recalls'])
for i in numbers:
    cm1 = metrics.confusion_matrix(true_df, preds_df[i])
    total1=sum(sum(cm1))
    accs = (cm1[0][0]+cm1[1][1])/total1
    
    def_recalls = cm1[1][1]/(cm1[1][1]+cm1[1][0])
    nondef_recalls = cm1[0][0]/(cm1[0][0]+cm1[0][1])
    cutoff_df.loc[i] =[ i ,accs,def_recalls,nondef_recalls]
print(cutoff_df.head(5))
cutoff_df["diff"] = abs(cutoff_df.def_recalls - cutoff_df.nondef_recalls)
best_threshold = cutoff_df["prob"].loc[cutoff_df["diff"] == min(cutoff_df["diff"])]
best_threshold = best_threshold.iloc[0]
print(best_threshold)
preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)
matrix_10 = confusion_matrix(y_test,preds_df['prob_accept_deposit'])
print(matrix_10)
accuracy_XGB_1_test = round((matrix_10[0][0]+matrix_10[1][1])/sum(sum(matrix_10)), 3)
print(accuracy_XGB_1_test)
recall_XGB_1_test = round(matrix_10[1][1]/(matrix_10[1][1]+matrix_10[1][0]), 2)
print(recall_XGB_1_test)
prob_deposit_xgb_1_test = preds[:, 1]
auc_XGB_1_test = round(roc_auc_score(y_test, prob_deposit_xgb_1_test), 3)
print(auc_XGB_1_test)

# train

preds = clf_gbt.predict_proba(X_train)
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_accept_deposit'])
true_df = y_train

preds_df['prob_accept_deposit'] = preds_df['prob_accept_deposit'].apply(lambda x: 1 if x > best_threshold else 0)

matrix_11 = confusion_matrix(y_train,preds_df['prob_accept_deposit'])
print(matrix_11)
accuracy_XGB_1_train = round((matrix_11[0][0]+matrix_11[1][1])/sum(sum(matrix_11)), 3)
print(accuracy_XGB_1_train)

recall_XGB_1_train = round(matrix_11[1][1]/(matrix_11[1][1]+matrix_11[1][0]), 2)
print(recall_XGB_1_train)

prob_deposit_xgb_1_train = preds[:, 1]
auc_XGB_1_train = round(roc_auc_score(y_train, prob_deposit_xgb_1_train), 3)
print(auc_XGB_1_train)

fallout_xgb_1, sensitivity_xgb_1, thresholds_xgb_1 = roc_curve(y_eval, prob_deposit_xgb_1)
fallout_xgb_1_test, sensitivity_xgb_1_test, thresholds_xgb_1_test = roc_curve(y_test, prob_deposit_xgb_1_test)
fallout_xgb_1_train, sensitivity_xgb_1_train, thresholds_xgb_1_train = roc_curve(y_train, prob_deposit_xgb_1_train)


# ROC Chart with both
plt.plot(fallout_xgb_1, sensitivity_xgb_1, color = 'blue', label='%s' % 'ROC validation set')
plt.plot(fallout_xgb_1_test, sensitivity_xgb_1_test, color = 'red', label='%s' % 'ROC test set')
plt.plot(fallout_xgb_1_train, sensitivity_xgb_1_train, color = 'black', label='%s' % 'ROC train set')


plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for all XGB models on the Probability of Deposit")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
plt.close()

data = {'Dataset': ['Validation Set', 'Test Set', 'Train set'], 
        'Accuracy': [accuracy_XGB_1, accuracy_XGB_1_test, accuracy_XGB_1_train],
        'Recall': [recall_XGB_1, recall_XGB_1_test, recall_XGB_1_train],
        'AUC': [auc_XGB_1, auc_XGB_1_test, auc_XGB_1_train]
        } 
comparison = pd.DataFrame(data) 
print(comparison.sort_values(["AUC"], ascending = False))



# overfitting assesment

fallout_xgb_1, sensitivity_xgb_1, thresholds_xgb_1 = roc_curve(y_eval, prob_deposit_xgb_1)
fallout_xgb_1_test, sensitivity_xgb_1_test, thresholds_xgb_1_test = roc_curve(y_test, prob_deposit_xgb_1_test)
fallout_xgb_1_train, sensitivity_xgb_1_train, thresholds_xgb_1_train = roc_curve(y_train, prob_deposit_xgb_1_train)


# ROC Chart with both
plt.plot(fallout_xgb_1, sensitivity_xgb_1, color = 'blue', label='%s' % 'ROC validation set')
plt.plot(fallout_xgb_1_test, sensitivity_xgb_1_test, color = 'red', label='%s' % 'ROC test set')
plt.plot(fallout_xgb_1_train, sensitivity_xgb_1_train, color = 'black', label='%s' % 'ROC train set')


plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for all XGB models on the Probability of Deposit")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
plt.close()


data = {'Dataset': ['Validation Set', 'Test Set', 'Train set'], 
        'Accuracy': [accuracy_XGB_1, accuracy_XGB_1_test, accuracy_XGB_1_train],
        'Recall': [recall_XGB_1, recall_XGB_1_test, recall_XGB_1_train],
        'AUC': [auc_XGB_1, auc_XGB_1_test, auc_XGB_1_train]
        } 
comparison = pd.DataFrame(data) 
print(comparison.sort_values(["AUC"], ascending = False))



