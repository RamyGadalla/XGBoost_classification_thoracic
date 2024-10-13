# %% Load required modules
import arff
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring during cross validation
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import ConfusionMatrixDisplay # creates and draws a confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

# %% Load dataset
with open('ThoraricSurgery.arff', 'r') as file:
    raw_data = arff.load(file)

# %%
df = pd.DataFrame(raw_data['data'], columns=[attr[0] for attr in raw_data['attributes']])
df.head()

# %%
print(df.shape)

# %%
print(df.dtypes)

# %%
# Check missing data
df.isnull().sum()

# %%
# Checking outcome variable is binary as described in the metadata
df.Risk1Yr.unique()

# %%
# Convert 'object' variable to categorical data and T/F into boolean data. 

df[['PRE6', 'PRE14', 'DGN']] = df[['PRE6', 'PRE14', 'DGN']].astype('category')

df = df.replace({'T': True, 'F': False})

# %%
df.head()

# %%
df.dtypes

# %%
# check outcome catgory balance (if need to split with stratification)
sum(df['Risk1Yr'])/len(df['Risk1Yr'])

# %%
# Splilt dataframe training and test datasets
X = df.drop('Risk1Yr', axis=1).copy() 
y = df['Risk1Yr'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# %%
sum(y==False)/sum(y==True)

# %% build preliminary model
clf_xgb = xgb.XGBClassifier(enable_categorical=True,
                        tree_method = 'hist',
                    eval_metric= 'aucpr',
                    early_stopping_rounds=20,
                    scale_pos_weight= sum(y==False)/sum(y==True)
                    )

clf_xgb.fit(X_train, 
            y_train,
            verbose=True,
            eval_set=[(X_test, y_test)])

# %% model accuracy in predicting 1 year survival
clf_pred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, clf_pred)
print(cm)
accuracy = accuracy_score(y_test, clf_pred)
print(accuracy)

# %%
# plot confusion matrix
ConfusionMatrixDisplay.from_estimator(clf_xgb,
                                      X_test,
                                      y_test,
                                      values_format='d'
                                      )

# %%
clf_pred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, clf_pred)
print(cm)
accuracy = accuracy_score(y_test, clf_pred)
print(accuracy)

# %% optimize model hyperparameters
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0]
}


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(#objective='binary:logistic', 
                                eval_metric="logloss", ## this avoids a warning...
                                subsample=0.9,
                                enable_categorical=True,
                                tree_method = 'hist',
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc', #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0,
    n_jobs = 10,
    cv = 3
)

optimal_params.estimator.set_params(**{"eval_metric":"auc", "early_stopping_rounds":10})


optimal_params.fit(X_train, 
                   y_train, 
                   # early_stopping_rounds=10,                
                   # eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)


print(optimal_params.best_params_)


# %% Build model with optimal parameters
clf_xgb = xgb.XGBClassifier(
                            gamma=1.0,
                            learning_rate=0.01,
                            max_depth=3,
                            reg_lambda=10.0,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            early_stopping_rounds=20,
                            eval_metric='aucpr',
                            enable_categorical=True,
                            tree_method = 'hist',
                            scale_pos_weight= sum(y==False)/sum(y==True))
clf_xgb.fit(X_train, 
            y_train, 
            verbose=True, 
            eval_set=[(X_test, y_test)])

# %%
clf_pred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, clf_pred)
print(cm)
accuracy = accuracy_score(y_test, clf_pred)
print(accuracy)

# %%
# plot confusion matrix
ConfusionMatrixDisplay.from_estimator(clf_xgb,
                                      X_test,
                                      y_test,
                                      values_format='d',
                                      display_labels=["Did not leave", "Left"])

# %%
# Draw averga of total trees.
bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape': 'box',
               'style': 'filled, rounded',
               'fillcolor': '#78cbe'} 
leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': '#e48038'}

xgb.to_graphviz(clf_xgb, 
                condition_node_params=node_params,
                leaf_node_params=leaf_params) 

# %%
# Features importance to the model
explainer = shap.Explainer(clf_xgb)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)


