# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 08:55:35 2024

@author: jules
"""

# Standard Libary imports
"""Standard library imports"""
import sys, os, math, time
import pandas as pd
import numpy as np
import statsmodels.api as sm
pd.set_option('display.max_colwidth',None)

# Classes for Data Preprocessing and Validation
"""Classes for Data Preprocessing and Validation"""
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from copy import deepcopy

# Classes for Fitting Logistic Regression
"""Classes for Fitting Logistic Regression"""
from AdvancedAnalytics.Regression import logreg, stepwise
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Classes for Fitting and Describing a Decision Tree
"""Classes for Fitting and Describing a Decision Tree"""
from AdvancedAnalytics.Tree import tree_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Classes for Fitting a Random Forest
"""Classes for Fitting a Random Forest"""
from AdvancedAnalytics.Forest import forest_classifier
from sklearn.ensemble import RandomForestClassifier

# Classes for Fitting and Describing a FNN Neural Netowrk
"""Classes for Fitting and Describing a FNN Neural Netowrk"""
from AdvancedAnalytics.NeuralNetwork import nn_classifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore",category=ConvergenceWarning)


print("REPLACE-IMPUTE-ENCODE COMBINED DATAFRAME") 
print(time.ctime())

attribute_map = {
    'Teams': [DT.String, ('Archers')],  # Team names
    'Game': [DT.Ordinal, (1, np.inf)],  # Game identifier as ordinal
    'Season': [DT.Ordinal, (2000, 2100)], # Season identifier as ordinal (year)
    'Win': [DT.Binary, (0, 1)], # Win outcome (0 for loss, 1 for win)
    'S': [DT.Interval, (0, 30)],   # Score
    'A': [DT.Interval, (-np.inf, np.inf)], # Assists
    'TO': [DT.Interval, (-100, 100)],   # Turnovers
    '2Pt_Sh': [DT.Interval, (-np.inf, np.inf)],  # 2pt shots
    'SOG': [DT.Interval, (0, 100)],       # Shots on goal
    'POSS': [DT.Interval, (0, np.inf)], # Possession
    'TCH': [DT.Interval, (0, 400)],       # Touches
    'PAS': [DT.Interval, (0, np.inf)],  # Passes
    'CT': [DT.Interval, (0, np.inf)],# Caused Turnovers
    'GB': [DT.Interval, (0, 300)],       # Groundballs
    'FO_%': [DT.Interval, (0, np.inf)], # Faceoff Win percentage
    'Sv': [DT.Interval, (0, np.inf)], # Saves
}

attribute_map

#Read and display shape of dataset
target = "Win"
df = pd.read_csv("All_Teams_Stats.csv")
print("Read", df.shape[0], "observations with", df.shape[1], "attributes\n")
print("\nTop 5:\n", df.head(5))
print("\nBottom 5:\n", df.tail(5))

# Checking Win class balance before splitting
print("Target distribution:\n", df['Win'].value_counts(normalize=True))


# Preprocessing the data using one hot encoding
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',\
no_impute=[target], interval_scale = None, drop=True, display=True)

#drop rows with any missing values    
encoded_df = rie.fit_transform(df).dropna()

print(encoded_df)

X = encoded_df.drop(target, axis=1)
y = encoded_df[target]

# Reviewing encoded DF
print("Encoded DataFrame Columns:", encoded_df.columns)
print("Encoded DataFrame Shape:", encoded_df.shape)

# Checking VIF for Multicollinearity
predictors = encoded_df.drop(columns=[target]).reset_index(drop=True)
vif_data=pd.DataFrame()
vif_data['Feature']=predictors.columns
vif_data['VIF']=[variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]
print(vif_data)

# Printing Correlation Matrix
corr_matrix=predictors.corr()
print(corr_matrix)

# Split/Train Size
n_folds        = 15
Xt, Xv, yt, yv = train_test_split(X, y, test_size = 1.0/n_folds, 
                                  random_state=12345)

#Requires running Forest or Tree before running FNN
FNN        = True 
Forest     = True
Tree       = True
Logistic   = True
score_list = ['accuracy', 'f1']
opt_score  = 'accuracy' 

if Logistic:
    print("\n*** LOGISTIC REGRESION OPTIMIZATION USING STEPWISE SELECTION ***")
    print(time.ctime())
    print("Hyperparameter optimization involves selecting features.")
    print("By default, selection done using p<=0.05 to select features.")
    print("Winning Attributes or probabilities can be selected, but", 
          "not both.")

    sw       = stepwise(encoded_df, target, reg="logistic", method="stepwise", 
                        crit_in=0.05, crit_out=0.05, verbose=True)
    selected = sw.fit_transform()
    
    Xts      = Xt[selected]
    Xvs      = Xv[selected] 
    
    Xc = sm.add_constant(Xts)
    lm = sm.Logit(yt, Xc)
    results = lm.fit()
    print(results.summary())
    
    print("\nSingle Split Analysis:")
    lgr      = LogisticRegression(penalty='l2', C=.01,solver='liblinear', 
                                  max_iter=1000, tol=1e-8)
    lgr.fit(Xts, yt)
    logreg.display_split_metrics(lgr, Xts, yt, Xvs, yv)

if Tree:
        print("\n*** DECISION TREE OPTIMIZATION USING N-FOLD CROSS-VALIDATION ***")
        print(time.ctime())
        print("This is a",str(n_folds)+"-fold cross-validation for a Decision Tree")
        print("Hyperparameter optimization is conducted using",str.upper(opt_score))
        print("Optimization search grid contains four Decision Tree parameters:")
        # print("   1) Use topic groups or probabilities Drop_Topic=",Drop_Topic,",")
        print("   2) The maximum depth of the decision tree,")
        print("   3) The minimum leaf size allowed in the tree, and")
        print("   4) The minimum split size which is twice the minimum leaf size.")
        
        best_val_score = -np.inf
        depth_list     = [3, 4, 5, 6, 7, 8, 9, 10, None] #best depth=6 
        leaf_size_list = [2, 3, 4, 5, 6, 7, 8, 9, #V. Acc = 0.869792 Overfit=0.328
                         10, 11, 12, 13, 14]      #13 is best with depth=6
        for depth in depth_list:
            for leaf_size in leaf_size_list:
                split_size = 2*leaf_size
                dtc = DecisionTreeClassifier(max_depth=depth, 
                                             min_samples_leaf  = leaf_size, 
                                             min_samples_split = split_size,
                                             random_state=12345)
                scores = cross_validate(dtc, X, y, scoring=score_list, cv=n_folds,
                                        return_train_score=True, n_jobs=-1)
                score  = np.mean(scores['test_'+opt_score])
                std    = np.std(scores['test_'+opt_score])
                if score>best_val_score:
                    best_val_score   = score
                    best_train_score = np.mean(scores['train_'+opt_score])
                    overfit          = best_train_score - best_val_score
                    best_tree        = deepcopy(dtc)
                    print("\nDecision Tree for Depth=", depth, "Min Split=", 
                          split_size, "Min Leaf Size=", leaf_size)
                    print("{:.<10s}{:>12s}{:>12s}{:>12s}{:>12s}"\
                          .format("Metric", "Train Mean", "Val Mean", "Overfit", 
                                  "Std. Dev."))
                    print("{:.<10s}{:>11.5f}{:>13.5f}{:>13.5f}{:>10.5f}"\
                          .format(str.upper(opt_score), best_train_score, 
                                  best_val_score, overfit, std))

        tree_parms = best_tree.get_params()
        print("\nBEST TREE SELECTED USING", n_folds, "FOLD CV TO MAXIMIZE ", 
              opt_score)
        print("\nDECISION TREE OPTIMIZED", str.upper(opt_score), "USING", 
              str(n_folds)+"-FOLD CV.")
        print("    {:.<35s} {:>4.0f}".format("Tree Depth", 
                                         tree_parms['max_depth']))
        print("    {:.<35s} {:>4.0f}".format("Min. Leaf Size", 
                                         tree_parms['min_samples_leaf'])) 
        print("    {:.<35s} {:>4.0f}".format("Min. Split Size", 
                                         tree_parms['min_samples_split'])) 
        print("    {:.<31s}{:>9.6f}"\
              .format("Train "+str.upper(opt_score)+" (CV avg)", best_train_score))
        print("    {:.<31s}{:>9.6f}"\
              .format("Validation "+str.upper(opt_score)+" (CV avg)",  
                     best_val_score))
        print("    {:.<31s}{:>9.6f}".format("Overfitting (CV avg)", overfit))
        print("")
        
        best_tree.fit(Xt, yt) #CV only returns model, not fit
        tree_classifier.display_importance(best_tree, Xt.columns, top=5, plot=True)
        tree_classifier.display_split_metrics(best_tree, Xt, yt, Xv, yv)
        
        """ *** SELECT TOP FEATURES USING BEST TREE FEATURE IMPORTANCE *** """
        boundary   = 0.005
        selector   = SelectFromModel(best_tree, threshold=boundary) 
        selector.set_output(transform="pandas")
        X_selected = selector.transform(X)
        n_selected = X_selected.shape[1]
        selected   = X_selected.columns
        Xts        = Xt[selected]
        Xvs        = Xv[selected]
        
        #DISPLAY TOP FEATURES 
        print("\nDECISION TREE FEATURES")
        print("(importance >= ", boundary,")")
        n_selected = X_selected.shape[1]
        selected   = X_selected.columns
        print("\n****************")
        print("* TOP FEATURES *")
        print("****************")
        for i in range(n_selected):
            print("  {:.<4.0f} {:<s}".format(i+1, selected[n_selected-i-1]))
        print("***************\n")

if Forest:
    print("*** RANDOM FOREST OPTIMIZATION USING N-FOLD CROSS-VALIDATION ***")
    print(time.ctime())
    print("This is a",str(n_folds)+"-fold cross-validation for a Random Forest")
    print("Hyperparameter optimization is done using", str.upper(opt_score))
    print("Optimization search grid contains these Random Forest parameters:")
    #print("   1) Use topic groups or probabilities Drop_Topic=", Drop_Topic,",")
    print("   2) The number of estimators in the Forest," )
    print("   3) The maximum number of features selected for each tree,")
    print("   4) The maximum depth of forest decision trees,")
    print("   5) The minimum leaf size allowed in each tree")
    print("   6) The minimum split size is set to twice the min leaf size")
    best_val_score   = -np.inf
    estimators_list  = [100,150,200]
    #leaf_size_list   = [2, 3, 4, 5, 6]
    #depth_list       = [2, 4, 5, 6, 7, 8, 9, 10, 11, 15, None]
    #features_list    = [5, 10, 12, 15, 21, 26, None]
    # Parameters that maximizes accuracy but overfit
    #leaf_size_list   = [2, 3] #Validation Accuracy 0.885896 Overfit 0.08155
    #depth_list       = [9, 10] #10
    #features_list    = [19, 20] #20
    # Parameters to reduce overfitting
    leaf_size_list   = [30]
    depth_list       = [10, 11, 12, 13] #11 Validation Accuracy 0.878200
    features_list    = [4, 5, 6, 10, 15, 20] #5  Overfit 0.018086
    
    for e in estimators_list:
        for depth in depth_list:
            for features in features_list:
                for leaf_size in leaf_size_list:
                    split_size = 2*leaf_size
                    rfc = RandomForestClassifier(n_estimators=e, 
                                criterion="gini", min_samples_split=split_size, 
                                max_depth=depth, min_samples_leaf=leaf_size, 
                                max_features=features, n_jobs=-1, 
                                bootstrap=True, random_state=12345)
                    scores = cross_validate(rfc, X, y, scoring=score_list,
                                            return_train_score=True, cv=n_folds)
                    val_score     = np.mean(scores['test_'+opt_score])
                    
                    if val_score > best_val_score:
                        best_val_score   = val_score
                        best_train_score = np.mean(scores['train_'+opt_score])
                        overfit          = best_train_score - best_val_score
                        std              = np.std(scores['test_'+opt_score])
                        best_forest      = deepcopy(rfc)
                        print("\nForest with Depth=", depth, "Min Split=", 
                              split_size, "Min Leaf Size=", leaf_size, 
                              "Max Features=", features)
                        print("{:.<10s}{:>12s}{:>12s}{:>12s}{:>12s}"\
                              .format("Metric", "Train Mean", "Val Mean", 
                                      "Overfit", "Std. Dev."))
                        print("{:.<10s}{:>11.5f}{:>13.5f}{:>13.5f}{:>10.5f}"\
                              .format(str.upper(opt_score), best_train_score, 
                                      best_val_score, overfit, std))

    forest_parms = best_forest.get_params()
    print("\nRANDOM FOREST OPTIMIZED", str.upper(opt_score), "USING", 
          str(n_folds)+"-FOLD CV.")
    print("    {:.<35s} {:>4.0f}".format("Number of Trees (estimators)",
          forest_parms['n_estimators']))
    print("    {:.<35s} {}".format("Tree Depth", 
                                     forest_parms['max_depth']))
    print("    {:.<35s} {:>4.0f}".format("Min. Leaf Size", 
                                     forest_parms['min_samples_leaf'])) 
    
    print("    {:.<35s} {:>4.0f}".format("Min. Split Size", 
                                     forest_parms['min_samples_split'])) 
    print("    {:.<35s} {}".format("Maximum Features", 
                                     forest_parms['max_features']))
    print("    {:.<31s}{:>9.6f}"\
          .format("Train "+str.upper(opt_score)+" (CV avg)", best_train_score))
    print("    {:.<31s}{:>9.6f}"\
          .format("Validation "+str.upper(opt_score)+" (CV avg)",  
                 best_val_score))
    print("    {:.<31s}{:>9.6f}".format("Overfitting (CV avg)", overfit))
    print("")
    
    # Evaluate the selected random forest using a single train/test split
    best_forest.fit(Xt, yt)
    forest_classifier.display_split_metrics(best_forest, Xt, yt, Xv, yv)
    forest_classifier.display_importance(best_forest, Xt.columns, top=10, 
                                         plot=True)
    
    """*** SELECT TOP FEATURES USING BEST FOREST FEATURE IMPORTANCE ***"""
    boundary   = 0.005  #0.006 for optimum, 0.0073 for leaf size 30
    selector   = SelectFromModel(best_forest, threshold=boundary) 
    selector.set_output(transform="pandas")
    X_selected = selector.transform(X)
    n_selected = X_selected.shape[1]
    selected   = X_selected.columns
    Xts        = Xt[selected]
    Xvs        = Xv[selected]

    #DISPLAY TOP FEATURES
    print("\nTOP RANDOM FOREST FEATURES")
    print("(importance > ", boundary,")")
    n_selected = X_selected.shape[1]
    selected   = X_selected.columns
    print("\n****************")
    print("* TOP FEATURES *")
    print("****************")
    for i in range(n_selected):
        print("  {:.<4.0f} {:<s}".format(i+1, selected[n_selected-i-1]))
    print("***************\n")

if FNN:
    print("\n* FNN NEURAL NETWORK OPTIMIZATION USING N-FOLD CROSS-VALIDATION *")
    print(time.ctime())
    print("This is a", str(n_folds)+"-fold cross-validation for an FNN")
    print("Hyperparameter optimization is done using", str.upper(opt_score))
    print("Optimization search grid contains these FNN parameters:")
    print("   1) The top 5-10 features identified by best Random Forest,")
    print("   2) Drop_Topic=True  uses topic probabilities as features;")
    print("      Drop_Topic=False uses topic groups as features. ")
    print("   3) The neural network configuration," )
    print("   4) The neuron activation function, and")
    print("   5) The L2 Regularization Parameter (alpha) from 1e-4 to 2.5")
    
    # Cross-Validation Hyperparameters: network, alpha, activation
    network_list    = [(4), (5), (6), (10), (11), (12), 
                       (4,4), (4,3), (4,2), (4,4,2),
                       (5,5), (5,4), (5,3), (5,2), (5,5,2),
                       (6,6), (6,5), (6,4), (6,3), (6,2), (6,6,3), (7,7),
                       (12, 12), (12,6), (12, 6, 3)]
    alpha_list      = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 1.0, 1.9, 2.0, 2.1, 2.2, 2.3]
    activation_list = ['tanh', 'logistic', 'relu', 'identity']
    
    network_list    = [(6,6)]  # Best FNN Configuration - Val. Accuracy 0.891365
    alpha_list      = [0.5]    # Best L2 Shrinkage (alpha) Overfit -0.000272
    activation_list = ['relu'] # Best Actimation
    best_val_score  = -np.inf
    maxiter         = 9000
    tol             = 1e-4
    #solver          = "lbfgs"
    solver          = "adam"
    for nn in network_list:
        for alpha in alpha_list:
            for activate in activation_list:
                fnn = MLPClassifier(hidden_layer_sizes=nn, activation=activate, 
                            solver=solver, max_iter=maxiter, tol=tol,
                            alpha=alpha, random_state=12345)
                scores = cross_validate(fnn, X_selected, y, scoring=score_list,
                                        n_jobs=-1, return_train_score=True)
                
                val_score  = np.mean(scores['test_'+opt_score])
                if val_score>best_val_score:
                    # NEW BEST SOLUTION - Save and Print
                    best_val_score   = val_score
                    best_train_score = np.mean(scores['train_'+opt_score])
                    overfit          = best_train_score - best_val_score
                    std              = np.std(scores['test_'+opt_score])
                    best_fnn         = deepcopy(fnn)

                    print("\nFNN Configuration:", nn, "Alpha:",alpha, 
                          "Activation:", activate)
                    print("{:.<10s}{:>12s}{:>12s}{:>12s}{:>12s}".format(
                        "Metric", "Train Mean", "Val Mean", "Overfit", 
                        "Std. Dev."))
                    print("{:.<10s}{:>11.5f}{:>13.5f}{:>13.5f}{:>10.5f}".\
                          format(str.upper(opt_score), best_train_score, 
                                  best_val_score, overfit, std))
           
    fnn_parms = best_fnn.get_params()
    print("\nNEURAL NETWORK OPTIMIZED", str.upper(opt_score), "USING", 
          str(n_folds)+"-FOLD CV.")
    print("    {:.<35s} {}".format("FNN Network Perceptrons", 
                                         fnn_parms['hidden_layer_sizes']))
    print("    {:.<35s} {:>4.2f}".format("L2 Shrinkage (alpha)", 
                                     fnn_parms['alpha']))
    print("    {:.<35s} {}".format("Activation", 
                                     fnn_parms['activation'])) 
    print("    {:.<31s}{:>9.6f}"\
          .format("Train "+str.upper(opt_score)+" (CV avg)", best_train_score))
    print("    {:.<31s}{:>9.6f}"\
          .format("Validation "+str.upper(opt_score)+" (CV avg)",  
                 best_val_score))
    print("    {:.<31s}{:>9.6f}".format("Overfitting (CV avg)", overfit))
    print("")
   
    print("***** SPLIT METRICS FROM BEST FNN FITTED TO  Xts/yt  *****")
    best_fnn.fit(Xts, yt)
    nn_classifier.display_split_metrics(best_fnn, Xts, yt, Xvs, yv)
    
    
def select_top_features(model, X, y):
    if isinstance(model, RandomForestClassifier) or isinstance(model, DecisionTreeClassifier):
        model.fit(X, y)
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        })
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        top_features = feature_importance_df[feature_importance_df['Importance'] > 0.05]['Feature'].tolist()
        print(f"Top Features for {model.__class__.__name__}: {top_features}")
    elif isinstance(model, LogisticRegression):
        model.fit(X, y)
        coefs = abs(model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': coefs
        })
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        top_features = feature_importance_df[feature_importance_df['Importance'] > 0.1]['Feature'].tolist()
        print(f"Top Features for {model.__class__.__name__}: {top_features}")
    return top_features

# 2. Train and Evaluate Each Model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Evaluation for {model.__class__.__name__}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return model

# 3. Predict 2025 Wins
def predict_2025_wins(model, X_2025, df, season="2024"):
    # Filter df for the 2025 season and "Archers" team
    df_archers = df[(df['Season'] == 2024) & (df['Teams'] == 'Archers')]
    
    # Ensure there are rows for "Archers"
    if df_archers.empty:
        print(f"No data available for team: Archers in {season}")
        return 0

    # Extract features from df_archers corresponding to X_2025's columns
    X_2025_archers = df_archers[X_2025.columns]
    
    # Make predictions for "Archers"
    y_pred_2025 = model.predict(X_2025_archers)
    predicted_wins = sum(y_pred_2025)

    print(f"Predicted Wins in 2025 Season ({model.__class__.__name__}) for team Archers: {predicted_wins}")
    return predicted_wins

# Data Preparation
X = encoded_df.drop(columns=['Win'])
y = encoded_df['Win']

# Simulate data for 2024 season
df_2025 = df[df['Season'] == 2024]
X_2025 = df_2025[X.columns]

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41, stratify=y)

# Initialize Models
models = [
    LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=41),
    DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=41),
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=41, class_weight='balanced')
]

# Process Each Model
for model in models:
    # Select Top Features
    top_features = select_top_features(model, X, y)
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    X_2025_top = X_2025[top_features]
    
    # Train and Evaluate
    trained_model = train_and_evaluate_model(model, X_train_top, X_test_top, y_train, y_test)
    
    # Predict 2024 Wins
    predict_2025_wins(trained_model, X_2025_top,df)
