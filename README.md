# Advanced-Analytics-for-Business-and-Big-Data
Assignments of this master-level course
Assignment1-To predict whether a customer will churn for a telco server
  Details: http://seppe.net/aa/compete/
  AUC of testing set:
  1st model: Lasso_Logistic: 80.816667 (MaxMinScaler)
  2nd model: PCA_Lasso_Logistic: 60.988571           
  3rd model: Decision_tree: 79; 
             Random_Forest: 91.179524
  4th model: Gradient_boosting: 93.10381
             Stochastic_gradient_boosting:
               01: 93.119524 (subsample=0.75, max_depth=6, n_est=150, eta=0.25)
               02:93.978571 (subsample=0.85, max_depth=3, n_est=150, eta=0.25)
               03:93.879048 (subsample=0.85, max_depth=3, n_est=300, eta=0.25)
               04:93.786667 (subsample=0.75, max_depth=3, n_est=200, eta=0.2)
               05:93.981429 (subsample=0.85, max_depth=3, n_est=500, eta=0.2)
               improving n_estimators from 150 to 500 results in a slight improvement in test AUC with a high computational cost.
  5th model: Naive_bayesian: 71.034048
  6th model: Voting_classification: 89.851905 (weighted)
    Soft voting of Logistic_regression, Random_forest, Gradient_boosting and Naive_bayesian doesn't help to improve prediction.
