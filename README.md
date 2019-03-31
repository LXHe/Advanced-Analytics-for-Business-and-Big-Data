# Advanced-Analytics-for-Business-and-Big-Data
Assignments of this master-level course<br />

Assignment1-To predict whether a customer will churn for a telco server<br />
  Details: http://seppe.net/aa/compete/<br />
  AUC of testing set:<br />
  
  1st model: Lasso_Logistic: 80.816667 (MaxMinScaler)<br />
  
  2nd model: PCA_Lasso_Logistic: 60.988571<br />
  
  3rd model: Decision_tree: 79<br />
             Random_Forest: 91.179524<br />
             
  4th model: Gradient_boosting: 93.10381<br />
             Stochastic_gradient_boosting:<br />
               01: 93.119524 (subsample=0.75, max_depth=6, n_est=150, eta=0.25)<br />
               02:93.978571 (subsample=0.85, max_depth=3, n_est=150, eta=0.25)<br />
               03:93.879048 (subsample=0.85, max_depth=3, n_est=300, eta=0.25)<br />
               04:93.786667 (subsample=0.75, max_depth=3, n_est=200, eta=0.2)<br />
               05:93.981429 (subsample=0.85, max_depth=3, n_est=500, eta=0.2)<br />
               Improving n_estimators from 150 to 500 results in a slight improvement in test AUC with a high computational cost.<br />
               
  5th model: Naive_bayesian: 71.034048<br />
  
  6th model: Voting_classification: 89.851905 (weighted)<br />
    Soft voting of Logistic_regression, Random_forest, Gradient_boosting and Naive_bayesian doesn't help to improve prediction.<br />

  7th model: Neural_network:<br />
    For activation function of hidden layer, 'sigmoid' beats 'relu'. <br />
    For optimizer, 'adam' beats 'sgd'. <br />
    Yet, AUCs on the test set are not good enough.
