# Credit Card Fraud Detection Project  
## Analysis Report

## 1. Executive Summary

The main objective of this project is to build a Machine Learning system that can identify fraudulent credit card transactions while reducing false alarms on genuine customer transactions.

The dataset contains credit card transactions where:

- Class 0 represents a legitimate transaction.
- Class 1 represents a fraudulent transaction.

One of the biggest challenges in this project is class imbalance. The number of legitimate transactions is much higher than the number of fraud transactions. Because of this, accuracy alone is not a good measure of model performance.

For example, a model may achieve more than 99% accuracy simply by predicting almost every transaction as legitimate. However, such a model may still fail to detect many fraud cases.

Therefore, this project mainly focuses on:

- Precision
- Recall
- F1-Score
- ROC-AUC
- False Positive Rate

Four Machine Learning models were tested:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine

The models were trained using an 80% training and 20% testing split. Stratified sampling was used so that the fraud percentage remained similar in both datasets.

Feature scaling was performed using 'StandardScaler', and 'class_weight='balanced'' was used to help the models pay more attention to the small fraud class.

After comparing the models, the model with the highest F1-Score was selected as the best-performing model.

Random Forest was also tuned using 'GridSearchCV' to find better hyperparameter values.

Finally, feature importance and LIME were used to understand which features influenced the model's fraud predictions.

Overall, the project shows that Machine Learning can be useful for detecting suspicious credit card transactions, but the model should be carefully monitored because missing fraud and wrongly blocking genuine transactions both have business costs.



# 2. Methodology

## 2.1 Data Loading

The credit card transaction dataset was loaded using the Pandas library.

Basic checks were performed to understand:

- Number of rows and columns
- Column names
- Data types
- Missing values
- Duplicate records
- Summary statistics

Duplicate records were also identified and reviewed during the data-cleaning stage.



## 2.2 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before training Machine Learning models.

The following areas were checked:

- Distribution of legitimate and fraudulent transactions
- Fraud percentage
- Class imbalance ratio
- Transaction amount distribution
- Correlation between features and fraud
- Important features related to the target variable

Bar charts and pie charts were used to clearly visualize the class imbalance.

The analysis showed that fraudulent transactions represent only a very small percentage of the complete dataset.

This confirmed that the project is an imbalanced classification problem.



## 2.3 Feature and Target Separation

The dataset was divided into:

- X → Input features used by the model
- y → Target column called 'Class'

The 'Class' column was removed from the input features to prevent the model from seeing the answer during training.



## 2.4 Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

The training dataset was used to teach the models.

The testing dataset was kept separately to evaluate how well the trained models perform on unseen data.

'stratify=y' was used so that both training and testing datasets maintained a similar ratio of legitimate and fraudulent transactions.

A fixed 'random_state=42' was used to make the results reproducible.



## 2.5 Feature Scaling

'StandardScaler' was used to scale the numerical features.

After scaling, the values are generally centered around:

- Mean ≈ 0
- Standard deviation ≈ 1

Scaling is especially useful for models such as:

- Logistic Regression
- Support Vector Machine

The scaler was fitted only on the training dataset.

The same scaler was then used to transform the testing dataset.

This prevents information from the test data from leaking into the training process.



## 2.6 Machine Learning Models

The following four models were trained and evaluated.

### Logistic Regression

Logistic Regression was used as a simple baseline classification model.

'class_weight='balanced'' was used to give additional importance to fraud transactions.

### Decision Tree

A Decision Tree model was trained with a limited tree depth.

The maximum depth was controlled to reduce the risk of overfitting.

### Random Forest

Random Forest combines multiple Decision Trees and uses their combined predictions.

It was selected because it can handle complex patterns and also provides feature importance values.

### Support Vector Machine

Support Vector Machine was trained using the RBF kernel.

The RBF kernel helps the model identify more complex and non-linear patterns.

Probability prediction was enabled so that ROC-AUC could also be calculated.



## 2.7 Model Evaluation

Each model was evaluated using the same evaluation function.

The following metrics were calculated.

### Precision

Precision answers:

Out of all transactions predicted as fraud, how many were actually fraud?

High precision means fewer genuine customers are incorrectly flagged.



### Recall

Recall answers:

Out of all actual fraud transactions, how many did the model successfully detect?

High recall is especially important in fraud detection because missing fraud can cause financial losses.



### F1-Score

F1-Score provides a balance between precision and recall.

It is particularly useful when working with highly imbalanced datasets.

For this project, the F1-Score was used as an important metric for comparing models.



### ROC-AUC

ROC-AUC measures how well the model separates legitimate and fraudulent transactions across different probability thresholds.

A higher ROC-AUC generally indicates better separation between the two classes.



### False Positive Rate

False Positive Rate shows how often legitimate transactions are incorrectly classified as fraud.

This is important because too many false alarms may:

- Block genuine customers
- Increase manual review work
- Reduce customer satisfaction



# 3. Results and Findings

The performance of all four models was stored in a comparison table containing:

| Metric | Meaning |
|||
| Precision | Correct fraud predictions |
| Recall | Actual frauds detected |
| F1-Score | Balance of Precision and Recall |
| ROC-AUC | Ability to separate fraud and legitimate transactions |
| False Positive Rate | Genuine transactions wrongly flagged |

The models were then sorted using their F1-Score.

The model with the highest F1-Score was selected as the best-performing model.

### Main Findings

The analysis showed that:

- Accuracy is not enough for fraud detection because the dataset is highly imbalanced.
- Recall is important because the bank should detect as many fraud transactions as possible.
- Precision is also important because too many false alarms can inconvenience genuine customers.
- F1-Score provides a useful balance between these two objectives.
- ROC-AUC helps compare the overall ability of models to separate fraud from legitimate transactions.
- Random Forest provided useful feature-importance information and was selected for further hyperparameter tuning.

Confusion matrices were also created for each model.

These helped clearly identify:

- True Negatives
- False Positives
- False Negatives
- True Positives

ROC curves were plotted to visually compare the classification ability of all models.



# 4. Hyperparameter Tuning

Random Forest was further improved using 'GridSearchCV'.

The following parameters were tested:

- 'n_estimators'
- 'max_depth'
- 'min_samples_split'
- 'min_samples_leaf'

Five-fold cross-validation was used.

The objective of Grid Search was to find the parameter combination that produced the best F1-Score.

The best model was then evaluated again using the untouched test dataset.

This provided a more reliable estimate of how the tuned model may perform on new transactions.



# 5. Feature Importance Insights

Feature importance was obtained from the best Random Forest model.

The top 15 features were identified and displayed using a horizontal bar chart.

Features with higher importance scores had a greater influence on the model's fraud predictions.

The feature importance analysis helps us understand:

- Which transaction characteristics are most useful for fraud detection
- Which variables the model depends on most
- Whether some features contribute very little
- Which areas may need further investigation

Because many variables in the credit card dataset are anonymized as features such as 'V1', 'V2', 'V3', etc., their direct business meaning cannot easily be interpreted.

However, their importance scores still tell us which transformed features are most useful to the Machine Learning model.



# 6. LIME Explanation

LIME was used to explain individual predictions.

Three fraudulent transactions and three legitimate transactions were selected.

For each transaction, LIME showed the features that contributed most strongly to the prediction.

LIME also helped us understand whether a particular feature:

- Increased the chance of predicting fraud
- Reduced the chance of predicting fraud

This is useful because Machine Learning models should not only make predictions but should also provide understandable reasons, especially in financial applications.

For example, if a transaction is flagged for investigation, LIME can help an analyst understand which features influenced the decision.



# 7. Recommendations

Based on the project analysis, the following recommendations can be considered.

### 1. Focus on Recall and Precision Instead of Accuracy

Accuracy should not be the main metric because fraud transactions are very rare.

The main focus should remain on:

- Recall
- Precision
- F1-Score
- False Positive Rate



### 2. Use Probability Thresholds Carefully

The default classification threshold is usually 0.50.

However, the bank can adjust this value depending on business requirements.

For example:

- Lower threshold → Detect more fraud but increase false alarms
- Higher threshold → Reduce false alarms but may miss more fraud

A practical system could use different risk levels.

For example:

- Low probability → Approve normally
- Medium probability → Additional verification
- High probability → Send for manual review



### 3. Monitor False Positives

Too many false positives can negatively affect genuine customers.

The False Positive Rate should therefore be continuously monitored.

The goal should be to detect maximum fraud while keeping legitimate customer inconvenience as low as possible.



### 4. Monitor Model Performance Regularly

Fraud patterns can change over time.

Model performance should be reviewed regularly using recent transaction data.

A monthly monitoring process can check:

- Recall
- Precision
- F1-Score
- False Positive Rate
- Fraud percentage
- Changes in feature distributions



### 5. Retrain the Model

The fraud detection model should be retrained when enough new transaction data becomes available.

Quarterly retraining can be considered initially.

However, the exact frequency should depend on how quickly fraud patterns change.



### 6. Use Human Review for High-Risk Transactions

Instead of automatically blocking every suspicious transaction, very high-risk transactions can be sent for investigation.

This can reduce the negative impact on genuine customers.



### 7. Keep Model Explanations

Tools such as LIME can help fraud analysts understand why a transaction was marked suspicious.

This can improve trust in the model and help during manual investigations.



# 8. Limitations

Although the project demonstrates a complete fraud detection workflow, it has some limitations.

### Highly Imbalanced Dataset

Fraud cases are extremely rare compared with legitimate transactions.

This makes model training and evaluation more difficult.



### Limited Time Period

The dataset represents transactions collected during a relatively short period.

Therefore, it may not contain all possible fraud behaviours.



### Anonymized Features

Most features are named 'V1', 'V2', 'V3', etc.

Their real business meanings are not available.

Because of this, it is difficult to give detailed business explanations for individual features.



### Historical Dataset

Fraud methods continuously change.

A model trained on older transaction patterns may not always detect new types of fraud.



### Class Weighting Only

This project mainly uses 'class_weight='balanced'' to handle class imbalance.

Other techniques such as:

- SMOTE
- Random undersampling
- Random oversampling

were not deeply explored.



### Limited Hyperparameter Search

Grid Search was performed mainly for Random Forest.

More parameters and other models could also be tuned.



### Computational Cost

Support Vector Machine and GridSearchCV can take a long time on a dataset containing hundreds of thousands of transactions.

This may require more computing resources for faster experimentation.



# 9. Future Work

The project can be improved further in several ways.

### Try Additional Models

Future experiments can include:

- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting

These models often perform well on structured tabular datasets.



### Try Different Imbalance Handling Techniques

Different methods can be tested, such as:

- SMOTE
- Oversampling
- Undersampling
- Balanced Random Forest

Their effect on precision and recall can then be compared.



### Perform Threshold Optimization

Instead of using only the default 0.50 probability threshold, different thresholds can be tested.

The final threshold can be selected based on business cost.

For example:

- Cost of missing fraud
- Cost of investigating a false alarm
- Cost of blocking a genuine customer



### Use Precision-Recall Curves

Precision-Recall curves are especially useful for highly imbalanced datasets.

They can provide additional information beyond ROC curves.



### Add Cost-Based Evaluation

Future versions can calculate the financial impact of:

- False negatives
- False positives
- Correct fraud detection

This would help select the model based on business value instead of only statistical metrics.



### Real-Time Deployment

The model can later be converted into a real-time fraud detection API.

A possible process could be:

New Transaction → Preprocessing → Model Prediction → Fraud Probability → Decision

High-risk transactions can then be sent for manual review or additional verification.



# 10. Conclusion

This project successfully demonstrates the complete Machine Learning process for credit card fraud detection.

The project covered:

- Data understanding
- Data cleaning
- Class imbalance analysis
- Data visualization
- Train-test splitting
- Feature scaling
- Model training
- Model evaluation
- Model comparison
- Hyperparameter tuning
- Feature importance
- LIME explanation
- Deployment recommendations

The most important lesson from this project is that high accuracy does not always mean a good fraud detection model.

For an imbalanced fraud dataset, metrics such as Precision, Recall, F1-Score, ROC-AUC, and False Positive Rate provide a much better understanding of model performance.

The final goal should not simply be to detect fraud, but to achieve a practical balance between:

Detecting as much fraud as possible while avoiding unnecessary blocking of legitimate customers.



## Questions to Answer in the Report

### 1. What is the class imbalance ratio? How did you handle it?

The dataset is highly imbalanced. There are approximately 577 legitimate transactions for every 1 fraudulent transaction, so the class imbalance ratio is about 577:1.

This means fraud cases are very rare compared with normal transactions.

I handled this problem mainly by using:

- 'class_weight='balanced'' in the Machine Learning models. This gives more importance to the smaller fraud class.
- 'stratify=y' during train-test splitting. This keeps almost the same fraud percentage in both training and testing datasets.
- Metrics such as Precision, Recall, F1-Score and ROC-AUC instead of depending only on accuracy.

This is important because a model can show very high accuracy even if it fails to detect many fraud transactions.



### 2. Which model performed best? Why?

I compared four models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine

The best model was selected based mainly on the highest F1-Score.

F1-Score was important because it gives a balance between:

- Precision – how many transactions predicted as fraud were actually fraud.
- Recall – how many actual fraud transactions were successfully detected.

In the project code, the best model is selected automatically using:

'''python
comparison_df.loc[comparison_df['F1-Score'].idxmax()]
'''

So, in the final report, the exact model name should be taken from your notebook output.

You can write:

> [Best Model Name] performed best because it achieved the highest F1-Score while maintaining a good balance between fraud detection and false alarms.

If your Random Forest obtained the highest F1-Score, you can write:

> Random Forest performed best because it gave the best balance between Precision and Recall and achieved the highest F1-Score among the tested models.



### 3. What are the top 5 features driving fraud detection?

The top five features are obtained from the feature importance of the tuned Random Forest model.

Your code finds them using:

'''python
feature_importance.head(5)
'''

Therefore, the final five feature names should be copied from your notebook output.

You can present them like this:

| Rank | Feature | Importance |
|||:|
| 1 | [Feature 1] | [Value] |
| 2 | [Feature 2] | [Value] |
| 3 | [Feature 3] | [Value] |
| 4 | [Feature 4] | [Value] |
| 5 | [Feature 5] | [Value] |

These features had the strongest influence on the Random Forest model while deciding whether a transaction was fraudulent or legitimate.

Since most features are named 'V1', 'V2', 'V3', etc., their actual business meaning is hidden because the original dataset anonymized them.



### 4. What is the ROC-AUC score? What does it mean?

ROC-AUC measures how well the model can separate fraudulent transactions from legitimate transactions.

Its value normally ranges from 0 to 1.

- 0.50 → almost like random guessing
- 0.70–0.80 → reasonable performance
- 0.80–0.90 → good performance
- Above 0.90 → very good separation

For my best model, the ROC-AUC score was:

> ROC-AUC = [Enter your notebook result]

A high ROC-AUC means the model is generally good at giving fraud transactions higher risk scores than legitimate transactions.

However, ROC-AUC should not be used alone. Precision, Recall and F1-Score are also very important because the dataset is highly imbalanced.



### 5. Select one fraudulent and one legitimate transaction: Explain the predictions using LIME

#### Fraudulent Transaction

I selected one transaction where the actual class was Fraud (Class 1).

The model predicted a fraud probability of:

> [Enter Fraud Probability]%

LIME showed the main features that influenced this prediction.

Example format:

| Feature | LIME Weight | Effect |
||:||
| [Feature 1] | +[value] | Supported fraud prediction |
| [Feature 2] | +[value] | Supported fraud prediction |
| [Feature 3] | -[value] | Worked against fraud prediction |

Positive LIME weights generally push the prediction more towards Fraud, while negative weights push it more towards Legitimate.

Therefore, this transaction was classified as fraud because several important feature values strongly supported the fraud class.

#### Legitimate Transaction

I also selected one transaction where the actual class was Legitimate (Class 0).

The predicted fraud probability was:

> [Enter Fraud Probability]%

LIME showed that most important feature contributions pushed the prediction towards the legitimate class.

Example:

| Feature | LIME Weight | Effect |
||:||
| [Feature 1] | -[value] | Supported legitimate prediction |
| [Feature 2] | -[value] | Supported legitimate prediction |
| [Feature 3] | +[value] | Slightly supported fraud |

Since the overall fraud probability was low, the model correctly treated this transaction as legitimate.



### 6. What patterns indicate fraud? Provide 3–5 key insights

From the analysis, I observed the following important patterns:

1. Fraud transactions are extremely rare.  
   Fraud represents only a very small part of the complete dataset, making fraud detection a highly imbalanced classification problem.

2. Some features are much more useful than others.  
   The Random Forest feature importance shows that a small group of features contributes more strongly to fraud detection.

3. Fraud cannot be identified using transaction amount alone.  
   Both legitimate and fraudulent transactions can have different transaction amounts. Therefore, the model needs to consider several features together.

4. Certain combinations of feature values indicate higher fraud risk.  
   LIME explanations show that fraud prediction usually depends on multiple features working together rather than only one feature.

5. Missing fraud and false alarms require a balance.  
   Increasing fraud detection may also increase the number of legitimate transactions wrongly flagged. Therefore, Precision and Recall need to be balanced carefully.



### 7. What would you recommend for production deployment?

For production deployment, I would recommend the following approach:

- Use the best-performing trained model to generate a fraud probability for every new transaction.
- Do not rely only on the default '0.5' threshold. Test different thresholds based on business requirements.
- Transactions with a very high fraud probability, for example above '0.70', can be sent for immediate review or additional verification.
- Medium-risk transactions can require OTP, additional authentication or manual checking rather than being automatically blocked.
- Monitor Precision, Recall, F1-Score and False Positive Rate regularly.
- Retrain the model using newer transactions because fraud patterns can change over time.
- Keep human review for important or uncertain cases.
- Use explanation tools such as LIME so fraud analysts can understand why a transaction was flagged.

A simple production flow could be:

New Transaction → Data Preprocessing → Fraud Model → Fraud Probability → Risk Decision → Approve / Verify / Review

The final threshold should be decided together with the business team because missing fraud and blocking a genuine customer have different financial costs.



### 8. What are the limitations of your model?

The main limitations of this project are:

1. Highly imbalanced data  
   There are very few fraud transactions compared with legitimate transactions, which makes model training difficult.

2. Features are anonymized  
   Most columns are named 'V1', 'V2', 'V3', etc. Their real business meanings are not available, so interpretation is limited.

3. The dataset is historical  
   Fraud techniques change over time. A model trained on older fraud patterns may not detect completely new fraud methods.

4. Limited period of transaction data  
   The dataset covers only a relatively short period, so it may not represent every possible fraud situation.

5. Limited model tuning  
   Detailed hyperparameter tuning was mainly performed for Random Forest. Other models could also be tuned further.

6. Threshold needs business testing  
   The default probability threshold may not be the best choice for a real bank. It should be optimized according to the cost of missed fraud and false alarms.

7. More advanced models can be tested  
   Future work can compare models such as XGBoost, LightGBM and CatBoost and also explore methods such as SMOTE or other imbalance-handling techniques.