# Predictive Modeling: Hong Kong Horse Racing
With supervised machine learning models, predict the winning horse with data of various features. 

## Data Source
Data from [Kaggle](https://www.kaggle.com/gdaley/hkracing) user Graham Daley, containing two sets of data about horse information and race information. 

## Features Explanation
> won - whether horse won (1) or otherwise (0)<br/>
horse_age - current age of this horse at the time of the race<br/>
horse_rating - rating number assigned by HKJC to this horse at the time of the race<br/>
horse_gear - string representing the gear carried by the horse in the race. An explanation of the codes used may be found on the HKJC website.<br/>
declared_weight - declared weight of the horse and jockey, in lbs<br/>
actual_weight - actual weight carried by the horse, in lbs<br/>
draw - post position number of the horse in this race<br/>
win_odds - win odds for this horse at start of race<br/>
place_odds - place (finishing in 1st, 2nd or 3rd position) odds for this horse at start of race<br/>
surface - a number representing the type of race track surface: 1 = dirt, 0 = turf<br/>
distance - distance of the race, in metres<br/>
race_class - a number representing the class of the race<br/>
horse_country - country of origin of this horse<br/>
horse_type - sex of the horse, e.g. 'Gelding', 'Mare', 'Horse', 'Rig', 'Colt', 'Filly'<br/>
venue - a 2-character string, representing which of the 2 race courses this race took place at: ST = Shatin, HV = Happy Valley<br/>
config - race track configuration, mostly related to the position of the inside rail. For more details, see the HKJC website.<br/>
going - track condition. For more details, see the HKJC website.

## Data Preprocessing
![Labels](/images/labels.png)<br/>
As the data is extremely skewed, resampling library [`imblearn`](https://imbalanced-learn.readthedocs.io/en/stable/index.html) is used. Under-sampling method RandomUnderSampler (RUS) and over-sampling method Synthetic Minority Over-sampling Technique (SMOTE) are used for different model experiments. 

## Modeling
<b>1. KNeighborsClassifier (kNN Classifier)</b>
   
   For this dataset, the target is to minimize False Positive, which means `prediction: win, actual: lose`. So the metric is set to be precision score of the positive class (1), which is the win label. Thus, find out the optimized k-value with for loops. 
   
   ![knn_ori](/images/knn_ori.png)![knn_rus](/images/knn_rus.png)![knn_sm](/images/knn_sm.png)

<b>2. LightGBM</b>
   
   Building a fast gradient boosting framework with adjusting the optimized threshold value to obtain the precision score of the positive class (1) as high as possible. 
   
   <img src="/images/lightgbm_ori.png" alt="top10_ori" width=400><img src="/images/lightgbm_rus.png" alt="top10_rus" width=400>
   <img src="/images/lightgbm_smote.png" alt="top10_smote" width=400><br/>
   Top 10 important features are shown, win odds and place odds are particularly ranked highly for all of the models. 

## Training Summary

|  |Size|	Time (sec) | Precision (0) | Precision (1) | F1-score (0) | F1-score (1) | True Positive | False Positive|
|---|---|-------------|---------------|---------------|--------------|--------------|---------------|---------------|
|**kNN_original_data**|49.1 MB	|3.23525|	0.92	|0.32	|0.96|	0.02|	13|28|
|**kNN_rus**|	8 MB|	1.06418|	0.95|	0.15	|0.81	|0.24	|752|	4222|
|**kNN_smote**|	90.6 MB|	6.65747|	0.93|	0.15|	0.90|	0.20	|328	|1808|
|**lgb_original_data**|	729 KB|	1.32079|	0.95|	0.28|	0.93|	0.33|	493	|1287|
|**lgb_rus**|	130 KB|	0.19044|	0.94|	0.30|	0.93	|0.32|	429	|1020|
|**lgb_smote**|823 KB|	1.98941	|0.93	|0.36|	0.95|	0.15|	113	|204|

* By processing a lot of data, kNN model trained with over-sampled data took the longest time, while LightGBM model trained with under-sampled data took the shortest time. 
* kNN models performed relatively worse with low precision score and f1-score of the positive class (1). 
* Training models aimed at minimize False Positive (predict: win, actual: lose), but it seems True Positive and False Positive are correlated. Same as gambling and investment, you have the chance to win and the risk to lose at the same time.
* File sizes of LightGBM models are incredibly small and the time spent on training models is really quick. 

## Predictions
With data for one of the races in the dataset (which is excluded in training the models), predict the winning horse. 

<b>1. KNeighborsClassifier (kNN Classifier)</b>

   For kNN models, only model trained with under-sampled data can predict the winning horse. However, there is one False Positive in the prediction. 
   
   ```
   "Classification Report of kNN Classifier model trained with under-sampled data"

                    precision  recall   f1-score   support

            0.0       1.00      0.92      0.96        13
            1.0       0.50      1.00      0.67         1

       accuracy                           0.93        14
      macro avg       0.75      0.96      0.81        14
   weighted avg       0.96      0.93      0.94        14
   ```
   ![knn_rus_cm](/images/knn_rus_cm.png)

<b>2. LightGBM</b>

   With setting the threshold value same as the models trained, all LightGBM models achieved 100% accuracy. 
   ```
   "Classification Report of all LightGBM models"

                    precision  recall   f1-score   support

            0.0       1.00      1.00      1.00        13
            1.0       1.00      1.00      1.00         1

       accuracy                           1.00        14
      macro avg       1.00      1.00      1.00        14
   weighted avg       1.00      1.00      1.00        14
   ```
   ![knn_lgb](/images/knn_lgb.png)

## Things to be Improved
* Correlation between features is ignored. 
* Feature scaling was not performed for different range of numeric values. 
* One-hot encoding was not performed and just keeping the numeric values for some categorical features such as `draw` and `race_class`. 
* Finetuning of hyperparameters for kNN classifier models is ignored. GridSearchCV can be performed to experiment better results. 
* The volume of test data is small, so the highly-accurate prediction result is more of luckiness. More data can be used to do the testing experiment. 

## Detailed Presentation
* Check out complete workflow with [Jupyter Notebook Slides](./code/slides).
* Check out complete code with [Jupyter Notebook](./code). 

## Skills Acquired
* Pandas: e.g. data cleaning and data imputation, one-hot encoding for categorical features
* Scikit-learn: e.g. fine-tuning hyperparameters of the training models, setting metrics in evaluating models
* Joblib: saving and loading models

## Acknowledgements
LightGBM code reference from Medium [article](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc) by Pushkar Mandot, and confusion matrix plot code reference from [Stack Overflow](https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib) user Joe Kington. Thank you coders for sharing your experience! =]
