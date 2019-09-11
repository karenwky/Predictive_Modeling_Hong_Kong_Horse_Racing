# Predictive Modeling: Hong Kong Horse Racing
With supervised machine learning models, predict the winning horse with data of various features. 

### Data Source
Data from [Kaggle](https://www.kaggle.com/gdaley/hkracing) user Graham Daley, containing two sets of data about horse information and race information. 

### Features Explanation
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
going - track condition. For more details, see the HKJC website.<br/>

### Data Pre-processing


### Modeling
1. KNeighborsClassifier (kNN Classifier)
2. LightGBM

### Training

### Predictions
With data for one of the races in the dataset (which is excluded in training the models), predict the winning horse. 

### Acknowledgement
LightGBM code reference from Medium [article](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc) by Pushkar Mandot. 
