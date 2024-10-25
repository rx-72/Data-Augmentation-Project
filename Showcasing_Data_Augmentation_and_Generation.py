#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn import model_selection
from sklearn import linear_model
import pandas as pd
import numpy as np
import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(10)


# In[2]:


import warnings
warnings.filterwarnings("ignore")


def main():
    # The goal here is to showcase the advantages of data augmentation and generation on
    # dataset models with issues that prevent them from being ideal for training prediction models
    # on. Consider the below set:
    # 
   
    # In[3]:
   
   
   
    polish_companies_bankruptcy = fetch_ucirepo(id=365)
    
    # note we drop the missing values for ease of predictions down the line towards proving improvements with
    # data augmentation. In a more thorough study, we would do something to keep these rows
    df = pd.DataFrame(polish_companies_bankruptcy.data.features).dropna()
   
    targets = polish_companies_bankruptcy.data.targets["class"]
    targets = targets[df.index]
    df.head()
   
   
    # Above is a dataset based off of Polish Banks. The features aren't labeled here (for example the first 2 are "net profit/total assests" and "total liabilities / total assets") but they're largely irrelevant to know here, just realize they have some weight to the prediction we're aiming for. In this case, we're trying to predict banks that are likely to go bankrupt where (1) is positive and (0) is negative. Let's get some statistics on the dataset we have here, along with includ
   
    # In[4]:
   
   
    len(df) #number of rows
   
   
    # In[5]:
   
   
    targets.value_counts() #number of bankruptcy(1) and non-bankruptcy(0) values
   
   
    # In[6]:
   
   
    #percentage of the above values. Notice how greatly imbalanced the two targets are.
    targets.value_counts()/len(targets) 
   
   
    # The imbalance between the labels will prove to be an issue in predictions. Now let's split the data set as follows:
    # 
    # - ~ 20% testing
    # - ~ 80% training
    #     - ~ 80% training
    #     - ~ 20% validation
   
    # In[7]:
   
   
    X = df
    y = targets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train, test_size=0.2)
   
   
    # In[8]:
   
   
    X_train.head()
   
   
    # We'll use the accuracy, precision, recall, and the f1-score to determine the performance of each model. Our reason will be listed below when determining the base model performance without any data augmentation
   
    # In[9]:
   
   
    #accuracy score checker
    def accuracy(predictions, true_y):
        y = true_y.tolist() #convert to list format
        correct = 0
        for i in range(0, len(y)):
           if predictions[i] == y[i]:
                correct += 1
        return correct/len(y)
   
    #precision, recall, and f1 score checker
    def precision_recall_F1_score(predictions, true_y):
        TP, FP, FN = 0, 0, 0
        y = true_y.tolist() #convert to list format
        for i in range(0, len(y)):
            true_val = y[i]
            predicted_val = predictions[i]
   
            if predicted_val == 1: #if positive predicted
                if true_val == predicted_val: #positive correctly predicted
                    TP += 1
                else: #positive incorrectly predicted
                    FP += 1
            else: #negative predicted
                if true_val != predicted_val: #negative incorrectly predicted
                    FN += 1 #negative incorrectly predicted
                   
        precision = TP/(TP + FP+1e-10)
        recall = TP/(TP + FN+1e-10)
        f1_score = (2 * precision * recall)/(precision + recall+1e-10)
        return precision, recall, f1_score
   
   
    # First let's try out a basic logistic regression model:
   
    # In[10]:
   
   
    logistic_mod = linear_model.LogisticRegression().fit(X_train,y_train) #max_iter=2000
    pred_valid = logistic_mod.predict(X_valid)
    pred_test = logistic_mod.predict(X_test)
   
   
    # In[11]:
   
   
    #Validation set result:
    acc_valid = accuracy(pred_valid, y_valid)
    precision_valid, recall_valid, f1_score_valid = precision_recall_F1_score(pred_valid, y_valid) 
    print("Accuracy: ", acc_valid)
    print("Precision: ", precision_valid)
    print("Recall: ", recall_valid)
    print("f1_score: ", f1_score_valid)
   
   
    # In[12]:
   
   
    #Test set result:
    acc_test = accuracy(pred_test, y_test)
    precision_test, recall_test, f1_score_test = precision_recall_F1_score(pred_test, y_test) 
    print("Accuracy: ", acc_test)
    print("Precision: ", precision_test)
    print("Recall: ", recall_test)
    print("f1_score: ", f1_score_test) 
   
   
    # Our accuracy seems really high but the rest of our performance measurers are at 0! What happened?
   
    # In[13]:
   
   
    #printing the number of positives that were predicted from both sets:
    sum(pred_valid), sum(pred_test)
   
   
    # It seems like our model here had a strong preference for predicting non-bankruptcy on almost all of the predictions. That explains the very low precision and recall values respectively: the model we trained here is not great (frankly terrible) at predicting the bankruptcy values since it thinks better to just assume 0. This almost means we have 0 TPs, we didn't get any predictions on bankruptcy correct!. As such the the f1_score is pretty 0 due to the poor performance of the precision and recall scores.
    # 
    # Overall accuracy is very high, but we're a lot more interested in predicting the bankruptcy cases then the non bankruptcy cases. In that sense, our base model is a failure. Now what if we applied some data augmentation methods?
   
    # # SMOTE (Synthetic Minority Over-sampling Technique)
   
    # SMOTE works to help increase the imbalance between the number of classes. It works as follows:
    #  1. Make note of the values that are part of the class with fewer values contained (in this case, the bankruptcy cases)
    #  
    #  2. Run k-nearest neighbors on a bankruptcy value (think vectors) and selecting a random batch of these neighbors to generate synthetic values whose features lie between the original value and the neighbor value, repeatedly for each bankruptcy point
    #  
    #  3. Add these new synthetic values to the dataset to bolster the bankruptcy cases
    #  
    # This seems problematic at a glance but let's see how it works:
    # (make note of the increased sample size below with SMOTE)
   
    # In[14]:
   
   
    from imblearn.over_sampling import SMOTE
   
    smote = SMOTE(sampling_strategy='auto')
    X_resampled_train, y_resampled_train = smote.fit_resample(X_train, y_train)
   
    print("Print X_train length", len(X_train))
    print("Print resampled X_train length", len(X_resampled_train))
   
    X_resampled_train.head()
   
   
    # Let's try running a model here now:
   
    # In[15]:
   
   
    logistic_mod_smote = linear_model.LogisticRegression().fit(X_resampled_train,y_resampled_train)
    pred_valid = logistic_mod_smote.predict(X_valid)
    pred_test = logistic_mod_smote.predict(X_test)
   
   
    # In[16]:
   
   
    #Validation set result:
    acc_valid = accuracy(pred_valid, y_valid)
    precision_valid, recall_valid, f1_score_valid = precision_recall_F1_score(pred_valid, y_valid) 
    print("Accuracy: ", acc_valid)
    print("Precision: ", precision_valid)
    print("Recall: ", recall_valid)
    print("f1_score: ", f1_score_valid)
   
   
    # In[17]:
   
   
    #Test set result:
    acc_test = accuracy(pred_test, y_test)
    precision_test, recall_test, f1_score_test = precision_recall_F1_score(pred_test, y_test) 
    print("Accuracy: ", acc_test)
    print("Precision: ", precision_test)
    print("Recall: ", recall_test)
    print("f1_score: ", f1_score_test)
   
   
    # Our accuracy dropped in the validation and test but we don't have 0 values for any of the other measurers anymore! Recall went up significantly, so we're clearly able to better indicate most of the bankruptcy cases in either set but our precision is still pretty small. We can figure out what happened here by checking the value counts of how many bankruptcy cases we really have in both sets:
   
    # In[18]:
   
   
    sum(pred_valid), sum(y_valid)
   
   
    # In[19]:
   
   
    sum(pred_test), sum(y_test)
   
   
    # Now we can see we did end up inflating the number of bankruptcy predictions we made but by too much! We have more than 10 times the number of actual bankruptcy cases here in the predictions. That explains why our precision is so low still, we're having issues truly predicting one that would exist in an unknown set in comparison to just finding the one in our given test/validation set here. This is to be expected since SMOTE also introduced a lot of noise here which is why we saw such an increase in bankruptcy predictions for our model.
   
    # # Noise Injection to features
   
    # The technique sounds exactly as it describes; we're introducing a certain amount of noise to our features that we train in. This sounds like a negative practice at first but in reality, the model we train learns the noise as a sort of invariability to the data which helps out the imbalance class issues we have here
   
    # In[20]:
   
   
    import numpy as np
   
    # Injecting Gaussian noise (mean=0, standard deviation=1)
    noise_factor = 1000 #0.01, 0.1, 1, 10, 100, 1000
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
   
    # Clip the values to ensure no negative values where it doesn't make sense
    X_train_noisy = np.clip(X_train_noisy, 0, None)
   
   
    # In[21]:
   
   
    logistic_mod_noisy = linear_model.LogisticRegression().fit(X_train_noisy,y_train)
    pred_valid = logistic_mod_noisy.predict(X_valid)
    pred_test = logistic_mod_noisy.predict(X_test)
   
   
    # In[22]:
   
   
    #Validation set result:
    acc_valid = accuracy(pred_valid, y_valid)
    precision_valid, recall_valid, f1_score_valid = precision_recall_F1_score(pred_valid, y_valid) 
    print("Accuracy: ", acc_valid)
    print("Precision: ", precision_valid)
    print("Recall: ", recall_valid)
    print("f1_score: ", f1_score_valid)
   
   
    # In[23]:
   
   
    #Test set result:
    acc_test = accuracy(pred_test, y_test)
    precision_test, recall_test, f1_score_test = precision_recall_F1_score(pred_test, y_test) 
    print("Accuracy: ", acc_test)
    print("Precision: ", precision_test)
    print("Recall: ", recall_test)
    print("f1_score: ", f1_score_test)
   
   
    # On the plus side, noise factor doesn't inflate the sample set as as hard SMOTE but the result improvements are much scale in comparison. We didn't see that much improvement in recall as we did previously, though we did improve precision about the same amount and accuracy wasn't reduced as severely. It's best to not rely solely on Noise Injection to fix this imbalance
   
    # # Feature Augmentation
   
    # Sometimes our data feature may have inherent relationships between one another that may not necessarily be clear and/or easy to fit to the data classes. For example there may not exist direct features that can 1 to 1 predict whether a bank data value will become bankrupt in the future or not but what if there was some complex mathematical combination of features that we could discover that could do so? This can prove useful to predicting data with little correlations to features but can be time intensive due to the number of potential combinations one must test along with the expansion of new feature columns to each data value.
   
    # In[24]:
   
   
   
   
   # Generate interaction terms and polynomial features (degree=2 for pairwise interactions)
   poly = PolynomialFeatures(degree=2, include_bias=False)
   X_train_poly = poly.fit_transform(X_train)
   X_valid_poly = poly.fit_transform(X_valid)
   X_test_poly = poly.fit_transform(X_test)
   
   
   # In[25]:
   
   
   logistic_mod_feature = linear_model.LogisticRegression().fit(X_train_poly,y_train)
   pred_valid = logistic_mod_feature.predict(X_valid_poly)
   pred_test = logistic_mod_feature.predict(X_test_poly)
   
   
   # In[26]:
   
   
   #Validation set result:
   acc_valid = accuracy(pred_valid, y_valid)
   precision_valid, recall_valid, f1_score_valid = precision_recall_F1_score(pred_valid, y_valid) 
   print("Accuracy: ", acc_valid)
   print("Precision: ", precision_valid)
   print("Recall: ", recall_valid)
   print("f1_score: ", f1_score_valid)
   
   
   # In[27]:
   
   
   #Test set result:
   acc_test = accuracy(pred_test, y_test)
   precision_test, recall_test, f1_score_test = precision_recall_F1_score(pred_test, y_test) 
   print("Accuracy: ", acc_test)
   print("Precision: ", precision_test)
   print("Recall: ", recall_test)
   print("f1_score: ", f1_score_test)
   
   
   # Based on the results here, it seems we could find any combination of polynomial functions to the 2nd degree that could help improve our measurers here in comparison to our other two methods. Due to the longer run time here, we likely shouldn't focus too much on this method if our improvements were barely made for longer time taken. 
   
   # # Targeted Undersampling (for Class Balance)
   
   # Similar to how SMOTE creates more of the minority classes values, we can also a data augmentation technique to reduce the amount of majority class values to help with imbalance. The concepts and ideas are generally the same with similar pros and cons. Here we use the RandomUnderSampler which chooses a random subset of points from the majority class to be removed. Note this is the main difference to something like SMOTE which increases the number of values wheareas here we decrease them.
   
   # In[28]:
   
   
   rus = RandomUnderSampler(random_state=42)
   X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
   logistic_mod_rus = linear_model.LogisticRegression().fit(X_train_resampled,y_train_resampled)
   pred_valid = logistic_mod_rus.predict(X_valid)
   pred_test = logistic_mod_rus.predict(X_test)
   
   
   # In[29]:
   
   
   print(len(X_train))
   print(len(X_train_resampled))
   
   
   # In[30]:
   
   
   #Validation set result:
   acc_valid = accuracy(pred_valid, y_valid)
   precision_valid, recall_valid, f1_score_valid = precision_recall_F1_score(pred_valid, y_valid) 
   print("Accuracy: ", acc_valid)
   print("Precision: ", precision_valid)
   print("Recall: ", recall_valid)
   print("f1_score: ", f1_score_valid)
   
   
   # In[31]:
   
   
   #Test set result:
   acc_test = accuracy(pred_test, y_test)
   precision_test, recall_test, f1_score_test = precision_recall_F1_score(pred_test, y_test) 
   print("Accuracy: ", acc_test)
   print("Precision: ", precision_test)
   print("Recall: ", recall_test)
   print("f1_score: ", f1_score_test)
   
   
   # Interestingly, it seems the results here are quite similar to what was achieved with SMOTE. We have a very much improved recall but we're making a lot more incorrect predictions for bankruptcy still, and we have a reduction in accuracy as well.
   
   # # Combined Techniques
   # 
   # Now let's apply all of these methods all on one singular model:
   
   # In[32]:
   
   
   def add_noise(X, noise_level=0.1):
       noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
       return X + noise
   
   
   # In[33]:
   
   
   #SMOTe
   smote = SMOTE(sampling_strategy='auto')
   X_smote_train, y_smote_train = smote.fit_resample(X_train, y_train)
   
   #RUS
   rus = RandomUnderSampler(random_state=42)
   X_train_resampled, y_train_resampled = rus.fit_resample(X_smote_train, y_smote_train)
   
   #noise injection
   X_resampled_noisy = add_noise(X_train_resampled, noise_level=0.05)
   
   #polynomial features
   poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
   X_train_poly = poly.fit_transform(X_resampled_noisy)
   X_valid_poly = poly.fit_transform(X_valid)
   X_test_poly = poly.fit_transform(X_test)
   
   logistic_mod_feature = linear_model.LogisticRegression().fit(X_train_poly,y_train_resampled)
   pred_valid = logistic_mod_feature.predict(X_valid_poly)
   pred_test = logistic_mod_feature.predict(X_test_poly)
   
   
   # In[34]:
   
   
   #Validation set result:
   acc_valid = accuracy(pred_valid, y_valid)
   precision_valid, recall_valid, f1_score_valid = precision_recall_F1_score(pred_valid, y_valid) 
   print("Accuracy: ", acc_valid)
   print("Precision: ", precision_valid)
   print("Recall: ", recall_valid)
   print("f1_score: ", f1_score_valid)
   
   
   # In[35]:
   
   
   #Test set result:
   acc_test = accuracy(pred_test, y_test)
   precision_test, recall_test, f1_score_test = precision_recall_F1_score(pred_test, y_test) 
   print("Accuracy: ", acc_test)
   print("Precision: ", precision_test)
   print("Recall: ", recall_test)
   print("f1_score: ", f1_score_test)
   
   
   # These results ended up being the most balanced out of what we got, seeing as we have the highest combination of accuracy and recall, along with moderately high precision and f1_score in comparison to our other models. Of course, we still hit quite low values for precision and f1_score so we can always improve to something more ideal by changing methods or tuning the hyperparameters. Nevertheless, data augmentation in this case has proven to help remedy and train models such that they can understand the data they're working with better by factoring noise and classifications with weight. Now that we highlighted these benefits, let's try something more complex. 

if __name__ == "__main__":
    main()
