
# coding: utf-8

# # Programming Assignment 4
# ## Decision Tree Classifier using Spark Mllib 

# ### Data Used: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29/wdbc.data

# ##### Importing Required Libraries

# In[1]:


import findspark
findspark.init()

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#sc.stop()
conf = SparkConf().setMaster("local").setAppName("Dc")
sc = SparkContext(conf = conf)


# #### Reading Data as Text file

# In[2]:


raw_data = sc.textFile('Data_Input.csv')
print("Train data size is {}".format(raw_data.count()))
csv_data = raw_data.map(lambda x: x.split(","))


# In[31]:


print(csv_data.take(2))


# ## a) Program Code
# 
# ### Preparing the data RDD to pass into Decision tree classifier
# ### Splitting the cleaned data into 80:20 ratio of Training and Testing
# 
# ##### Please note that only the last 10 features are considered for model building : look into section(c) for further details.

# In[137]:


def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[21:31]
    class_ = 1.0
    if line_split[31] == '0':
        class_ = 0.0
    return LabeledPoint(class_, array([float(x) for x in clean_line_split]))
data = csv_data.map(create_labeled_point)

(trainingData, testData) = data.randomSplit([0.8, 0.2],seed=100)


# ### Training the Model

# In[138]:


tree_model = DecisionTree.trainClassifier(trainingData, numClasses=2, 
                                          categoricalFeaturesInfo={},
                                          impurity='gini', maxDepth=3,maxBins=7,minInstancesPerNode=3)


# ### Predicting the Class values of the Test data and Calculating the test error

# In[140]:


predictions = tree_model.predict(testData.map(lambda p: p.features))
labels_and_preds = testData.map(lambda p: p.label).zip(predictions)
testErr = labels_and_preds.filter(
    lambda p: p[0] != p[1]).count() / float(testData.count())

print('Test Error = ' + str(testErr))


# ##### Please note that additional performance metrics are presented in the section (f) , of the notebook

# ## b) The choice of parameters :
# * #### Impurity or Attribute Selection Method = "Gini"
# * #### MaxDepth = 3 (Given)
# * #### Max Bins = 7

# ## c) Notes and Any assumptions made :
# 
# ### * Understanding the columns :
# * The 10 features radius,texture,perimeter,area,smoothness,compactness,concavity,concave points,symmetry,fractal_dimension            are presented in 3 dimensions : Mean, Standard error and worst sequentially.
# * Mean is the mean of all the cells, Standard error is the SD of all the cells and worst is the mean of 3 largest mean values.
# ### * Assumptions :
# * The first feature ID is not contributing to the model hence ignored.
# * The features captured as Worst measure represent the data better than just measure and Standard error. Hence I have used only the columns from 22 till the end as my feature set.: radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst	compactness_worst,concavity_worst,concave points_worst,symmetry_worstfractal_dimension_worst.
# * I have validated the model on the other two sets (Mean and Standard Deviation) and found a better performance when I passed the worst_dimension features. You can look at the last section for the other runs.
# * The test errors are ::
# Mean : 0.097, SD : 0.141, Worst_ = 0.035( Smallest and hence the best gives the best features for model building)
# 

# ## d) Validation and Train/Test Startegy Used:
# 
#  ### Used the k-fold cross validation to evaluate the skill of the decision tree algorithm being learnt in general. I used the value k = 5 and the 4th fold seemed to be the best split giving out the accuracy upto 96.4%
#  ### Used the criteria and split of that model in my program as parameters.
#  ### max_depth is fixed, min_samples_leaf as minInstancesPerNode and criteria: gini
#  
#  ## Below is the code for Cross Validation employed. Ran on the feature_worst Measures only(Last 10 feature set)

# In[154]:


File = 'C:\\Users\\yandr\\OneDrive\\Desktop\\BigData\\spark\\Data_Input.csv'

df1 = pd.read_csv(File)

Train,Test = train_test_split(df1, test_size=0.2)
Data_X = Train.values[:,20:31]
Data_Y = Train.values[:,31]

X_Test=Test.values[:,20:31]
Y_Test=Test.values[:,31]


# In[149]:


kf = KFold(n_splits=5,random_state=None, shuffle=True)
tree_fold = []
acc_tree=[]
prec_tree=[]
rec_tree=[]

def train_tree(X_train,X_test,Y_train,Y_test):
   tree = DecisionTreeClassifier(criterion = "gini",max_depth=3,min_samples_leaf=3,random_state = 200)
   tree.fit(X_train, Y_train)
   pred=tree.predict(X_test)
   tree_fold.append(tree)
   acc_tree.append(accuracy_score(Y_test,pred))
   prec_tree.append(precision_score(Y_test,pred,average= 'macro'))
   rec_tree.append(recall_score(Y_test,pred,average= 'macro')) 
   return

for train_index, test_index in kf.split(Data_X):
  #print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = Data_X[train_index], Data_X[test_index]
  Y_train, Y_test = Data_Y[train_index], Data_Y[test_index]

  train_tree(X_train,X_test,Y_train,Y_test)


# In[150]:


acc_tree,prec_tree,rec_tree
#Calculating the average Performances:
Avg_acc_tree= sum(acc_tree)/len(acc_tree)
Avg_prec_tree= sum(prec_tree)/len(prec_tree)
Avg_rec_tree= sum(rec_tree)/len(rec_tree)

print("Averge of the metrics:")

Avg_acc_tree,Avg_prec_tree,Avg_rec_tree
acc_tree,prec_tree,rec_tree


# In[153]:


pred_tree=tree_fold[3].predict(X_Test)
c_tree=confusion_matrix(Y_Test, pred_tree, labels=None, sample_weight=None)
acc_tree=accuracy_score(Y_Test,pred_tree)
prec_tree=precision_score(Y_Test,pred_tree,average= None)
rec_tree=recall_score(Y_Test,pred_tree,average= None)
print("Performance Using the best Fold:")

print("Accuracy =",acc_tree,"Precision =",prec_tree,"Recall =" ,rec_tree)

print("Confusion Matrix =")
print(c_tree)


# ## e) Decision tree Obtained: model is built in the pyspark. Look above for section a) for code
# 
# ####  * Please note that the features used are the last 10 features (Worst measure) from the dataset.

# In[127]:


print(tree_model.toDebugString())


# ## f) Performance shown by the confusion matrix :

# In[117]:


from pyspark.mllib.evaluation import MulticlassMetrics
metrics=MulticlassMetrics(labels_and_preds)


# In[118]:


print("Recall = %s" % metrics.recall())
print("Precision = %s" % metrics.precision())
#print("F1 measure = %s" % metrics.f1Measure())
print("Accuracy = %s" % metrics.accuracy)
print(metrics.confusionMatrix().toArray())


# ## Additional Runs : Validating the assumptions:
# 
# ## Model for Standard deviation features: radius_mean	texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave points_mean	symmetry_mean	fractal_dimension_mean

# In[142]:


def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[1:11]
    class_ = 1.0
    if line_split[31] == '0':
        class_ = 0.0
    return LabeledPoint(class_, array([float(x) for x in clean_line_split]))
data = csv_data.map(create_labeled_point)

(trainingData, testData) = data.randomSplit([0.8, 0.2],seed=100)


# In[143]:


tree_model = DecisionTree.trainClassifier(trainingData, numClasses=2, 
                                          categoricalFeaturesInfo={},
                                          impurity='gini', maxDepth=3,maxBins=7,minInstancesPerNode=3)


# In[144]:


predictions = tree_model.predict(testData.map(lambda p: p.features))
labels_and_preds = testData.map(lambda p: p.label).zip(predictions)
testErr = labels_and_preds.filter(
    lambda p: p[0] != p[1]).count() / float(testData.count())

print('Test Error = ' + str(testErr))


# ## Model for Standard deviation features : radius_se	texture_se	perimeter_se	area_se	smoothness_se	compactness_se	concavity_se	concave points_se	symmetry_se	fractal_dimension_se

# In[145]:


def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[11:21]
    class_ = 1.0
    if line_split[31] == '0':
        class_ = 0.0
    return LabeledPoint(class_, array([float(x) for x in clean_line_split]))
data = csv_data.map(create_labeled_point)

(trainingData, testData) = data.randomSplit([0.8, 0.2],seed=100)


# In[146]:


tree_model = DecisionTree.trainClassifier(trainingData, numClasses=2, 
                                          categoricalFeaturesInfo={},
                                          impurity='gini', maxDepth=3,maxBins=7,minInstancesPerNode=3)


# In[147]:


predictions = tree_model.predict(testData.map(lambda p: p.features))
labels_and_preds = testData.map(lambda p: p.label).zip(predictions)
testErr = labels_and_preds.filter(
    lambda p: p[0] != p[1]).count() / float(testData.count())

print('Test Error = ' + str(testErr))

