
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project 2: Building a Student Intervention System

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ### Question 1 - Classification vs. Regression
# *Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

# **Answer: we should use classfication because we are trying to label each student with a binary label (i.e. yes, needs intervention because we predict failure or no, does not require intervention because we predict passing). Regression is more sutibale for problems in which we want to assign a number/value or continuous label to. Classification is more suited to no continuous labels, which is the case in this problem.**

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.

# In[6]:

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"


# ### Implementation: Data Exploration
# Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
# - The total number of students, `n_students`.
# - The total number of features for each student, `n_features`.
# - The number of those students who passed, `n_passed`.
# - The number of those students who failed, `n_failed`.
# - The graduation rate of the class, `grad_rate`, in percent (%).
# 

# In[7]:

# TODO: Calculate number of students
n_students = len(student_data)

# TODO: Calculate number of features
n_features = sum(1 for row in student_data)

# TODO: Calculate passing students
n_passed = student_data.passed.value_counts()['yes']

# TODO: Calculate failing students
n_failed = student_data.passed.value_counts()['no']

# TODO: Calculate graduation rate
grad_rate = float(student_data.passed.value_counts()['yes'])/len(student_data)

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.

# In[8]:

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


# ### Preprocess Feature Columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.

# In[9]:

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Implementation: Training and Testing Data Split
# So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
# - Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
#   - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
#   - Set a `random_state` for the function(s) you use, if provided.
#   - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

# In[10]:

# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split
# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=num_train, test_size=num_test, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# ## Training and Evaluating Models
# In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.
# 
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# *List three supervised learning models that are appropriate for this problem. For each model chosen*
# - Describe one real-world application in industry where the model can be applied. *(You may need to do a small bit of research for this — give references!)* 
# - What are the strengths of the model; when does it perform well? 
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?

# Decision Trees:
# One example of the use of decision tree learning is in machine learning assisted investing. A decision tree can be utilized to select companies that may be worth investing in. An example of such an application can be found in the following article: http://www.academia.edu/2994194/Application_of_Decision_Trees_for_Portfolio_Diversification_in_Indian_Share_Market.
# Decision tree learning algorithms are suitable if instances can be described by attribute, value pairs and if the target variable has discrete output values. This method also handles errors in the data well and can handle missing values as well. They are well suited for classification problems and are relatively easy to understand and code.
# The problem with decision tree methods is that they can easily over fit the data and are sensitive to small perturbations in the data which can lead to drastically different trees. They arent well suited to continious data and do not model the relationships between variables very well.
# This is a good model for the data because each instance is described by attribute, value pairs and the target variable is a discrete value.
# 
# SVM:
# SVM may also be a suitable algorithm since it too is suitable for classification problems like this. Somewhat similarly, SVMs can be used to check for mechanical fault potential of mechanical components based on the vibration patterns of the components (http://www.sciencedirect.com/science/article/pii/S0957417410013801), This is somewhat similir to our problem in that we are looking for likelihood of failure based on some known informtion about the student. The model is strong in scenarios with clear margins of separation between the data sets and also works well in high dimensional spaces and in cases where the number of samples is lower than the number of dimensions. it is also memory efficient since it uses a subset of data points in the training set. This method is not suitable for large data sets as calculation can be time consuming and it also performs poorly when there is a significant amount of noise in the data.
# This may be a good model for the data because this is a classification problem that likely has relatively clear margins of separation. Also, the data set is not that large but has high dimensional space which SVMs excel at figuring out.
# 
# Ensemble Methods:
# Ensemble methods have been used recently to boost the prediction accuracy of the netflix recommendation engine by over 10% (https://www.cs.utah.edu/~piyush/teaching/ensemble_sdm10.ppt). Ensemble methods are advantageous because they are simple, require few parameters and are very flexible. They also provide a theoratical guarantee that the method wil work and can be good for data sets with high variance. Additionally they work well with high dimensionality and on data sets with large numbers of training examples and on non linear relationships. On the other hand, due to the presence of multiple models, they are computationally expensive, and can be very difficult to interpret.
# This may be a good model for the data since there are many dimensions and the data could be non linear.
# 
# 
# 

# ### Setup
# Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
# - `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
# - `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
# - `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
#  - This function will report the F<sub>1</sub> score for both the training and testing data separately.

# In[22]:

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# # Implementation: Model Performance Metrics
# With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
#  - Use a `random_state` for each model you use, if provided.
#  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Create the different training set sizes to be used to train each model.
#  - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
# - Fit each model with each training set size and make predictions on the test set (9 in total).  
# **Note:** Three tables are provided after the following code cell which can be used to store your results.

# In[18]:

# TODO: Import the three supervised learning models from sklearn
# from sklearn import model_A
from sklearn import tree
# from sklearn import model_B
from sklearn.svm import SVC
# from skearln import model_C
from sklearn.ensemble import AdaBoostClassifier

# TODO: Initialize the three models
clf_A = tree.DecisionTreeClassifier()
clf_B = SVC(random_state = 0)
clf_C = AdaBoostClassifier(random_state=0)


    
# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)

clf_A100 = train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
clf_A200 = train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
clf_A300 = train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
clf_B100 = train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
clf_B200 = train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
clf_B300 = train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
clf_C100 = train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
clf_C200 = train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
clf_C300 = train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)



# ### Tabular Results
# Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

# ** Classifer 1 - ?**  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |          0.0014         |              0.003     |        1         |   0.6195        |
# | 200               |        0.0012           |          0.001         |        1         |     0.7218      |
# | 300               |         0.0016          |             0.0002     |        1         |    0.6281       |
# 
# ** Classifer 2 - ?**  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |          0.0035         |              0.0007    |     0.8777       |  0.7746         |
# | 200               |        0.0029           |         0.0011         |    0.8679        |     0.7815      |
# | 300               |         0.0061          |             0.0016     |    0.8761        |    0.7838       |
# 
# ** Classifer 3 - ?**  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |          0.0798         |              0.0036    |     0.9481       |  0.7669         |
# | 200               |        0.0799           |         0.0036         |    0.8927        |     0.8281      |
# | 300               |         0.0882          |             0.0036     |    0.8637        |    0.7820       |

# ## Choosing the Best Model
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

# ### Question 3 - Choosing the Best Model
# *Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

# Looking at the above results, it seems that if we were to select a model based on performance alone, we should choose the ensemble metod Adaboost classifier as it achieves the highest F1 scores on the data. That being said, ir also takes the most time for testing predictions. Compared to SVM methods, it is slower at predicting by a factor of 3-4 in most cases and slower in training by a factor of 20-30. While SVM scores are slightly lower, I think the board would appreciate the lower computing costs associated with this method. Because of this fact, and the generally high F1 scores, especially when using the larger size training set, I would recommend this model to the board.
# 
# SVM is a good overall algorithm for this problem because of the high dimensionality of the problem. Luckily for us, there is not alot of data to go through. Additionally, SVMs guarantee a global optimum and are not very prone to overfitting. In the case of our data, I imagine that there are clear margins of separation which is helpful for SVMs. Finally, SVMs are quick compared to Ensemble methods and take little memory for storage, making them sutiable for this case.
# 

# ### Question 4 - Model in Layman's Terms
# *In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

# If you imagine all of our student data plotted on a chart, with x's representing the students that failed and 0's representing the students that passed, the SVM method is a way of finding a dividing line that separates the x's from the o's. Not only that, but the SVM method finds a line that separates this data correctly and that creates the highest margins between the line and the data points. This is the training stage. Once a line has been created, the algorithm can read new data, and classify the student based on which side of the line the student falls on. This is the testing phase.
# 
# The above description is for a regular SVM. One of the reasons SVMs are so powerful is that you dont need to draw a line necessarily. one can draw a circle or a squiggle or any other type of dividing line/surface/shape in the data. This is done using what is known as the kernel  trick and is basically a way of converting your data to a higher dimension in which the original non-linear sepaerator becomes linear. In this way, data sets with lots of parameters can be used to train the separator and make extremely accurate predictions without sacrificing too much computing time.

# In[ ]:

### Implementation: Model Tuning
#Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
#- Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
#- Create a dictionary of parameters you wish to tune for the chosen model.
# - Example: `parameters = {'parameter' : [list of values]}`.
#- Initialize the classifier you've chosen and store it in `clf`.
#- Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
 #- Set the `pos_label` parameter to the correct value!
#- Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
#- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.


# In[ ]:

# TODO: Import 'GridSearchCV' and 'make_scorer'

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Create the parameters list you wish to tune
parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1,2,3],'gamma':[1,2,3], 'coef0':[0,1,2]}

# TODO: Initialize the classifier
clf = SVC(random_state = 0)


# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label = 'yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train_300,y_train_300)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Question 5 - Final F<sub>1</sub> Score
# *What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

# **Answer: **

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
