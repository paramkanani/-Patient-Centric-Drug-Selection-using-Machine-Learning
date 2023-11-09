import pandas as pd
def load_dataset():
    dataset = pd.read_csv('drug_dataset.csv')
    return dataset

# Function to get the shape (number of rows and columns) of the loaded dataset
def dataframe_shape():
    # Calling the load_dataset() function to load the data
    # and then getting the shape of the DataFrame using the shape attribute
   dataset = load_dataset()
   shape = dataset.shape
   return shape

import pandas as pd
import numpy as np
from task1_1 import load_dataset

def sum_of_null_values():
    # Load the dataset and calculate the sum of null values for each column
   dataset = load_dataset() 
   sumofnull = dataset.isnull().sum()
   return sumofnull

import pandas as pd
import numpy as np
from task1_1 import load_dataset

def check_datatypes():
    # Load the dataset
    dataset = load_dataset()
    data = dataset.dtypes
    # Get data types of each column in the datase
    return data

import pandas as pd
from task1_1 import load_dataset

# Define a function to get descriptive statistics of the dataset
def data_describe():
    # Load the dataset and calculate descriptive statistics
    dataset = load_dataset()
    data = dataset.describe()
    # Return the descriptive statistics
    return data

# distribution of drugs
from task1_1 import load_dataset

# Define a function to check the count of the target variable in the dataset
def check_count_of_target_variable():
    # Load the dataset
    df = load_dataset()
    # Calculate the count of each unique value in the 'Drug' column ( it is the target variable)
    data = df['Drug'].value_counts()
    # Return the count of the target variable
    return data

from task1_1 import load_dataset
import pandas as pd

# Function to check for duplicate rows in the DataFrame
def check_duplicates():
    # Load the dataset
    df = load_dataset()
    # Calculate the number of duplicate rows using the duplicated() method and sum them
    duplicates = df.duplicated().sum()
    # Return the count of duplicate rows
    return duplicates

from task1_1 import load_dataset
import pandas as pd

# Function to drop duplicate rows from the DataFrame
def drop_duplicates():
    df = load_dataset()
    # Drop duplicate rows 
    df.drop_duplicates(inplace=True)
    return df

from task2_2 import drop_duplicates
import pandas as pd


# Function to drop unnecessary columns from the DataFrame
def drop_unnecessary_columns():
    # Get DataFrame with duplicate rows dropped
    df = drop_duplicates()

    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    data.drop(['Allergy','Medication_Duration'], axis=1, inplace=True)

    # Drop the 'Allergy','Medication_Duration' column from the copied DataFrame

    # Return the DataFrame with the 'Allergy','Medication_Duration' column removed
    return data

from task2_3 import drop_unnecessary_columns
import pandas as pd


def grouping_age():
    # Drop unnecessary columns from the DataFrame
    df = drop_unnecessary_columns()

    # Define a function to map ages to specific age groups
    '''The get_age_group(age) function categorizes input ages into specific groups. 
    If the age is 30 or below, the person belongs to the '0-30' group. If the age is over 30 
    but not exceeding 40, they are in the '30-40' group, and so on.If the age is more than 60, 
    the person is in the '60+' age group.'''
    def get_age_group(age):
        pass
        if age <= 30:
            return '0-30'
        elif 31 <= age <= 40:
            return '30-40'
        elif 41 <= age <= 50:
            return '40-50'
        elif 51 <= age <= 60:
            return '50-60'
        else:
            return '60+'
            
    df['AgeGroup'] = df['Age'].apply(get_age_group)

    # Apply the age grouping function to create a new 'AgeGroup' column

    # Drop the original 'Age' column since we've created the 'AgeGroup' column
    df.drop('Age', axis=1, inplace=True)
    # Return the DataFrame with the age groups
    return df

import pandas as pd
import numpy as np
from task2_4 import grouping_age


def grouping_Na_to_K():
    # Group ages first
    df = grouping_age()

    # Define a function to map Na-to-K ratio values to specific groups
    '''The get_na_to_k_group(value) function categorizes Na-to-K ratio values into distinct groups.
     If the value is 10 or less, it falls into the '5-10' group. If the value is more than 10 
     but not exceeding 15, it is in the '10-15' group, and so forth.If the value is more than 30, 
     it belongs to the '30+' group'''
    def get_na_to_k_group(na_to_k):
        pass
        if na_to_k <= 10:
           return '5-10'
        elif  10 < na_to_k <= 15:
           return '10-15'
        elif  15 < na_to_k <= 20:
           return '15-20'
        elif  20 < na_to_k <= 25:
           return '20-25'
        elif  25 < na_to_k <= 30: 
           return '25-30'
        else: 
           return '30+'    
           
    df['Na_to_K_groups'] = df['Na_to_K'].apply(get_na_to_k_group)
    
    df.drop('Na_to_K', axis=1, inplace=True)
    return df

from task2_5 import grouping_Na_to_K
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt


# Function to plot the count of each target class Drug in the DataFrame
def plot_target_count():
    # Get DataFrame with categorical variables encoded
    data = grouping_Na_to_K()

    # Plot the count of each target class using a countplot
    ax = sns.countplot(x='Drug', data = data)
    
    plt.savefig('plotting/task2.2.png')

    # Save the plot to a file (if needed)
    # plt.savefig('plotting/task2.2.png')

    # Show the countplot (optional)
    # plt.show()
    plt.show()

    # Return the axes object
    return ax

from task2_5 import grouping_Na_to_K
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Function to plot the count of AgeGroup in the DataFrame
def plot_countplot_AgeGroup():
    # Get DataFrame with categorical variables encoded
    data = grouping_Na_to_K()

    # Plot the count of each target class using a countplot
    ax = sns.countplot(x = 'AgeGroup', data=data)
    plt.savefig('plotting/task2.2.png')
    

    # Save the plot to a file (if needed)
    # plt.savefig('plotting/task2.2.png')

    # Show the countplot (optional)
    plt.show()

from task2_5 import grouping_Na_to_K
from sklearn.model_selection import train_test_split


def splitting_dataset():
    # Get DataFrame with categorical variables encoded
    data = grouping_Na_to_K()

    # Separate the features (X) and the target variable (y)
    X = data.drop('Drug', axis=1)
    y = data['Drug']

    # Split the data into training and testing sets[train_size=0.8, test_size=0.2, random_state=4]
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=4)

    # Return the split datasets
    return x_train, x_test, y_train, y_test

from task4_1 import splitting_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def get_dummies_smote_sampling():
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = splitting_dataset()

    # Encode categorical variables using one-hot encoding(get_dummies) on x_train and x_test.
    
    x_train, x_test, y_train, y_test = splitting_dataset()
    x_train = pd.get_dummies(pd.DataFrame(x_train))
    x_test = pd.get_dummies(pd.DataFrame(x_test))
    x_train = x_train.astype(int)
    x_test = x_test.astype(int)
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    return x_train, x_test, y_train, y_test

    # Convert the encoded data to integer type

    # Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the training data(x_train and y_train)


    # Return the balanced training set, testing set, and corresponding labels

import task4_2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def fit_logistic_regression(): # similarly guissan, deciesion tree, and  random forest 
    
    # Split the dataset into training and validation sets using task4_1.splitting_dataset()
    
    X_train, X_validation, Y_train, Y_validation = task4_2.get_dummies_smote_sampling()
   
    # Create a logistic_regression model
    
    logistic_regression = LogisticRegression()
    
     # Train (fit) the model using the training data
    
    logistic_regression.fit (X_train, Y_train)
    
    # Make predictions on the validation set using the trained model
    
    Y_pred = logistic_regression.predict(X_validation)

    # Calculate the accuracy of the model's predictions
    
    accuracy = accuracy_score(Y_validation, Y_pred, )

    # Calculate the precision of the model's predictions
    
    precision = precision_score(Y_validation, Y_pred, average ='weighted')

    # Calculate the recall of the model's predictions
    
    recall = recall_score(Y_validation, Y_pred, average = 'weighted')

    # Calculate the F1 score of the model's predictions
    
    f1 = f1_score(Y_validation, Y_pred, average = 'weighted')
    
    return accuracy, precision, recall, f1


import task4_2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def predict_model():
    data = [[1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0]]

    X_train, X_validation, Y_train, Y_validation = task4_2.get_dummies_smote_sampling()
    
    # Fit the model on the training data
    model = LogisticRegression()
    
    model.fit(X_train, Y_train)
    
    # Make a prediction on the new data using the your model
    prediction = model.predict(data)

    # Return the predicted class label
    return prediction
