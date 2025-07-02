# Calories-Burnt-Prediction-Model

## Project Description:

This project aims to predict the number of calories burned by an individual based on their exercise data, physical characteristics such as height, weight, age, and heart rate. The model uses a regression approach with the XGBoost algorithm to predict the calories burned during physical activity.

The project showcases the application of machine learning techniques, including data preprocessing, feature engineering, model optimization (hyperparameter tuning), and evaluation. This model can be used for health and fitness applications or personal fitness trackers to estimate calories burnt during exercise.

## Libraries Used
![Python](https://img.shields.io/badge/Python-3.11-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24.3-blue)
![pandas](https://img.shields.io/badge/pandas-2.0.3-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-green)


![image](https://github.com/user-attachments/assets/60de2be9-254c-4df4-b400-0f5c9be6a970)


# Table of Contents:

1. Dataset Visualizations

2. Model Details

3. Evaluation
4. Results
   
## 1. Dataset Visualizations: 

Exploratory Data Analysis (EDA) is an important step in understanding the dataset and visualizing patterns. In this project, several visualizations were used to analyze the data and gain insights about the relationships between different features. Below are the key visualizations included in this project:

1. Gender Distribution:

It helps identify any potential imbalances between the two genders, which is important for understanding the data composition.
sns.countplot(x='Gender', data=calories_data, hue='Gender')
plt.show()

![image](https://github.com/user-attachments/assets/58c4d7df-ecb6-4ef6-a99f-dbe5bf59f65c)



2. Correlation heatmap:

It visualizes the correlation between all pairs of features in the dataset. The correlation matrix indicates how strongly each feature is related to others, by examining the correlation matrix, we can identify highly correlated features that may lead to multicollinearity in the model. For example, if Height and Weight are highly correlated, we might consider removing one to avoid redundancy. The heatmap also helps identify which features are most strongly correlated with the target variable (Calories), which is important for feature selection.

![image](https://github.com/user-attachments/assets/3e0a26c1-5ece-457b-aae5-0030208abdd3)


3. Pairplot for Feature Relationships:
sns.pairplot(calories_data, hue='Gender')
plt.show()

![image](https://github.com/user-attachments/assets/8deaa510-922e-463f-8443-e770ead28fe7)

The pairplot helps to visually identify relationships between features for instance, whether Weight and Height are linearly related, and see how the distribution of features changes with respect to gender. It can reveal clusters of data points, trends, and potential outliers, aiding in further exploration or feature engineering.





## 2. Model Details:
The main model in this project is the XGBoost Regressor, a powerful gradient boosting algorithm, which has been widely used for regression tasks. The model is trained using features such as: Height, Weight, Age, Heart Rate, and Gender. Exercise Type (categorical variable encoded numerically)

Steps Involved:
Data Preprocessing:
* Handling missing values using imputation.

* Encoding categorical variables like Gender (male = 0, female = 1).

* Feature engineering: Adding BMI as an additional feature.

Model Training:

* Hyperparameter tuning using GridSearchCV to find the best parameters for XGBoost.

* Training the XGBoost model on the training data.

Model Evaluation:

* Evaluating the model using various metrics like MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R2 (Coefficient of Determination).

* Visualizing feature importance and the actual vs predicted calories burned.


## 3.Evaluation: 

After training the model, the following evaluation metrics will be printed:

* Mean Absolute Error (MAE) which is the average of the absolute differences between predicted and actual values.

* Root Mean Squared Error (RMSE) representing the square root of the average of squared differences between predicted and actual values.

* R2 Score, a statistical measure that indicates how well the model explains the variance in the target variable (calories burned).

* Cross-Validation, which evaluates the model's performance is also evaluated using cross-validation, providing a more robust evaluation.

* Feature Importance, a plot showing which features contribute most to predicting the calories burned.


## 4. Results:

![image](https://github.com/user-attachments/assets/ed461a0f-85d3-435a-aa4d-e340ca87e17d)

As shown, the data points are tightly clustered along the dashed diagonal line, which is a strong indicator of a highly accurate model. The pattern shows minimal spread or deviation, meaning your predictions are close to the actual values for most of the samples. So, there is no significant systematic bias (e.g., consistent underestimation or overestimation).

### So did the model achieve the aim?
Yes, it achieved the aim effectively. The goal was to predict calories burnt using features such as gender, height, weight, age, heart rate, and exercise details.
The model was optimized with GridSearchCV, which helped it achieve:

* Low MAE — meaning on average, predictions are very close to actual values.

* Low RMSE — indicating small average error magnitude.

* High R² score — close to 1 would mean very high accuracy.






