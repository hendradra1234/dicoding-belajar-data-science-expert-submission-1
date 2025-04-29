import pandas as pd
import pickle
import os

# Load Trained Models
# Folders
models_folder = 'models'

# Filename
models_name = 'model.pkl'

# File Path
models_path = os.path.join(models_folder, models_name)

# Load Models
with open(models_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Make Input Dayaset
input_dataset = {
    'EmployeeId':'1465', # 1
    'Age': [28], # 2
    'BusinessTravel': ['Travel_Rarely'], # 3
    'DailyRate': [1366], # 4
    'Department': ['Research & Development'], # 5
    'DistanceFromHome': [24], # 6
    'Education': [2], # 7
    'EducationField': ['Technical Degree'], # 8
    'EnvironmentSatisfaction': [2], # 9
    'Gender': ['Male'], # 10
    'HourlyRate': [72], # 11
    'JobInvolvement': [2], # 12
    'JobLevel': [3], # 13
    'JobRole': ['Healthcare Representative'], # 14
    'JobSatisfaction': [1], # 15
    'MaritalStatus': ['Single'], # 16
    'MonthlyIncome': [8722], # 17
    'MonthlyRate': [12355], # 18
    'NumCompaniesWorked': [1], # 19
    'OverTime': ['No'], # 20
    'PercentSalaryHike': [12], # 21
    'PerformanceRating': [3], # 22
    'RelationshipSatisfaction': [1], # 23
    'StockOptionLevel': [0], # 24
    'TotalWorkingYears': [10], # 25
    'TrainingTimesLastYear': [2], # 26
    'WorkLifeBalance': [2], # 27
    'YearsAtCompany': [10], # 28
    'YearsInCurrentRole': [7], # 29
    'YearsSinceLastPromotion': [1], # 30
    'YearsWithCurrManager': [9], # 31
    'StabilityInRole': [0.70], # 32
    'LoyaltyToManager': [0.9], # 33
    'AvgTrainingPerYear': [1.5], # 34
    'AgeWhenStarted': [18], # 35
    'AvgYearsPerCompany': [0.2], # 36
    'IncomePerKm': [200], # 37
    'CompanyLoyalty': [0.7], # 38
    'PromotionFrequency': [2], # 39
    'AvgMonthlyIncomePerYear': [2000] # 40
}

# Convert Input Data into Dataframe
x_input = pd.DataFrame(input_dataset)

# Models Prediction
y_pred = model.predict(x_input)

# Outputing Result
print("Prediction Result:", y_pred)
