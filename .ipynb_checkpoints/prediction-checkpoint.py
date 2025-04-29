import pandas as pd
import pickle
import os

# Load Trained Models
models_name = 'model.pkl'
models_folder = 'models'
models_path = os.path.join(models_folder, models_name)

with open(models_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Adding Input Data
x_input = pd.DataFrame({
    'Age': [35],
    'BusinessTravel': ['Travel_Rarely'],
    'DailyRate': [1100],
    'Department': ['Sales'],
    'DistanceFromHome': [5],
    'Education': [3],
    'EducationField': ['Life Sciences'],
    'EnvironmentSatisfaction': [3],
    'Gender': ['Male'],
    'HourlyRate': [70],
    'JobInvolvement': [3],
    'JobLevel': [2],
    'JobRole': ['Sales Executive'],
    'JobSatisfaction': [4],
    'MaritalStatus': ['Married'],
    'MonthlyIncome': [5000],
    'MonthlyRate': [20000],
    'NumCompaniesWorked': [1],
    'OverTime': ['No'],
    'PercentSalaryHike': [12],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [4],
    'StockOptionLevel': [1],
    'TotalWorkingYears': [10],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [3],
    'YearsAtCompany': [5],
    'YearsInCurrentRole': [2],
    'YearsSinceLastPromotion': [1],
    'YearsWithCurrManager': [3],
    'StabilityInRole': [0.6],
    'LoyaltyToManager': [0.6],
    'AvgTrainingPerYear': [1.5],
    'AgeWhenStarted': [25],
    'AvgYearsPerCompany': [10],
    'IncomePerKm': [200],
    'CompanyLoyalty': [0.7],
    'PromotionFrequency': [2],
    'AvgMonthlyIncomePerYear': [2000]
})

# Models Prediction
y_pred = model.predict(x_input)

print("Prediction Result:",y_pred)
