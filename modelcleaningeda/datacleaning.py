import pandas as pd
import numpy as np

data_frame = pd.read_csv('healthcare-dataset-stroke-data - healthcare-dataset-stroke-data.csv')

data_frame = data_frame.drop(['id'], axis=1)

pd.set_option('future.no_silent_downcasting', True)

mean_bmi = data_frame['bmi'].mean()
data_frame.replace(['N/A'], np.nan, inplace=False)
data_frame['bmi'] = data_frame['bmi'].fillna(mean_bmi, inplace=False)

data_frame['gender'] = data_frame['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})

data_frame['married'] = data_frame['ever_married'].replace({'Yes': 0, 'No': 1})

data_frame['job_type'] = data_frame['work_type'].replace({'children': 0, 'Never_worked': 1, 'Self-employed': 2, 'Private': 3, 'Govt_job': 4})

data_frame['location'] = data_frame['Residence_type'].replace({'Urban': 0, 'Rural': 1})

data_frame['smoke'] = data_frame['smoking_status'].replace({'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3})

data_frame.to_csv('processed.csv', index=False)
