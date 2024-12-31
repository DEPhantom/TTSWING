import pandas as pd

file_path = '../dataset/TTSWING.csv'
data = pd.read_csv(file_path)

# Drop the 'id' column
data = data.drop(columns=['id'])

# Define categorical and numerical columns
categorical_columns = ['testmode', 'teststage', 'gender', 'handedness', 'hold racket handed']
numerical_columns = ['fileindex', 'count', 'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 
                     'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 
                     'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 
                     'a_fft', 'g_fft', 'a_psdx', 'g_psdx', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 
                     'a_entropy', 'g_entropy']

# Calculate statistical summary for numerical columns
statistics = data[numerical_columns].describe()
output_path = './numerical_stat_summary.csv'
statistics.to_csv(output_path)


# Calculate statistical summary for categorical columns
categorical_summary = data[categorical_columns].describe()
output_path_categorical = './categorical_stat_summary.csv'
categorical_summary.to_csv(output_path_categorical)
