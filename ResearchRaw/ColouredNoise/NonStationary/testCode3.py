import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from datetime import datetime

# Create a new directory based on the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
directory = f"C:\\Users\\HP\\OneDrive\\Documents\\Codes\\python\\ResearchRaw\\ColouredNoise\\NonStationary\\{current_time}_NonStationaryAnalysis"
os.makedirs(directory, exist_ok=True)

# Generate synthetic noise for demonstration
np.random.seed(0)
n = 1000
# Non-stationary noise example (random walk)
non_stationary_noise = np.cumsum(np.random.normal(size=n))
# Stationary noise example (white noise)
stationary_noise = np.random.normal(size=n)

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}):', value)
    return {'ADF Statistic': result[0], 'p-value': result[1], **result[4]}

# Function to perform KPSS test
def kpss_test(series, regression='c'):
    result = kpss(series, regression=regression)
    print('KPSS Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[3].items():
        print(f'Critical Value ({key}):', value)
    return {'KPSS Statistic': result[0], 'p-value': result[1], **result[3]}

# Perform tests on non-stationary noise
adf_results_non_stationary = adf_test(non_stationary_noise)
kpss_results_non_stationary = kpss_test(non_stationary_noise)

# Perform tests on stationary noise
adf_results_stationary = adf_test(stationary_noise)
kpss_results_stationary = kpss_test(stationary_noise)

# Calculate rolling mean and variance
rolling_mean_non_stationary = pd.Series(non_stationary_noise).rolling(window=50).mean()
rolling_var_non_stationary = pd.Series(non_stationary_noise).rolling(window=50).var()
rolling_mean_stationary = pd.Series(stationary_noise).rolling(window=50).mean()
rolling_var_stationary = pd.Series(stationary_noise).rolling(window=50).var()

# Save results to CSV
results = {
    'Non-Stationary ADF': adf_results_non_stationary,
    'Non-Stationary KPSS': kpss_results_non_stationary,
    'Stationary ADF': adf_results_stationary,
    'Stationary KPSS': kpss_results_stationary
}
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(directory, 'stationarity_tests_results.csv'))

# Plot and save rolling statistics
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(non_stationary_noise, label='Non-Stationary Noise')
plt.plot(rolling_mean_non_stationary, label='Rolling Mean', color='red')
plt.plot(rolling_var_non_stationary, label='Rolling Variance', color='green')
plt.legend(loc='best')
plt.title('Non-Stationary Noise: Rolling Mean & Variance')

plt.subplot(2, 1, 2)
plt.plot(stationary_noise, label='Stationary Noise')
plt.plot(rolling_mean_stationary, label='Rolling Mean', color='red')
plt.plot(rolling_var_stationary, label='Rolling Variance', color='green')
plt.legend(loc='best')
plt.title('Stationary Noise: Rolling Mean & Variance')

plt.tight_layout()
plot_path = os.path.join(directory, 'rolling_statistics_plot.png')
plt.savefig(plot_path)
plt.show()

# Save the script
script_path = os.path.join(directory, 'stationarity_analysis.py')
with open(script_path, 'w') as f:
    f.write(open(__file__).read())