import numpy as np
import skimage.exposure as exposure
from scipy.signal import medfilt
from statsmodels.tsa.stattools import adfuller

# Rasched - here are various functions for use in other scripts - some just exist and I put in here after I didn't end up using
def moving_point_average(trace, window_size = 3):
    moving_avg_series = []
    
    for i in range(len(trace)):
        start_idx = max(0, i - window_size + 1)
        end_idx = min(i + 1, len(trace))
        window = trace[start_idx:end_idx]
        moving_avg = np.mean(window)
        moving_avg_series.append(moving_avg)
    
    return moving_avg_series


def apply_median_filter(data, window_size=3):
    assert window_size % 2 != 0
    return medfilt(data, kernel_size=window_size)


def normalize99(X):
    
    X = np.array(X)
    if np.max(X) > 0:
        X = X.copy()
        v_min, v_max = np.percentile(X[X != 0], (1, 99))
        X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

    return X


def rescale01(x):
        
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
        
    return x


def thresholding(trace):
    return np.array([1 if x > 0.5 else 0 for x in trace])


def remove_consecutive(arr):
    if not arr:
        return arr  # Return an empty array if the input is empty

    result = [arr[0]]  # Initialize the result with the first element of the array

    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            result.append(arr[i])  # If the current element is different from the previous one, add it to the result

    return result


def get_consecutive_ones_durations(arr):
    ones_durations = []  # Initialize an empty list to store durations of consecutive 1s
    current_duration = 0  # Initialize the current duration to 0

    for num in arr:
        if num == 1:
            current_duration += 1  # Increment the current duration for consecutive 1s
        elif current_duration > 0:
            ones_durations.append(current_duration)  # Append the current duration if it's greater than 0
            current_duration = 0  # Reset the current duration when encountering 0

    if current_duration > 0:
        ones_durations.append(current_duration)  # Append the last duration if the array ends with consecutive 1s

    return ones_durations


def check_stationarity(time_series):
    result = adfuller(time_series)
    p_value = result[1]
    if p_value <= 0.05:
        print("Time series is stationary (p-value <= 0.05)")
    else:
        print("Time series is non-stationary (p-value > 0.05)")