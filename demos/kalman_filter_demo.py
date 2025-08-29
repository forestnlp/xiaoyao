
import numpy as np
from pykalman import KalmanFilter

# Create some noisy data
np.random.seed(0)
# True values
true_states = np.linspace(0, 10, 50)
# Measurements with noise
measurements = true_states + np.random.normal(0, 1, 50)

# Create a Kalman Filter
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

# Use the EM algorithm to learn the parameters
kf = kf.em(measurements, n_iter=5)

# Smooth the data
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

print("Original Measurements:")
print(measurements)
print("\nSmoothed States:")
print(smoothed_state_means.flatten())
