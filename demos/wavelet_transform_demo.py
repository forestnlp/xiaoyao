
import numpy as np
import pywt

# Create a simple signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

# Choose a wavelet
wavelet = 'db1' # Daubechies wavelet

# Perform a single level Discrete Wavelet Transform (DWT)
coeffs = pywt.dwt(signal, wavelet)

# The result is a tuple containing approximation and detail coefficients
approximation_coeffs, detail_coeffs = coeffs

print("Original Signal:")
print(signal)
print(f"\nWavelet used: {wavelet}")
print("\nApproximation Coefficients (cA):")
print(approximation_coeffs)
print("\nDetail Coefficients (cD):")
print(detail_coeffs)

# Reconstruct the signal from the coefficients
reconstructed_signal = pywt.idwt(approximation_coeffs, detail_coeffs, wavelet)

print("\nReconstructed Signal:")
print(reconstructed_signal)
