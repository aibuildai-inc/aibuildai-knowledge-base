# 25th Place - Polynomnial Fitting / Nelder-Mead

Competition: ariel-data-challenge-2025
Rank: #25
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/25th-place-polynomnial-fitting-nelder-mead

## 1. Data Preprocessing


Data calibration was done in accordance with the organizers' suggested solution. The sole modification was to utilize a subset of the sensor data's pixels. Consequently, the array slices [:, 10:22, 39:321], [:, 10:22, :] were utilized for the AIRS-CH0 and FGS1 sensors.


## 2. Detecting Ingress / Egress.


Ingress and egress were computed using an algorithm based on the gradient approach. Windows of `phase_a_start` till `phase_a_end` and `phase_b_start` till `phase_b_end` were excluded  prior to polynomnial fitting. In case of invalid detection of `phase_a` i used only egress part to estimate depth and vice versa. In case of invalid detection of `phase_a` and `phase_b` i did not make any predictions.


    def detect_breakpoints(flux, verbose=False):
        """
        Detect the phases in the airs flux signal by calculating the gradient of the detrended signal.
        """
    
        # Mean and filter signal
        flux = flux.mean(axis=-1)

        # Find middle of transit
        min_index = np.argmin(flux)
        in_a = flux[:min_index]
        in_b = flux[min_index:]

        # Compute the gradients of both halves
        gradient_a = np.gradient(in_a, edge_order=1)

        gradient_2a = np.gradient(gradient_a, edge_order=1)
        gradient_2b = np.gradient(gradient_b, edge_order=1)

        # Identify the phase indices based on the gradients
        phase_a = gradient_a[1:].argmin() + 1
        phase_b = gradient_b[:-1].argmax() + len(gradient_a)

        phase_a_start = gradient_2a.argmin()
        phase_a_end = gradient_2a.argmax()
    
        phase_b_start = gradient_2b.argmax() + len(gradient_a)
        phase_b_end = gradient_2b.argmin() + len(gradient_a)
    
        breakpoints = {'center': min_index,
                       'phase_a_start': phase_a_start - 1,
                       'phase_a': phase_a,
                       'phase_a_end': phase_a_end + 1,
                       'phase_b_start': phase_b_start - 1,
                       'phase_b': phase_b,
                       'phase_b_end': phase_b_end + 1}

        return breakpoints

.png?generation=1758973537444006&alt=media)

## 3. Transit depths estimation.
The transit depths were determined using Polynomnial Fitting and Nelder-Mead optimization. 


    def F(depth, phase_a_data, phase_x_data, phase_b_data, polyorder):
        y = np.concatenate((
            phase_a_data,
            phase_x_data * depth,
            phase_b_data
        ))

        X = np.arange(len(y))

        z = np.polyfit(X, y, polyorder)
        p = np.poly1d(z)
        score = np.abs(p(X) - y).mean()
        return score

### 3.1  Dynamic Spectrum Estimation
The transit depth were determined using Polynomnial Fitting (up to 3rd polynomnial grade)



### 3.2 Flat Spectrum

I began the approach by forecasting solely the flat outputs because the majority of the spectra in the data have relatively minor oscillations.

The transit depth were determined at first using Polynomnial Fitting (up to 18th polynomnial grade) just once acrossed averaged wavelenghts. 
Employing polynomial fitting up to degree 18 may appear counterintuitive; however, it yielded superior results compared to fitting up to degree 3 / 4.

  


### 3.3 Transit Depths Fluctuations

To accurately forecast the final spectrum, the amount of dynamics in depth measurements was determined. Measurements were computed on non-smoothed time-series.

Zero Crossings Rate - (ZCR) is a feature used to measure the number of times a signal crosses the zero level (i.e., changes its sign).

      def compute_zcr(signal, burn_in=20):
          return librosa.zero_crossings(signal - signal.mean())[:-burn_in].sum()

 

## 4. Final Solution - Combining - Flat Spectrum / Dynamic Spectrum and Hyperparameters - Zero-Crossings-Rate / Orbital Period / Transit Wide  / Transit Correctness.

### To put it briefly:

* A flat spectrum with low sigma was employed as a solution to the problem if the computed transit depths had minimal fluctuations (High ZCR) because there was little chance of a prediction error.

* Since there was a significant chance of prediction error, Non-Flattened (dynamic) spectra with larger sigma were employed as a solution if the computed transit depths exhibited significant swings (Low ZCR).

* The predicted transit depth is modeled as a cubic polynomial function of the scaled signal, using coefficients a0, a1, a2, and a3.
   signal = a0 + a1 * signal + a2 * signal ** 2 + a3 * signal ** 3

The Nelder-Mead algorithm and the following code were used to determine the values of sigma / signal coefficients and the amount of non-flattening depths to employ.

    def operator(signal, zcr, additional_features, dynamic_depths, args):
        """
        Simplified version:
        - Combine base parameters with ZCR and features to modify sigma.
        - Adjust the signal based on features.
        - Add residual depths to flatten based on zcr if in 'airs' mode.
        - Finally, scale the signal with polynomial coefficients.
        """

        # Unpack parameters
        a0, a1, a2, a3, cross_sigma, sigma, residual_coef, post_sigma = args[:8]
        feature_coeffs = args[8:]  # Coefficients for additional features

        # Initialize sigma with base and ZCR influence
        sigma = sigma + cross_sigma * zcr

        # Incorporate effects of additional features into sigma and signal
        for i, (feature_name, feature_value) in enumerate(additional_features.items()):
            sigma += feature_coeffs[i] * feature_value
            signal += feature_coeffs[i] * feature_value

        # Additional correction if mode is 'airs'
        if ScalerSigma.mode == 'airs':
            # Compute residual (difference from mean) and flatten it
            residual = dynamic_depths - np.mean(dynamic_depths, axis=-1, keepdims=True)
            residual = savgol_filter(residual, window_length=10, polyorder=1)

            # Add residual to the signal scaled by zcr
            signal += zcr[:, None] * residual

        # Final scaling of the signal
        scaled_signal = a0 + a1 * signal + a2 * signal ** 2 + a3 * signal ** 3

        return scaled_signal, sigma

### 4.1 Example Predictions



