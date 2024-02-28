First, we exclude outliers using by winsorizing the data with a threshold of 5 median absolute deviation. Anything below or above that threshold was convert to nans and interpolated using scipy's interpolate function.

Next, I create a boxcar based on the stimulus epoch delivery. On average, the stimulus duration is about 9 seconds on average.

Third, I use the canonical SCR from Bach's PSPM.
From that, we convolve the SCR function to the constructed event time boxcar function.

This is fed in the model for beta coefficient estimation, construct a glm,
