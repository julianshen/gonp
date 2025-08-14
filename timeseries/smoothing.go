package timeseries

import (
	"errors"
	"math"
)

// SimpleMovingAverage computes simple moving average with specified window and method
func (ts *TimeSeries) SimpleMovingAverage(window int, method string) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if window <= 0 {
		return nil, errors.New("window size must be positive")
	}
	if window > ts.Len() {
		return nil, errors.New("window size cannot exceed series length")
	}

	switch method {
	case "forward":
		return ts.forwardSMA(window)
	case "centered":
		return ts.centeredSMA(window)
	default:
		return nil, errors.New("unsupported SMA method: " + method)
	}
}

// forwardSMA computes forward moving average
func (ts *TimeSeries) forwardSMA(window int) (*TimeSeries, error) {
	values := make([]float64, ts.Len())

	for i := 0; i < ts.Len(); i++ {
		if i < window-1 {
			// Insufficient data for window
			values[i] = math.NaN()
		} else {
			// Calculate average for window ending at position i
			sum := 0.0
			count := 0

			for j := i - window + 1; j <= i; j++ {
				val := ts.At(j)
				if !math.IsNaN(val) {
					sum += val
					count++
				}
			}

			if count == 0 {
				values[i] = math.NaN()
			} else {
				values[i] = sum / float64(count)
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_sma")
}

// centeredSMA computes centered moving average
func (ts *TimeSeries) centeredSMA(window int) (*TimeSeries, error) {
	values := make([]float64, ts.Len())
	halfWindow := window / 2

	for i := 0; i < ts.Len(); i++ {
		if i < halfWindow || i >= ts.Len()-halfWindow {
			// Edge cases - insufficient data for centered window
			values[i] = math.NaN()
		} else {
			// Calculate centered average
			sum := 0.0
			count := 0

			for j := i - halfWindow; j <= i+halfWindow; j++ {
				val := ts.At(j)
				if !math.IsNaN(val) {
					sum += val
					count++
				}
			}

			if count == 0 {
				values[i] = math.NaN()
			} else {
				values[i] = sum / float64(count)
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_centered_ma")
}

// WeightedMovingAverage computes weighted moving average with custom weights
func (ts *TimeSeries) WeightedMovingAverage(weights []float64) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if len(weights) == 0 {
		return nil, errors.New("weights cannot be empty")
	}
	if len(weights) > ts.Len() {
		return nil, errors.New("weights length cannot exceed series length")
	}

	// Normalize weights
	weightSum := 0.0
	for _, w := range weights {
		weightSum += w
	}
	if weightSum == 0 {
		return nil, errors.New("sum of weights cannot be zero")
	}

	values := make([]float64, ts.Len())
	window := len(weights)

	for i := 0; i < ts.Len(); i++ {
		if i < window-1 {
			// Insufficient data for window
			values[i] = math.NaN()
		} else {
			// Calculate weighted average
			weightedSum := 0.0
			validWeightSum := 0.0

			for j := 0; j < window; j++ {
				val := ts.At(i - window + 1 + j)
				weight := weights[j]

				if !math.IsNaN(val) {
					weightedSum += val * weight
					validWeightSum += weight
				}
			}

			if validWeightSum == 0 {
				values[i] = math.NaN()
			} else {
				values[i] = weightedSum / validWeightSum
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_wma")
}

// ExponentialMovingAverage computes exponential moving average
func (ts *TimeSeries) ExponentialMovingAverage(alpha float64) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if alpha <= 0 || alpha > 1 {
		return nil, errors.New("alpha must be in range (0, 1]")
	}

	values := make([]float64, ts.Len())

	// Find first non-NaN value for initialization
	var ema float64
	initialized := false

	for i := 0; i < ts.Len(); i++ {
		val := ts.At(i)

		if !math.IsNaN(val) {
			if !initialized {
				// Initialize EMA with first valid value
				ema = val
				initialized = true
			} else {
				// Update EMA: EMA_t = alpha * X_t + (1-alpha) * EMA_{t-1}
				ema = alpha*val + (1-alpha)*ema
			}
			values[i] = ema
		} else {
			// Missing value - keep previous EMA or NaN if not initialized
			if initialized {
				values[i] = ema
			} else {
				values[i] = math.NaN()
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_ema")
}

// HoltLinearTrend implements Holt's double exponential smoothing for trending data
func (ts *TimeSeries) HoltLinearTrend(alpha, beta float64) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if alpha <= 0 || alpha > 1 {
		return nil, errors.New("alpha must be in range (0, 1]")
	}
	if beta < 0 || beta > 1 {
		return nil, errors.New("beta must be in range [0, 1]")
	}
	if ts.Len() < 2 {
		return nil, errors.New("need at least 2 observations for Holt's method")
	}

	values := make([]float64, ts.Len())

	// Find first two non-NaN values for initialization
	firstVal, secondVal := math.NaN(), math.NaN()
	firstIdx, secondIdx := -1, -1

	for i := 0; i < ts.Len() && (math.IsNaN(firstVal) || math.IsNaN(secondVal)); i++ {
		val := ts.At(i)
		if !math.IsNaN(val) {
			if math.IsNaN(firstVal) {
				firstVal = val
				firstIdx = i
			} else {
				secondVal = val
				secondIdx = i
			}
		}
	}

	if math.IsNaN(firstVal) || math.IsNaN(secondVal) {
		return nil, errors.New("need at least 2 non-NaN values for Holt's method")
	}

	// Initialize level and trend
	level := firstVal
	trend := (secondVal - firstVal) / float64(secondIdx-firstIdx)

	// Fill initial values
	for i := 0; i <= firstIdx; i++ {
		values[i] = level
	}

	// Apply Holt's method
	for i := firstIdx + 1; i < ts.Len(); i++ {
		val := ts.At(i)

		if !math.IsNaN(val) {
			// Update equations
			prevLevel := level
			level = alpha*val + (1-alpha)*(level+trend)
			trend = beta*(level-prevLevel) + (1-beta)*trend
		}
		// else keep previous level and trend

		values[i] = level + trend
	}

	return NewTimeSeries(values, ts.index, ts.name+"_holt")
}

// HoltWinters implements triple exponential smoothing for seasonal data
func (ts *TimeSeries) HoltWinters(alpha, beta, gamma float64, seasonLength int, seasonType string) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if alpha <= 0 || alpha > 1 {
		return nil, errors.New("alpha must be in range (0, 1]")
	}
	if beta < 0 || beta > 1 {
		return nil, errors.New("beta must be in range [0, 1]")
	}
	if gamma < 0 || gamma > 1 {
		return nil, errors.New("gamma must be in range [0, 1]")
	}
	if seasonLength <= 0 {
		return nil, errors.New("season length must be positive")
	}
	if seasonType != "additive" && seasonType != "multiplicative" {
		return nil, errors.New("season type must be 'additive' or 'multiplicative'")
	}
	if ts.Len() < 2*seasonLength {
		return nil, errors.New("need at least 2 complete seasons for Holt-Winters")
	}

	values := make([]float64, ts.Len())

	// Initialize seasonal indices (simplified initialization)
	seasonal := make([]float64, seasonLength)
	if seasonType == "additive" {
		// Initialize seasonal factors to 0 for additive
		for i := range seasonal {
			seasonal[i] = 0.0
		}
	} else {
		// Initialize seasonal factors to 1 for multiplicative
		for i := range seasonal {
			seasonal[i] = 1.0
		}
	}

	// Initialize level and trend from first few observations
	level := 0.0
	count := 0
	for i := 0; i < seasonLength && i < ts.Len(); i++ {
		val := ts.At(i)
		if !math.IsNaN(val) {
			level += val
			count++
		}
	}
	if count == 0 {
		return nil, errors.New("no valid values found for initialization")
	}
	level /= float64(count)

	trend := 0.0 // Start with no trend assumption

	// Apply Holt-Winters method
	for i := 0; i < ts.Len(); i++ {
		val := ts.At(i)
		seasonIdx := i % seasonLength

		if !math.IsNaN(val) && i >= seasonLength {
			// Update equations for Holt-Winters
			var prevLevel float64
			if seasonType == "additive" {
				prevLevel = level
				level = alpha*(val-seasonal[seasonIdx]) + (1-alpha)*(level+trend)
				trend = beta*(level-prevLevel) + (1-beta)*trend
				seasonal[seasonIdx] = gamma*(val-level-trend) + (1-gamma)*seasonal[seasonIdx]
				values[i] = level + trend + seasonal[seasonIdx]
			} else {
				prevLevel = level
				level = alpha*(val/seasonal[seasonIdx]) + (1-alpha)*(level+trend)
				trend = beta*(level-prevLevel) + (1-beta)*trend
				seasonal[seasonIdx] = gamma*(val/(level+trend)) + (1-gamma)*seasonal[seasonIdx]
				values[i] = (level + trend) * seasonal[seasonIdx]
			}
		} else {
			// Use current estimates for missing values or initialization period
			if seasonType == "additive" {
				values[i] = level + trend + seasonal[seasonIdx]
			} else {
				values[i] = (level + trend) * seasonal[seasonIdx]
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_hw")
}

// LOWESS implements locally weighted scatterplot smoothing
func (ts *TimeSeries) LOWESS(bandwidth float64, iterations int) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if bandwidth <= 0 || bandwidth > 1 {
		return nil, errors.New("bandwidth must be in range (0, 1]")
	}
	if iterations < 1 {
		return nil, errors.New("iterations must be at least 1")
	}

	n := ts.Len()
	if n < 3 {
		return nil, errors.New("need at least 3 observations for LOWESS")
	}

	// Create x values (indices) and y values
	x := make([]float64, n)
	y := make([]float64, n)
	validIndices := make([]int, 0, n)

	for i := 0; i < n; i++ {
		val := ts.At(i)
		if !math.IsNaN(val) {
			x[len(validIndices)] = float64(i)
			y[len(validIndices)] = val
			validIndices = append(validIndices, i)
		}
	}

	if len(validIndices) < 3 {
		return nil, errors.New("need at least 3 non-NaN values for LOWESS")
	}

	// Trim arrays to valid data
	x = x[:len(validIndices)]
	y = y[:len(validIndices)]

	// LOWESS smoothing
	smoothed := make([]float64, len(validIndices))
	copy(smoothed, y) // Initialize with original values

	for iter := 0; iter < iterations; iter++ {
		newSmoothed := make([]float64, len(validIndices))

		for i := 0; i < len(validIndices); i++ {
			newSmoothed[i] = ts.lowessPoint(x, smoothed, i, bandwidth)
		}

		copy(smoothed, newSmoothed)
	}

	// Map back to original indices
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = math.NaN() // Default to NaN
	}

	for i, originalIdx := range validIndices {
		values[originalIdx] = smoothed[i]
	}

	return NewTimeSeries(values, ts.index, ts.name+"_lowess")
}

// lowessPoint computes LOWESS smoothed value for a single point
func (ts *TimeSeries) lowessPoint(x, y []float64, targetIdx int, bandwidth float64) float64 {
	n := len(x)
	k := int(math.Max(1, bandwidth*float64(n))) // Number of neighbors

	// Find k nearest neighbors
	distances := make([]struct {
		dist float64
		idx  int
	}, n)

	targetX := x[targetIdx]
	for i := 0; i < n; i++ {
		distances[i] = struct {
			dist float64
			idx  int
		}{math.Abs(x[i] - targetX), i}
	}

	// Sort by distance
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if distances[i].dist > distances[j].dist {
				distances[i], distances[j] = distances[j], distances[i]
			}
		}
	}

	// Get k nearest neighbors
	if k > n {
		k = n
	}

	maxDist := distances[k-1].dist
	if maxDist == 0 {
		maxDist = 1.0 // Avoid division by zero
	}

	// Weighted linear regression
	sumW := 0.0
	sumWX := 0.0
	sumWY := 0.0
	sumWXX := 0.0
	sumWXY := 0.0

	for i := 0; i < k; i++ {
		idx := distances[i].idx
		dist := distances[i].dist

		// Tricube weight function
		u := dist / maxDist
		if u >= 1 {
			continue // Weight would be 0
		}
		weight := math.Pow(1-u*u*u, 3)

		xi := x[idx]
		yi := y[idx]

		sumW += weight
		sumWX += weight * xi
		sumWY += weight * yi
		sumWXX += weight * xi * xi
		sumWXY += weight * xi * yi
	}

	if sumW == 0 {
		return y[targetIdx] // Fallback to original value
	}

	// Solve for linear fit: y = a + b*x
	denominator := sumW*sumWXX - sumWX*sumWX
	if math.Abs(denominator) < 1e-10 {
		// Fallback to weighted mean
		return sumWY / sumW
	}

	a := (sumWY*sumWXX - sumWXY*sumWX) / denominator
	b := (sumW*sumWXY - sumWX*sumWY) / denominator

	return a + b*targetX
}

// SavitzkyGolay implements Savitzky-Golay polynomial smoothing
func (ts *TimeSeries) SavitzkyGolay(window, degree int) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if window%2 == 0 {
		return nil, errors.New("window size must be odd")
	}
	if window < 3 {
		return nil, errors.New("window size must be at least 3")
	}
	if degree < 0 {
		return nil, errors.New("polynomial degree must be non-negative")
	}
	if degree >= window {
		return nil, errors.New("polynomial degree must be less than window size")
	}
	if ts.Len() < window {
		return nil, errors.New("time series length must be at least window size")
	}

	values := make([]float64, ts.Len())
	halfWindow := window / 2

	// Generate Savitzky-Golay coefficients for center point
	coeffs := ts.savitzkyGolayCoeffs(window, degree, 0) // 0 = center point

	for i := 0; i < ts.Len(); i++ {
		if i < halfWindow || i >= ts.Len()-halfWindow {
			// Edge case: use original value or implement boundary conditions
			values[i] = ts.At(i)
		} else {
			// Apply Savitzky-Golay filter
			sum := 0.0
			validCount := 0
			coeffSum := 0.0

			for j := -halfWindow; j <= halfWindow; j++ {
				val := ts.At(i + j)
				if !math.IsNaN(val) {
					coeff := coeffs[j+halfWindow]
					sum += coeff * val
					coeffSum += coeff
					validCount++
				}
			}

			if validCount == 0 {
				values[i] = math.NaN()
			} else if validCount == window {
				values[i] = sum // All values were valid, use direct sum
			} else {
				// Normalize for missing values
				if coeffSum != 0 {
					values[i] = sum / coeffSum * coeffSum // Maintain coefficient sum property
				} else {
					values[i] = math.NaN()
				}
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_savgol")
}

// savitzkyGolayCoeffs computes Savitzky-Golay coefficients
func (ts *TimeSeries) savitzkyGolayCoeffs(window, degree, derivative int) []float64 {
	// This is a simplified implementation for derivative=0 (smoothing)
	// For full implementation, would need matrix operations

	halfWindow := window / 2
	coeffs := make([]float64, window)

	if degree == 0 {
		// Simple moving average
		weight := 1.0 / float64(window)
		for i := range coeffs {
			coeffs[i] = weight
		}
	} else if degree == 1 {
		// Linear fit weights (approximation)
		sum := 0.0
		for i := 0; i < window; i++ {
			x := float64(i - halfWindow)
			coeffs[i] = 1.0 - 2.0*x*x/float64(window*window)
			sum += coeffs[i]
		}
		// Normalize to sum to 1
		for i := range coeffs {
			coeffs[i] /= sum
		}
	} else {
		// Higher degree - use simplified quadratic approximation
		sum := 0.0
		for i := 0; i < window; i++ {
			x := float64(i - halfWindow)
			normalizedX := x / float64(halfWindow)
			coeffs[i] = 1.0 - normalizedX*normalizedX
			if coeffs[i] < 0 {
				coeffs[i] = 0
			}
			sum += coeffs[i]
		}
		// Normalize
		for i := range coeffs {
			coeffs[i] /= sum
		}
	}

	return coeffs
}

// KernelSmoothing implements kernel-based smoothing
func (ts *TimeSeries) KernelSmoothing(kernelType string, bandwidth float64) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if bandwidth <= 0 {
		return nil, errors.New("bandwidth must be positive")
	}
	if kernelType != "gaussian" && kernelType != "uniform" && kernelType != "triangular" {
		return nil, errors.New("supported kernel types: gaussian, uniform, triangular")
	}

	values := make([]float64, ts.Len())

	for i := 0; i < ts.Len(); i++ {
		sum := 0.0
		weightSum := 0.0

		// Apply kernel to all points
		for j := 0; j < ts.Len(); j++ {
			val := ts.At(j)
			if !math.IsNaN(val) {
				distance := math.Abs(float64(i) - float64(j))
				weight := ts.kernelFunction(kernelType, distance, bandwidth)

				sum += weight * val
				weightSum += weight
			}
		}

		if weightSum > 0 {
			values[i] = sum / weightSum
		} else {
			values[i] = math.NaN()
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_kernel")
}

// kernelFunction computes kernel weight for given distance
func (ts *TimeSeries) kernelFunction(kernelType string, distance, bandwidth float64) float64 {
	u := distance / bandwidth

	switch kernelType {
	case "gaussian":
		// Gaussian kernel: (1/sqrt(2π)) * exp(-0.5 * u²)
		return math.Exp(-0.5*u*u) / math.Sqrt(2*math.Pi)

	case "uniform":
		// Uniform kernel: 0.5 if |u| <= 1, 0 otherwise
		if u <= 1.0 {
			return 0.5
		}
		return 0.0

	case "triangular":
		// Triangular kernel: (1 - |u|) if |u| <= 1, 0 otherwise
		if u <= 1.0 {
			return 1.0 - u
		}
		return 0.0

	default:
		return 0.0
	}
}
