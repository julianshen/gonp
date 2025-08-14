package timeseries

import (
	"errors"
	"fmt"
	"math"
)

// DecompositionResult represents the result of time series decomposition
type DecompositionResult struct {
	Original *TimeSeries // Original time series
	Trend    *TimeSeries // Trend component
	Seasonal *TimeSeries // Seasonal component
	Residual *TimeSeries // Residual/irregular component
	Period   int         // Seasonal period used
	Type     string      // "additive" or "multiplicative"
}

// Decompose performs classical time series decomposition
func (ts *TimeSeries) Decompose(decompositionType string, period int) (*DecompositionResult, error) {
	// Parameter validation
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if period <= 1 {
		return nil, errors.New("period must be greater than 1")
	}
	if period >= ts.Len() {
		return nil, errors.New("period cannot be greater than or equal to series length")
	}
	if decompositionType != "additive" && decompositionType != "multiplicative" {
		return nil, errors.New("decomposition type must be 'additive' or 'multiplicative'")
	}

	// Step 1: Extract trend using centered moving average
	trend, err := ts.extractCenteredMovingAverage(period)
	if err != nil {
		return nil, fmt.Errorf("trend extraction failed: %v", err)
	}

	// Step 2: Remove trend to get seasonal + residual
	var detrended *TimeSeries
	if decompositionType == "additive" {
		// detrended = original - trend
		detrended, err = ts.subtractSeries(trend)
	} else {
		// detrended = original / trend
		detrended, err = ts.divideSeries(trend)
	}
	if err != nil {
		return nil, fmt.Errorf("detrending failed: %v", err)
	}

	// Step 3: Extract seasonal pattern from detrended series
	seasonal, err := ts.extractSeasonalPattern(detrended, period, decompositionType)
	if err != nil {
		return nil, fmt.Errorf("seasonal extraction failed: %v", err)
	}

	// Step 4: Calculate residuals
	var residual *TimeSeries
	if decompositionType == "additive" {
		// residual = original - trend - seasonal
		temp, err := ts.subtractSeries(trend)
		if err != nil {
			return nil, fmt.Errorf("residual calculation failed: %v", err)
		}
		residual, err = temp.subtractSeries(seasonal)
	} else {
		// residual = original / (trend * seasonal)
		temp, err := trend.multiplySeries(seasonal)
		if err != nil {
			return nil, fmt.Errorf("residual calculation failed: %v", err)
		}
		residual, err = ts.divideSeries(temp)
	}
	if err != nil {
		return nil, fmt.Errorf("residual calculation failed: %v", err)
	}

	// Create copy of original for result
	original := ts.copy()

	return &DecompositionResult{
		Original: original,
		Trend:    trend,
		Seasonal: seasonal,
		Residual: residual,
		Period:   period,
		Type:     decompositionType,
	}, nil
}

// DecomposeAuto performs decomposition with automatic period detection
func (ts *TimeSeries) DecomposeAuto(decompositionType string) (*DecompositionResult, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}

	// Detect seasonal period automatically
	period, err := ts.DetectSeasonalPeriod()
	if err != nil {
		return nil, fmt.Errorf("could not detect seasonal period: %v", err)
	}

	// Use detected period for decomposition
	return ts.Decompose(decompositionType, period)
}

// DetectSeasonalPeriod automatically detects the dominant seasonal period using autocorrelation
func (ts *TimeSeries) DetectSeasonalPeriod() (int, error) {
	if ts == nil {
		return 0, errors.New("time series cannot be nil")
	}
	if ts.Len() < 6 {
		return 0, errors.New("need at least 6 observations to detect seasonal period")
	}

	maxPeriod := ts.Len() / 2 // Don't test periods longer than half the series
	if maxPeriod > 50 {
		maxPeriod = 50 // Limit search to reasonable periods
	}

	bestPeriod := 2
	bestAutocorr := 0.0

	// Test different periods from 2 to maxPeriod
	for period := 2; period <= maxPeriod; period++ {
		autocorr := ts.computeAutocorrelation(period)
		if autocorr > bestAutocorr {
			bestAutocorr = autocorr
			bestPeriod = period
		}
	}

	// Require minimum autocorrelation to consider it seasonal
	if bestAutocorr < 0.3 {
		return 0, errors.New("no significant seasonal pattern detected")
	}

	return bestPeriod, nil
}

// DetectMultipleSeasonalPeriods detects multiple seasonal periods
func (ts *TimeSeries) DetectMultipleSeasonalPeriods(maxPeriods int) ([]int, error) {
	// TDD Red phase - this should fail initially
	return nil, errors.New("DetectMultipleSeasonalPeriods not yet implemented")
}

// ExtractTrend extracts the trend component using specified method
func (ts *TimeSeries) ExtractTrend(method string, window int) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if ts.Len() == 0 {
		return nil, errors.New("cannot extract trend from empty time series")
	}

	switch method {
	case "moving_average":
		if window <= 0 {
			return nil, errors.New("window size must be positive for moving average")
		}
		if window >= ts.Len() {
			return nil, errors.New("window size cannot exceed series length")
		}
		return ts.extractCenteredMovingAverage(window)

	case "linear":
		// Linear trend fitting using least squares
		return ts.extractLinearTrend()

	default:
		return nil, errors.New("unsupported trend extraction method: " + method)
	}
}

// Detrend removes trend from the time series
func (ts *TimeSeries) Detrend(method string) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}

	// Extract trend using specified method
	trend, err := ts.ExtractTrend(method, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to extract trend: %v", err)
	}

	// Subtract trend from original series
	return ts.subtractSeries(trend)
}

// DecompositionResult methods

// GetPeriod returns the seasonal period used in decomposition
func (dr *DecompositionResult) GetPeriod() int {
	if dr == nil {
		return 0
	}
	return dr.Period
}

// GetType returns the decomposition type
func (dr *DecompositionResult) GetType() string {
	if dr == nil {
		return ""
	}
	return dr.Type
}

// SeasonalStrength calculates the strength of seasonal component
func (dr *DecompositionResult) SeasonalStrength() float64 {
	if dr == nil || dr.Original == nil || dr.Seasonal == nil || dr.Residual == nil {
		return 0.0
	}

	// Calculate variance of seasonal and residual components
	seasonalVar := dr.Seasonal.variance()
	residualVar := dr.Residual.variance()

	if math.IsNaN(seasonalVar) || math.IsNaN(residualVar) || (seasonalVar+residualVar) == 0 {
		return 0.0
	}

	// Seasonal strength = Var(seasonal) / (Var(seasonal) + Var(residual))
	return seasonalVar / (seasonalVar + residualVar)
}

// TrendStrength calculates the strength of trend component
func (dr *DecompositionResult) TrendStrength() float64 {
	if dr == nil || dr.Original == nil || dr.Trend == nil || dr.Residual == nil {
		return 0.0
	}

	// Calculate variance of trend and residual components
	trendVar := dr.Trend.variance()
	residualVar := dr.Residual.variance()

	if math.IsNaN(trendVar) || math.IsNaN(residualVar) || (trendVar+residualVar) == 0 {
		return 0.0
	}

	// Trend strength = Var(trend) / (Var(trend) + Var(residual))
	return trendVar / (trendVar + residualVar)
}

// ReconstructionRMSE calculates reconstruction root mean square error
func (dr *DecompositionResult) ReconstructionRMSE() float64 {
	if dr == nil || dr.Original == nil || dr.Trend == nil || dr.Seasonal == nil {
		return 0.0
	}

	// Reconstruct the series
	var reconstructed *TimeSeries
	var err error

	if dr.Type == "additive" {
		// Additive: original = trend + seasonal + residual
		trend_plus_seasonal, err := dr.Trend.addSeries(dr.Seasonal)
		if err != nil {
			return 0.0
		}
		reconstructed = trend_plus_seasonal
	} else {
		// Multiplicative: original = trend * seasonal * residual
		reconstructed, err = dr.Trend.multiplySeries(dr.Seasonal)
		if err != nil {
			return 0.0
		}
	}

	// Calculate RMSE between original and reconstructed
	sumSquaredErrors := 0.0
	count := 0

	for i := 0; i < dr.Original.Len(); i++ {
		orig := dr.Original.At(i)
		recon := reconstructed.At(i)

		if !math.IsNaN(orig) && !math.IsNaN(recon) {
			error := orig - recon
			sumSquaredErrors += error * error
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return math.Sqrt(sumSquaredErrors / float64(count))
}

// Summary provides a text summary of the decomposition
func (dr *DecompositionResult) Summary() string {
	if dr == nil {
		return "Invalid decomposition result"
	}

	seasonalStrength := dr.SeasonalStrength()
	trendStrength := dr.TrendStrength()
	rmse := dr.ReconstructionRMSE()

	return fmt.Sprintf("Time Series Decomposition (%s)\nPeriod: %d\nSeasonal Strength: %.3f\nTrend Strength: %.3f\nReconstruction RMSE: %.6f",
		dr.Type, dr.Period, seasonalStrength, trendStrength, rmse)
}

// Helper functions for decomposition

// extractCenteredMovingAverage computes centered moving average for trend extraction
func (ts *TimeSeries) extractCenteredMovingAverage(period int) (*TimeSeries, error) {
	if period%2 == 0 {
		// Even period: two-step moving average
		return ts.extractEvenCenteredMA(period)
	} else {
		// Odd period: simple centered moving average
		return ts.extractOddCenteredMA(period)
	}
}

// extractOddCenteredMA computes centered moving average for odd periods
func (ts *TimeSeries) extractOddCenteredMA(period int) (*TimeSeries, error) {
	values := make([]float64, ts.Len())
	halfPeriod := period / 2

	// Fill with NaN at the beginning and end
	for i := 0; i < ts.Len(); i++ {
		if i < halfPeriod || i >= ts.Len()-halfPeriod {
			values[i] = math.NaN()
		} else {
			// Calculate centered moving average
			sum := 0.0
			count := 0
			for j := i - halfPeriod; j <= i+halfPeriod; j++ {
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

	return NewTimeSeries(values, ts.index, ts.name+"_trend")
}

// extractEvenCenteredMA computes centered moving average for even periods (2x2 moving average)
func (ts *TimeSeries) extractEvenCenteredMA(period int) (*TimeSeries, error) {
	// First compute simple moving average
	simpleMA := make([]float64, ts.Len())
	halfPeriod := period / 2

	for i := 0; i < ts.Len(); i++ {
		if i < period-1 {
			simpleMA[i] = math.NaN()
		} else {
			sum := 0.0
			count := 0
			for j := i - period + 1; j <= i; j++ {
				val := ts.At(j)
				if !math.IsNaN(val) {
					sum += val
					count++
				}
			}
			if count == 0 {
				simpleMA[i] = math.NaN()
			} else {
				simpleMA[i] = sum / float64(count)
			}
		}
	}

	// Then compute centered 2-point moving average
	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		if i == 0 || i >= ts.Len()-halfPeriod {
			values[i] = math.NaN()
		} else {
			val1 := simpleMA[i-1]
			val2 := simpleMA[i]
			if math.IsNaN(val1) || math.IsNaN(val2) {
				values[i] = math.NaN()
			} else {
				values[i] = (val1 + val2) / 2.0
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_trend")
}

// extractSeasonalPattern extracts seasonal pattern from detrended data
func (ts *TimeSeries) extractSeasonalPattern(detrended *TimeSeries, period int, decompositionType string) (*TimeSeries, error) {
	// Calculate seasonal averages for each position in the cycle
	seasonalAverages := make([]float64, period)
	seasonalCounts := make([]int, period)

	// Initialize
	for i := 0; i < period; i++ {
		seasonalAverages[i] = 0.0
		seasonalCounts[i] = 0
	}

	// Accumulate values for each seasonal position
	for i := 0; i < detrended.Len(); i++ {
		val := detrended.At(i)
		if !math.IsNaN(val) {
			seasonalPos := i % period
			seasonalAverages[seasonalPos] += val
			seasonalCounts[seasonalPos]++
		}
	}

	// Calculate averages
	for i := 0; i < period; i++ {
		if seasonalCounts[i] > 0 {
			seasonalAverages[i] /= float64(seasonalCounts[i])
		} else {
			if decompositionType == "multiplicative" {
				seasonalAverages[i] = 1.0
			} else {
				seasonalAverages[i] = 0.0
			}
		}
	}

	// Normalize seasonal indices
	if decompositionType == "multiplicative" {
		// For multiplicative, seasonal indices should average to 1.0
		sum := 0.0
		count := 0
		for i := 0; i < period; i++ {
			if !math.IsNaN(seasonalAverages[i]) {
				sum += seasonalAverages[i]
				count++
			}
		}
		if count > 0 {
			avg := sum / float64(count)
			if avg != 0.0 {
				for i := 0; i < period; i++ {
					seasonalAverages[i] /= avg
				}
			}
		}
	} else {
		// For additive, seasonal indices should sum to 0
		sum := 0.0
		count := 0
		for i := 0; i < period; i++ {
			if !math.IsNaN(seasonalAverages[i]) {
				sum += seasonalAverages[i]
				count++
			}
		}
		if count > 0 {
			avg := sum / float64(count)
			for i := 0; i < period; i++ {
				seasonalAverages[i] -= avg
			}
		}
	}

	// Create full seasonal series by repeating the pattern
	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		seasonalPos := i % period
		values[i] = seasonalAverages[seasonalPos]
	}

	return NewTimeSeries(values, ts.index, ts.name+"_seasonal")
}

// computeAutocorrelation computes the lag-k autocorrelation coefficient
func (ts *TimeSeries) computeAutocorrelation(lag int) float64 {
	if lag >= ts.Len() {
		return 0.0
	}

	// Calculate mean
	mean := ts.Mean()
	if math.IsNaN(mean) {
		return 0.0
	}

	// Calculate autocorrelation
	numerator := 0.0
	denominator := 0.0
	count := 0

	for i := 0; i < ts.Len()-lag; i++ {
		val1 := ts.At(i)
		val2 := ts.At(i + lag)

		if !math.IsNaN(val1) && !math.IsNaN(val2) {
			dev1 := val1 - mean
			dev2 := val2 - mean
			numerator += dev1 * dev2
			denominator += dev1 * dev1
			count++
		}
	}

	if count == 0 || denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

// extractLinearTrend fits a linear trend to the time series
func (ts *TimeSeries) extractLinearTrend() (*TimeSeries, error) {
	n := ts.Len()

	// Create time indices (0, 1, 2, ...)
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0
	count := 0

	for i := 0; i < n; i++ {
		val := ts.At(i)
		if !math.IsNaN(val) {
			x := float64(i)
			sumX += x
			sumY += val
			sumXY += x * val
			sumXX += x * x
			count++
		}
	}

	if count < 2 {
		return nil, errors.New("need at least 2 valid observations for linear trend")
	}

	// Calculate linear regression coefficients
	n_float := float64(count)
	slope := (n_float*sumXY - sumX*sumY) / (n_float*sumXX - sumX*sumX)
	intercept := (sumY - slope*sumX) / n_float

	// Generate trend values
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = intercept + slope*float64(i)
	}

	return NewTimeSeries(values, ts.index, ts.name+"_linear_trend")
}
