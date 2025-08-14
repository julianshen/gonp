package timeseries

import (
	"errors"
	"math"
)

// ACFResult represents the result of autocorrelation function computation
type ACFResult struct {
	ACF     []float64 // Autocorrelation coefficients
	LowerCI []float64 // Lower confidence interval
	UpperCI []float64 // Upper confidence interval
	Lags    []int     // Lag values
}

// PACFResult represents the result of partial autocorrelation function computation
type PACFResult struct {
	PACF           []float64   // Partial autocorrelation coefficients
	ARCoefficients [][]float64 // AR coefficients at each lag
	Lags           []int       // Lag values
}

// CrossCorrelationResult represents cross-correlation analysis result
type CrossCorrelationResult struct {
	Correlations []float64 // Cross-correlation coefficients
	Lags         []int     // Lag values (negative = x lags behind y, positive = x leads y)
}

// LeadLagResult represents lead-lag analysis result
type LeadLagResult struct {
	OptimalLag     int     // Optimal lag for maximum correlation
	MaxCorrelation float64 // Maximum cross-correlation value
	IsLeading      bool    // True if first series leads second
}

// StatisticalTestResult represents result of statistical tests
type StatisticalTestResult struct {
	Statistic        float64 // Test statistic
	PValue           float64 // P-value
	DegreesOfFreedom int     // Degrees of freedom
	CriticalValue    float64 // Critical value at 5% significance
	Interpretation   string  // Text interpretation
}

// AutocorrelationFunction computes the autocorrelation function
func (ts *TimeSeries) AutocorrelationFunction(maxLags int) ([]float64, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if maxLags < 0 {
		return nil, errors.New("max lags must be non-negative")
	}
	if maxLags >= ts.Len() {
		return nil, errors.New("max lags must be less than series length")
	}

	n := ts.Len()
	acf := make([]float64, maxLags+1)

	// Calculate mean
	mean := ts.Mean()
	if math.IsNaN(mean) {
		return nil, errors.New("cannot compute ACF with invalid mean")
	}

	// Calculate variance (denominator for lag 0)
	var0 := 0.0
	validCount := 0

	for i := 0; i < n; i++ {
		val := ts.At(i)
		if !math.IsNaN(val) {
			dev := val - mean
			var0 += dev * dev
			validCount++
		}
	}

	if validCount < 2 || var0 == 0 {
		return nil, errors.New("insufficient valid data for ACF computation")
	}

	// ACF at lag 0 is always 1.0
	acf[0] = 1.0

	// Calculate ACF for each lag
	for lag := 1; lag <= maxLags; lag++ {
		covariance := 0.0
		count := 0

		for i := 0; i < n-lag; i++ {
			val1 := ts.At(i)
			val2 := ts.At(i + lag)

			if !math.IsNaN(val1) && !math.IsNaN(val2) {
				dev1 := val1 - mean
				dev2 := val2 - mean
				covariance += dev1 * dev2
				count++
			}
		}

		if count == 0 {
			acf[lag] = 0.0
		} else {
			acf[lag] = covariance / var0
		}
	}

	return acf, nil
}

// AutocorrelationFunctionWithCI computes ACF with confidence intervals
func (ts *TimeSeries) AutocorrelationFunctionWithCI(maxLags int, confidence float64) (*ACFResult, error) {
	if confidence <= 0 || confidence >= 1 {
		return nil, errors.New("confidence level must be between 0 and 1")
	}

	// Compute basic ACF
	acf, err := ts.AutocorrelationFunction(maxLags)
	if err != nil {
		return nil, err
	}

	// Calculate confidence intervals
	n := float64(ts.Len())

	// For large n, confidence intervals are approximately ±z_{α/2}/√n
	// For 95% confidence, z_{0.025} ≈ 1.96
	var zValue float64
	if confidence == 0.95 {
		zValue = 1.96
	} else if confidence == 0.99 {
		zValue = 2.576
	} else {
		// Approximate for other confidence levels
		zValue = 1.96 // Default to 95%
	}

	ciWidth := zValue / math.Sqrt(n)

	result := &ACFResult{
		ACF:     acf,
		LowerCI: make([]float64, len(acf)),
		UpperCI: make([]float64, len(acf)),
		Lags:    make([]int, len(acf)),
	}

	for i := range acf {
		result.Lags[i] = i
		if i == 0 {
			// ACF at lag 0 is always 1, no confidence interval needed
			result.LowerCI[i] = 1.0
			result.UpperCI[i] = 1.0
		} else {
			result.LowerCI[i] = -ciWidth
			result.UpperCI[i] = ciWidth
		}
	}

	return result, nil
}

// PartialAutocorrelationFunction computes the partial autocorrelation function
func (ts *TimeSeries) PartialAutocorrelationFunction(maxLags int) ([]float64, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if maxLags < 0 {
		return nil, errors.New("max lags must be non-negative")
	}
	if maxLags >= ts.Len()-1 {
		return nil, errors.New("max lags must be less than series length - 1")
	}

	// First compute ACF up to maxLags
	acf, err := ts.AutocorrelationFunction(maxLags)
	if err != nil {
		return nil, err
	}

	pacf := make([]float64, maxLags+1)
	pacf[0] = 1.0 // PACF at lag 0 is always 1

	if maxLags == 0 {
		return pacf, nil
	}

	// PACF at lag 1 equals ACF at lag 1
	pacf[1] = acf[1]

	// For higher lags, use Durbin-Levinson recursion
	phi := make([][]float64, maxLags+1)
	for i := range phi {
		phi[i] = make([]float64, i+1)
	}

	phi[1][1] = acf[1]

	for k := 2; k <= maxLags; k++ {
		// Calculate phi[k][k] (the PACF value)
		numerator := acf[k]
		denominator := 1.0

		for j := 1; j < k; j++ {
			numerator -= phi[k-1][j] * acf[k-j]
		}

		for j := 1; j < k; j++ {
			denominator -= phi[k-1][j] * acf[j]
		}

		if denominator == 0 {
			pacf[k] = 0.0
		} else {
			phi[k][k] = numerator / denominator
			pacf[k] = phi[k][k]
		}

		// Update phi coefficients
		for j := 1; j < k; j++ {
			phi[k][j] = phi[k-1][j] - phi[k][k]*phi[k-1][k-j]
		}
	}

	return pacf, nil
}

// PartialAutocorrelationFunctionYW computes PACF using Yule-Walker equations
func (ts *TimeSeries) PartialAutocorrelationFunctionYW(maxLags int) (*PACFResult, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if maxLags < 0 {
		return nil, errors.New("max lags must be non-negative")
	}
	if maxLags >= ts.Len()-1 {
		return nil, errors.New("max lags must be less than series length - 1")
	}

	// Compute PACF using the simpler method first
	pacf, err := ts.PartialAutocorrelationFunction(maxLags)
	if err != nil {
		return nil, err
	}

	// Store AR coefficients for each lag
	arCoeffs := make([][]float64, maxLags+1)
	for i := range arCoeffs {
		arCoeffs[i] = make([]float64, i+1)
		if i > 0 {
			arCoeffs[i][i-1] = pacf[i] // Store PACF as the last AR coefficient
		}
	}

	lags := make([]int, maxLags+1)
	for i := range lags {
		lags[i] = i
	}

	return &PACFResult{
		PACF:           pacf,
		ARCoefficients: arCoeffs,
		Lags:           lags,
	}, nil
}

// CrossCorrelation computes cross-correlation between two time series
func (ts *TimeSeries) CrossCorrelation(other *TimeSeries, maxLags int) (*CrossCorrelationResult, error) {
	if ts == nil || other == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if ts.Len() != other.Len() {
		return nil, errors.New("time series must have the same length")
	}
	if maxLags < 0 {
		return nil, errors.New("max lags must be non-negative")
	}
	if maxLags >= ts.Len() {
		return nil, errors.New("max lags must be less than series length")
	}

	n := ts.Len()
	correlations := make([]float64, 2*maxLags+1)
	lags := make([]int, 2*maxLags+1)

	// Calculate means
	mean1 := ts.Mean()
	mean2 := other.Mean()

	if math.IsNaN(mean1) || math.IsNaN(mean2) {
		return nil, errors.New("cannot compute cross-correlation with invalid means")
	}

	// Calculate standard deviations
	std1 := ts.Std()
	std2 := other.Std()

	if math.IsNaN(std1) || math.IsNaN(std2) || std1 == 0 || std2 == 0 {
		return nil, errors.New("cannot compute cross-correlation with zero or invalid standard deviation")
	}

	// Compute cross-correlation for each lag
	idx := 0
	for lag := -maxLags; lag <= maxLags; lag++ {
		lags[idx] = lag

		sum := 0.0
		count := 0

		for i := 0; i < n; i++ {
			j := i + lag
			if j >= 0 && j < n {
				val1 := ts.At(i)
				val2 := other.At(j)

				if !math.IsNaN(val1) && !math.IsNaN(val2) {
					sum += ((val1 - mean1) / std1) * ((val2 - mean2) / std2)
					count++
				}
			}
		}

		if count > 0 {
			correlations[idx] = sum / float64(count)
		} else {
			correlations[idx] = 0.0
		}

		idx++
	}

	return &CrossCorrelationResult{
		Correlations: correlations,
		Lags:         lags,
	}, nil
}

// LeadLagAnalysis performs lead-lag analysis between two series
func (ts *TimeSeries) LeadLagAnalysis(other *TimeSeries, maxLags int) (*LeadLagResult, error) {
	ccf, err := ts.CrossCorrelation(other, maxLags)
	if err != nil {
		return nil, err
	}

	// Find maximum absolute correlation and its lag
	maxAbsCorr := 0.0
	optimalLag := 0
	maxCorr := 0.0

	for i, corr := range ccf.Correlations {
		absCorr := math.Abs(corr)
		if absCorr > maxAbsCorr {
			maxAbsCorr = absCorr
			optimalLag = ccf.Lags[i]
			maxCorr = corr
		}
	}

	return &LeadLagResult{
		OptimalLag:     optimalLag,
		MaxCorrelation: maxCorr,
		IsLeading:      optimalLag > 0, // Positive lag means first series leads
	}, nil
}

// LjungBoxTest performs the Ljung-Box test for autocorrelation
func (ts *TimeSeries) LjungBoxTest(lags int) (*StatisticalTestResult, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if lags <= 0 {
		return nil, errors.New("lags must be positive")
	}
	if lags >= ts.Len() {
		return nil, errors.New("lags must be less than series length")
	}

	// Compute autocorrelations
	acf, err := ts.AutocorrelationFunction(lags)
	if err != nil {
		return nil, err
	}

	n := float64(ts.Len())

	// Calculate Ljung-Box statistic: Q = n(n+2) * Σ(ρ²_k/(n-k))
	qStat := 0.0
	for k := 1; k <= lags; k++ {
		rho_k := acf[k]
		qStat += (rho_k * rho_k) / (n - float64(k))
	}
	qStat *= n * (n + 2)

	// Calculate p-value using chi-square distribution approximation
	// For large n, Q follows χ²(lags) distribution under null hypothesis
	pValue := ts.chiSquarePValue(qStat, lags)

	// Critical value for 5% significance level
	criticalValue := ts.chiSquareCritical(lags, 0.05)

	interpretation := "Fail to reject null hypothesis (no autocorrelation)"
	if qStat > criticalValue {
		interpretation = "Reject null hypothesis (autocorrelation detected)"
	}

	return &StatisticalTestResult{
		Statistic:        qStat,
		PValue:           pValue,
		DegreesOfFreedom: lags,
		CriticalValue:    criticalValue,
		Interpretation:   interpretation,
	}, nil
}

// DurbinWatsonTest performs the Durbin-Watson test for first-order autocorrelation
func (ts *TimeSeries) DurbinWatsonTest() (*StatisticalTestResult, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if ts.Len() < 3 {
		return nil, errors.New("need at least 3 observations for Durbin-Watson test")
	}

	n := ts.Len()

	// Calculate residuals (for time series, use first differences as approximation)
	residuals := make([]float64, 0, n-1)
	for i := 1; i < n; i++ {
		val_curr := ts.At(i)
		val_prev := ts.At(i - 1)

		if !math.IsNaN(val_curr) && !math.IsNaN(val_prev) {
			residuals = append(residuals, val_curr-val_prev)
		}
	}

	if len(residuals) < 2 {
		return nil, errors.New("insufficient valid residuals for Durbin-Watson test")
	}

	// Calculate Durbin-Watson statistic: DW = Σ(e_t - e_{t-1})² / Σe_t²
	sumSquaredDiff := 0.0
	sumSquaredResiduals := 0.0

	for i := 1; i < len(residuals); i++ {
		diff := residuals[i] - residuals[i-1]
		sumSquaredDiff += diff * diff
	}

	for _, res := range residuals {
		sumSquaredResiduals += res * res
	}

	if sumSquaredResiduals == 0 {
		return nil, errors.New("residual sum of squares is zero")
	}

	dwStatistic := sumSquaredDiff / sumSquaredResiduals

	// Interpretation
	var interpretation string
	if dwStatistic < 1.5 {
		interpretation = "Evidence of positive autocorrelation"
	} else if dwStatistic > 2.5 {
		interpretation = "Evidence of negative autocorrelation"
	} else {
		interpretation = "No strong evidence of autocorrelation"
	}

	// Note: Exact p-value calculation for DW test is complex
	// Using simplified interpretation based on statistic value
	var pValue float64
	if dwStatistic < 1.0 || dwStatistic > 3.0 {
		pValue = 0.01 // Strong evidence
	} else if dwStatistic < 1.5 || dwStatistic > 2.5 {
		pValue = 0.05 // Moderate evidence
	} else {
		pValue = 0.20 // Weak evidence
	}

	return &StatisticalTestResult{
		Statistic:        dwStatistic,
		PValue:           pValue,
		DegreesOfFreedom: len(residuals) - 1,
		CriticalValue:    2.0, // Approximate no-autocorrelation value
		Interpretation:   interpretation,
	}, nil
}

// BoxPierceTest performs the Box-Pierce test for independence
func (ts *TimeSeries) BoxPierceTest(lags int) (*StatisticalTestResult, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if lags <= 0 {
		return nil, errors.New("lags must be positive")
	}
	if lags >= ts.Len() {
		return nil, errors.New("lags must be less than series length")
	}

	// Compute autocorrelations
	acf, err := ts.AutocorrelationFunction(lags)
	if err != nil {
		return nil, err
	}

	n := float64(ts.Len())

	// Calculate Box-Pierce statistic: Q = n * Σ(ρ²_k)
	qStat := 0.0
	for k := 1; k <= lags; k++ {
		rho_k := acf[k]
		qStat += rho_k * rho_k
	}
	qStat *= n

	// Calculate p-value using chi-square distribution
	pValue := ts.chiSquarePValue(qStat, lags)

	// Critical value for 5% significance level
	criticalValue := ts.chiSquareCritical(lags, 0.05)

	interpretation := "Fail to reject null hypothesis (series appears independent)"
	if qStat > criticalValue {
		interpretation = "Reject null hypothesis (dependence detected)"
	}

	return &StatisticalTestResult{
		Statistic:        qStat,
		PValue:           pValue,
		DegreesOfFreedom: lags,
		CriticalValue:    criticalValue,
		Interpretation:   interpretation,
	}, nil
}

// Helper functions for chi-square distribution
func (ts *TimeSeries) chiSquarePValue(statistic float64, df int) float64 {
	// Simplified chi-square p-value calculation
	// For accurate implementation, would use gamma function

	if df == 1 {
		// For df=1, use normal approximation
		z := math.Sqrt(statistic)
		return 2.0 * (1.0 - ts.normalCDF(z))
	} else if df <= 5 {
		// Rough approximation for small df
		if statistic > float64(df)*3.0 {
			return 0.01
		} else if statistic > float64(df)*2.0 {
			return 0.05
		} else if statistic > float64(df)*1.5 {
			return 0.10
		} else {
			return 0.20
		}
	} else {
		// Normal approximation for large df
		mean := float64(df)
		variance := 2.0 * float64(df)
		z := (statistic - mean) / math.Sqrt(variance)
		return 1.0 - ts.normalCDF(z)
	}
}

func (ts *TimeSeries) chiSquareCritical(df int, alpha float64) float64 {
	// Approximate chi-square critical values for common cases
	if alpha == 0.05 {
		switch df {
		case 1:
			return 3.84
		case 2:
			return 5.99
		case 3:
			return 7.81
		case 4:
			return 9.49
		case 5:
			return 11.07
		default:
			// Rough approximation for larger df
			return float64(df) + 2.0*math.Sqrt(2.0*float64(df))
		}
	} else if alpha == 0.01 {
		switch df {
		case 1:
			return 6.63
		case 2:
			return 9.21
		case 3:
			return 11.34
		case 4:
			return 13.28
		case 5:
			return 15.09
		default:
			return float64(df) + 3.0*math.Sqrt(2.0*float64(df))
		}
	}

	// Default approximation
	return float64(df) + 2.0*math.Sqrt(2.0*float64(df))
}

func (ts *TimeSeries) normalCDF(x float64) float64 {
	// Standard normal CDF approximation
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt(2.0)))
}
