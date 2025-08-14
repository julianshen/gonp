package timeseries

import (
	"errors"
	"math"
	"time"
)

// ARIMAModel represents an ARIMA model
type ARIMAModel struct {
	p, d, q    int              // ARIMA orders
	fitted     bool             // Whether model is fitted
	parameters *ARIMAParameters // Fitted parameters
	residuals  *TimeSeries      // Model residuals
	fitted_ts  *TimeSeries      // Original time series used for fitting
}

// ARIMAParameters holds the fitted ARIMA parameters
type ARIMAParameters struct {
	AR       []float64 // Autoregressive coefficients
	MA       []float64 // Moving average coefficients
	Constant float64   // Model constant/intercept
	Sigma2   float64   // Innovation variance
}

// ForecastResult represents forecasting output
type ForecastResult struct {
	Values     []float64 // Point forecasts
	Timestamps []int     // Time indices for forecasts
	Model      string    // Model specification
}

// PredictionInterval represents forecast prediction intervals
type PredictionInterval struct {
	Lower      []float64 // Lower bounds
	Upper      []float64 // Upper bounds
	Confidence float64   // Confidence level
}

// DiagnosticResult represents model diagnostic tests
type DiagnosticResult struct {
	LjungBox   *StatisticalTestResult // Ljung-Box test on residuals
	JarqueBera *StatisticalTestResult // Jarque-Bera normality test
}

// NewARIMA creates a new ARIMA model with specified orders
func NewARIMA(p, d, q int) (*ARIMAModel, error) {
	if p < 0 {
		return nil, errors.New("autoregressive order (p) must be non-negative")
	}
	if d < 0 {
		return nil, errors.New("differencing order (d) must be non-negative")
	}
	if q < 0 {
		return nil, errors.New("moving average order (q) must be non-negative")
	}

	return &ARIMAModel{
		p:      p,
		d:      d,
		q:      q,
		fitted: false,
	}, nil
}

// P returns the autoregressive order
func (m *ARIMAModel) P() int {
	return m.p
}

// D returns the differencing order
func (m *ARIMAModel) D() int {
	return m.d
}

// Q returns the moving average order
func (m *ARIMAModel) Q() int {
	return m.q
}

// ModelType returns the model type string
func (m *ARIMAModel) ModelType() string {
	return "ARIMA"
}

// IsFitted returns whether the model has been fitted
func (m *ARIMAModel) IsFitted() bool {
	return m.fitted
}

// GetParameters returns the fitted parameters
func (m *ARIMAModel) GetParameters() *ARIMAParameters {
	return m.parameters
}

// Fit fits the ARIMA model to the given time series
func (m *ARIMAModel) Fit(ts *TimeSeries) error {
	if ts == nil {
		return errors.New("time series cannot be nil")
	}

	// Check minimum data requirements
	minObs := m.p + m.d + m.q + 2
	if ts.Len() < minObs {
		return errors.New("insufficient observations for ARIMA fitting")
	}

	// Store original time series
	m.fitted_ts = ts

	// Apply differencing if needed
	workingSeries := ts
	var err error

	for i := 0; i < m.d; i++ {
		workingSeries, err = workingSeries.Difference(1)
		if err != nil {
			return err
		}
	}

	// Estimate parameters using simplified method
	// In production, would use Maximum Likelihood Estimation
	err = m.estimateParameters(workingSeries)
	if err != nil {
		return err
	}

	// Calculate residuals
	m.calculateResiduals(workingSeries)

	m.fitted = true
	return nil
}

// estimateParameters estimates ARIMA parameters using simplified methods
func (m *ARIMAModel) estimateParameters(ts *TimeSeries) error {
	m.parameters = &ARIMAParameters{
		AR:     make([]float64, m.p),
		MA:     make([]float64, m.q),
		Sigma2: 1.0,
	}

	// Simplified parameter estimation
	if m.p > 0 {
		// Use Yule-Walker equations for AR parameters
		acf, err := ts.AutocorrelationFunction(m.p)
		if err != nil {
			return err
		}

		// Simple AR estimation: φ₁ = ρ₁ for AR(1)
		if m.p == 1 && len(acf) > 1 {
			m.parameters.AR[0] = acf[1] // First lag autocorrelation
		} else {
			// For higher orders, use simplified estimation
			for i := 0; i < m.p && i+1 < len(acf); i++ {
				m.parameters.AR[i] = acf[i+1] * 0.8 // Dampened coefficient
			}
		}
	}

	if m.q > 0 {
		// Simplified MA parameter estimation
		for i := 0; i < m.q; i++ {
			m.parameters.MA[i] = 0.3 * math.Pow(0.7, float64(i)) // Decaying MA coefficients
		}
	}

	// Calculate constant (mean of differenced series)
	m.parameters.Constant = ts.Mean()

	// Estimate innovation variance
	variance := ts.variance()
	if !math.IsNaN(variance) && variance > 0 {
		m.parameters.Sigma2 = variance
	}

	return nil
}

// calculateResiduals computes model residuals
func (m *ARIMAModel) calculateResiduals(ts *TimeSeries) {
	// Simplified residual calculation
	// In production, would use proper ARIMA recursions

	n := ts.Len()
	residualValues := make([]float64, n)

	for i := 0; i < n; i++ {
		actual := ts.At(i)

		// Simple prediction using AR terms
		predicted := m.parameters.Constant

		for j := 0; j < m.p && i-j-1 >= 0; j++ {
			if j < len(m.parameters.AR) {
				predicted += m.parameters.AR[j] * ts.At(i-j-1)
			}
		}

		residualValues[i] = actual - predicted
	}

	m.residuals, _ = NewTimeSeries(residualValues, ts.index, "residuals")
}

// Residuals returns the model residuals
func (m *ARIMAModel) Residuals() *TimeSeries {
	return m.residuals
}

// LogLikelihood returns the log-likelihood of the fitted model
func (m *ARIMAModel) LogLikelihood() float64 {
	if !m.fitted || m.residuals == nil {
		return math.NaN()
	}

	// Simplified log-likelihood calculation
	n := float64(m.residuals.Len())
	sigma2 := m.parameters.Sigma2

	if sigma2 <= 0 {
		return math.NaN()
	}

	// Log-likelihood for normal innovations
	logL := -0.5*n*math.Log(2*math.Pi) - 0.5*n*math.Log(sigma2)

	// Add residual sum of squares term
	ssq := 0.0
	for i := 0; i < m.residuals.Len(); i++ {
		res := m.residuals.At(i)
		if !math.IsNaN(res) {
			ssq += res * res
		}
	}

	logL -= ssq / (2 * sigma2)

	return logL
}

// AIC returns the Akaike Information Criterion
func (m *ARIMAModel) AIC() float64 {
	if !m.fitted {
		return math.NaN()
	}

	logL := m.LogLikelihood()
	k := float64(m.p + m.q + 1) // Number of parameters (including constant)

	return -2*logL + 2*k
}

// BIC returns the Bayesian Information Criterion
func (m *ARIMAModel) BIC() float64 {
	if !m.fitted {
		return math.NaN()
	}

	logL := m.LogLikelihood()
	k := float64(m.p + m.q + 1)
	n := float64(m.fitted_ts.Len())

	return -2*logL + k*math.Log(n)
}

// Forecast generates point forecasts
func (m *ARIMAModel) Forecast(horizon int) (*ForecastResult, error) {
	if !m.fitted {
		return nil, errors.New("model must be fitted before forecasting")
	}
	if horizon <= 0 {
		return nil, errors.New("forecast horizon must be positive")
	}

	// Get the most recent observations for AR components
	originalSeries := m.fitted_ts
	recentObs := make([]float64, m.p)

	// For AR terms, use differenced series if d > 0
	workingSeries := originalSeries
	var err error
	for i := 0; i < m.d; i++ {
		workingSeries, err = workingSeries.Difference(1)
		if err != nil {
			return nil, err
		}
	}

	// Extract recent observations for AR terms
	for i := 0; i < m.p && i < workingSeries.Len(); i++ {
		idx := workingSeries.Len() - 1 - i
		recentObs[m.p-1-i] = workingSeries.At(idx)
	}

	// Generate forecasts
	forecasts := make([]float64, horizon)

	for h := 0; h < horizon; h++ {
		forecast := m.parameters.Constant

		// Add AR components
		for i := 0; i < m.p; i++ {
			if i < len(m.parameters.AR) {
				if h > i {
					// Use previously forecasted values
					forecast += m.parameters.AR[i] * forecasts[h-1-i]
				} else {
					// Use historical observations
					forecast += m.parameters.AR[i] * recentObs[m.p-1-i]
				}
			}
		}

		forecasts[h] = forecast

		// Update recent observations for next forecast
		if h < m.p-1 {
			// Shift and add new forecast
			for i := 0; i < m.p-1; i++ {
				recentObs[i] = recentObs[i+1]
			}
			recentObs[m.p-1] = forecast
		}
	}

	// If model has differencing, need to "undifference" the forecasts
	if m.d > 0 {
		forecasts = m.undifferenceForecast(forecasts, originalSeries)
	}

	timestamps := make([]int, horizon)
	for i := range timestamps {
		timestamps[i] = originalSeries.Len() + i
	}

	return &ForecastResult{
		Values:     forecasts,
		Timestamps: timestamps,
		Model:      "ARIMA",
	}, nil
}

// PredictionIntervals generates prediction intervals
func (m *ARIMAModel) PredictionIntervals(horizon int, confidence float64) (*PredictionInterval, error) {
	if !m.fitted {
		return nil, errors.New("model must be fitted before generating prediction intervals")
	}
	if horizon <= 0 {
		return nil, errors.New("forecast horizon must be positive")
	}
	if confidence <= 0 || confidence >= 1 {
		return nil, errors.New("confidence level must be between 0 and 1")
	}

	// First get point forecasts
	forecast, err := m.Forecast(horizon)
	if err != nil {
		return nil, err
	}

	// Get z-value for confidence level
	var zValue float64
	if confidence == 0.95 {
		zValue = 1.96
	} else if confidence == 0.99 {
		zValue = 2.576
	} else if confidence == 0.90 {
		zValue = 1.645
	} else {
		zValue = 1.96 // Default to 95%
	}

	// Calculate prediction intervals around point forecasts
	lower := make([]float64, horizon)
	upper := make([]float64, horizon)

	for h := 0; h < horizon; h++ {
		// For simplicity, assume constant forecast variance
		// In practice, variance increases with forecast horizon
		variance := m.parameters.Sigma2 * float64(h+1) // Linear increase
		stdError := math.Sqrt(variance)

		margin := zValue * stdError
		pointForecast := forecast.Values[h]

		lower[h] = pointForecast - margin
		upper[h] = pointForecast + margin
	}

	return &PredictionInterval{
		Lower:      lower,
		Upper:      upper,
		Confidence: confidence,
	}, nil
}

// undifferenceForecast converts differenced forecasts back to original scale
func (m *ARIMAModel) undifferenceForecast(forecasts []float64, originalSeries *TimeSeries) []float64 {
	if m.d == 0 || len(forecasts) == 0 {
		return forecasts
	}

	undifferenced := make([]float64, len(forecasts))

	// Get last observation from original series
	lastValue := originalSeries.At(originalSeries.Len() - 1)

	// Simple undifferencing (for d=1)
	// For higher orders of differencing, would need more complex logic
	undifferenced[0] = lastValue + forecasts[0]

	for i := 1; i < len(forecasts); i++ {
		undifferenced[i] = undifferenced[i-1] + forecasts[i]
	}

	return undifferenced
}

// DiagnosticTests runs diagnostic tests on model residuals
func (m *ARIMAModel) DiagnosticTests() (*DiagnosticResult, error) {
	if !m.fitted {
		return nil, errors.New("model must be fitted before running diagnostics")
	}
	if m.residuals == nil {
		return nil, errors.New("residuals not available for diagnostic testing")
	}

	// Run Ljung-Box test for residual autocorrelation
	ljungBoxLags := 5
	if m.residuals.Len() < ljungBoxLags*2 {
		ljungBoxLags = m.residuals.Len() / 3 // Use fewer lags for short series
	}
	if ljungBoxLags < 1 {
		ljungBoxLags = 1
	}

	ljungBox, err := m.residuals.LjungBoxTest(ljungBoxLags)
	if err != nil {
		return nil, err
	}

	// Run Jarque-Bera test for normality of residuals
	jarqueBera, err := m.jarqueBeraTest()
	if err != nil {
		return nil, err
	}

	return &DiagnosticResult{
		LjungBox:   ljungBox,
		JarqueBera: jarqueBera,
	}, nil
}

// jarqueBeraTest performs the Jarque-Bera test for normality
func (m *ARIMAModel) jarqueBeraTest() (*StatisticalTestResult, error) {
	if m.residuals == nil {
		return nil, errors.New("residuals not available for Jarque-Bera test")
	}

	n := float64(m.residuals.Len())
	if n < 7 {
		return nil, errors.New("need at least 7 observations for Jarque-Bera test")
	}

	// Calculate sample moments
	mean := m.residuals.Mean()
	variance := m.residuals.variance()
	if variance <= 0 {
		return nil, errors.New("residual variance must be positive for Jarque-Bera test")
	}

	// Calculate skewness and kurtosis
	skewness := 0.0
	kurtosis := 0.0
	validCount := 0

	for i := 0; i < m.residuals.Len(); i++ {
		val := m.residuals.At(i)
		if !math.IsNaN(val) {
			standardized := (val - mean) / math.Sqrt(variance)
			skewness += math.Pow(standardized, 3)
			kurtosis += math.Pow(standardized, 4)
			validCount++
		}
	}

	if validCount < 7 {
		return nil, errors.New("insufficient valid residuals for Jarque-Bera test")
	}

	skewness = skewness / float64(validCount)
	kurtosis = (kurtosis / float64(validCount)) - 3.0 // Excess kurtosis

	// Calculate Jarque-Bera statistic: JB = n/6 * (S² + K²/4)
	jbStatistic := (float64(validCount) / 6.0) * (skewness*skewness + (kurtosis*kurtosis)/4.0)

	// Calculate p-value using chi-square distribution with 2 df
	pValue := m.residuals.chiSquarePValue(jbStatistic, 2)

	// Critical value for 5% significance level with 2 df
	criticalValue := 5.99

	interpretation := "Fail to reject null hypothesis (residuals appear normal)"
	if jbStatistic > criticalValue {
		interpretation = "Reject null hypothesis (residuals not normal)"
	}

	return &StatisticalTestResult{
		Statistic:        jbStatistic,
		PValue:           pValue,
		DegreesOfFreedom: 2,
		CriticalValue:    criticalValue,
		Interpretation:   interpretation,
	}, nil
}

// Difference applies first differencing to a time series
func (ts *TimeSeries) Difference(order int) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if order <= 0 {
		return nil, errors.New("differencing order must be positive")
	}
	if order >= ts.Len() {
		return nil, errors.New("differencing order cannot exceed series length")
	}

	current := ts
	var err error

	for i := 0; i < order; i++ {
		current, err = current.firstDifference()
		if err != nil {
			return nil, err
		}
	}

	return current, nil
}

// firstDifference applies single first differencing
func (ts *TimeSeries) firstDifference() (*TimeSeries, error) {
	if ts.Len() < 2 {
		return nil, errors.New("need at least 2 observations for differencing")
	}

	n := ts.Len()
	values := make([]float64, n-1)
	dates := make([]time.Time, n-1)

	for i := 1; i < n; i++ {
		curr := ts.At(i)
		prev := ts.At(i - 1)

		if math.IsNaN(curr) || math.IsNaN(prev) {
			values[i-1] = math.NaN()
		} else {
			values[i-1] = curr - prev
		}

		dates[i-1] = ts.TimeAt(i)
	}

	return NewTimeSeries(values, dates, ts.name+"_diff")
}

// SeasonalDifference applies seasonal differencing
func (ts *TimeSeries) SeasonalDifference(period int) (*TimeSeries, error) {
	if ts == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if period <= 0 {
		return nil, errors.New("seasonal period must be positive")
	}
	if period >= ts.Len() {
		return nil, errors.New("seasonal period cannot exceed series length")
	}

	n := ts.Len()
	values := make([]float64, n-period)
	dates := make([]time.Time, n-period)

	for i := period; i < n; i++ {
		curr := ts.At(i)
		seasonal := ts.At(i - period)

		if math.IsNaN(curr) || math.IsNaN(seasonal) {
			values[i-period] = math.NaN()
		} else {
			values[i-period] = curr - seasonal
		}

		dates[i-period] = ts.TimeAt(i)
	}

	return NewTimeSeries(values, dates, ts.name+"_seasonal_diff")
}
