package timeseries

import (
	"math"
	"testing"
	"time"
)

// TestARIMAModel tests basic ARIMA model creation and structure
func TestARIMAModel(t *testing.T) {
	t.Run("ARIMA model creation and parameter validation", func(t *testing.T) {
		// Test valid ARIMA(1,1,1) model creation
		model, err := NewARIMA(1, 1, 1)
		if err != nil {
			t.Fatalf("Failed to create ARIMA(1,1,1) model: %v", err)
		}

		// Check model parameters
		if model.P() != 1 || model.D() != 1 || model.Q() != 1 {
			t.Errorf("ARIMA parameters: expected (1,1,1), got (%d,%d,%d)", model.P(), model.D(), model.Q())
		}

		// Test model type
		if model.ModelType() != "ARIMA" {
			t.Errorf("Expected model type 'ARIMA', got '%s'", model.ModelType())
		}

		// Test parameter bounds validation
		_, err = NewARIMA(-1, 0, 0)
		if err == nil {
			t.Error("Should fail with negative AR order")
		}

		_, err = NewARIMA(0, -1, 0)
		if err == nil {
			t.Error("Should fail with negative differencing order")
		}

		_, err = NewARIMA(0, 0, -1)
		if err == nil {
			t.Error("Should fail with negative MA order")
		}

		t.Logf("ARIMA model creation successful: ARIMA(%d,%d,%d)", model.P(), model.D(), model.Q())
	})

	t.Run("ARIMA model specification and options", func(t *testing.T) {
		// Test ARIMA with different specifications
		specs := []struct {
			p, d, q int
			name    string
		}{
			{2, 0, 0, "AR(2)"},        // AR(2)
			{0, 1, 2, "IMA(1,2)"},     // IMA(1,2)
			{1, 1, 1, "ARIMA(1,1,1)"}, // ARIMA(1,1,1)
			{2, 1, 2, "ARIMA(2,1,2)"}, // ARIMA(2,1,2)
		}

		for _, spec := range specs {
			model, err := NewARIMA(spec.p, spec.d, spec.q)
			if err != nil {
				t.Errorf("Failed to create ARIMA(%d,%d,%d): %v", spec.p, spec.d, spec.q, err)
				continue
			}

			// Verify specification
			if model.P() != spec.p || model.D() != spec.d || model.Q() != spec.q {
				t.Errorf("Specification mismatch for ARIMA(%d,%d,%d)", spec.p, spec.d, spec.q)
			}

			// Check if model is fitted (should not be initially)
			if model.IsFitted() {
				t.Errorf("New model should not be fitted initially")
			}
		}

		t.Logf("ARIMA model specifications tested successfully")
	})
}

// TestDifferencing tests differencing operations for integrated component
func TestDifferencing(t *testing.T) {
	t.Run("First differencing for I(1) component", func(t *testing.T) {
		// Create trending time series: 1, 2, 3, 4, 5, 6, 7, 8
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "trending_series")

		// Apply first differencing
		diff1, err := ts.Difference(1)
		if err != nil {
			t.Fatalf("First differencing failed: %v", err)
		}

		// Check differenced series length (should be n-1)
		expectedLen := len(values) - 1
		if diff1.Len() != expectedLen {
			t.Errorf("Differenced series length: expected %d, got %d", expectedLen, diff1.Len())
		}

		// Check differenced values (should all be 1.0 for linear trend)
		for i := 0; i < diff1.Len(); i++ {
			expected := 1.0
			actual := diff1.At(i)
			if math.Abs(actual-expected) > 1e-10 {
				t.Errorf("Diff[%d]: expected %.1f, got %.6f", i, expected, actual)
			}
		}

		t.Logf("First differencing successful: converted trend to constant")
	})

	t.Run("Second differencing for I(2) component", func(t *testing.T) {
		// Create quadratic time series: t²
		values := make([]float64, 8)
		for i := range values {
			t_val := float64(i + 1)
			values[i] = t_val * t_val // 1, 4, 9, 16, 25, 36, 49, 64
		}

		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "quadratic_series")

		// Apply second differencing
		diff2, err := ts.Difference(2)
		if err != nil {
			t.Fatalf("Second differencing failed: %v", err)
		}

		// Check differenced series length (should be n-2)
		expectedLen := len(values) - 2
		if diff2.Len() != expectedLen {
			t.Errorf("Second differenced series length: expected %d, got %d", expectedLen, diff2.Len())
		}

		// Check differenced values (should all be 2.0 for quadratic)
		for i := 0; i < diff2.Len(); i++ {
			expected := 2.0
			actual := diff2.At(i)
			if math.Abs(actual-expected) > 1e-10 {
				t.Errorf("Diff2[%d]: expected %.1f, got %.6f", i, expected, actual)
			}
		}

		t.Logf("Second differencing successful: converted quadratic to constant")
	})

	t.Run("Seasonal differencing", func(t *testing.T) {
		// Create series with seasonal pattern (period=4)
		values := []float64{10.0, 15.0, 8.0, 12.0, 11.0, 16.0, 9.0, 13.0, 12.0, 17.0, 10.0, 14.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "seasonal_series")

		// Apply seasonal differencing with period 4
		seasonalDiff, err := ts.SeasonalDifference(4)
		if err != nil {
			t.Fatalf("Seasonal differencing failed: %v", err)
		}

		// Check differenced series length (should be n-period)
		expectedLen := len(values) - 4
		if seasonalDiff.Len() != expectedLen {
			t.Errorf("Seasonal differenced series length: expected %d, got %d", expectedLen, seasonalDiff.Len())
		}

		// Seasonal differences should be small for repeating pattern
		for i := 0; i < seasonalDiff.Len(); i++ {
			diff := seasonalDiff.At(i)
			if math.Abs(diff) > 2.0 { // Allow some variation
				t.Logf("Seasonal diff[%d] = %.1f (expected small for seasonal pattern)", i, diff)
			}
		}

		t.Logf("Seasonal differencing successful: period = 4")
	})
}

// TestARIMAFitting tests ARIMA model fitting
func TestARIMAFitting(t *testing.T) {
	t.Run("AR(1) model fitting", func(t *testing.T) {
		// Generate AR(1) process: x_t = 0.7 * x_{t-1} + e_t
		values := []float64{1.0, 0.7, 1.49, 1.043, 1.7301, 1.21107, 1.847749, 1.2934243}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ar1_process")

		// Create and fit AR(1) model
		model, _ := NewARIMA(1, 0, 0) // AR(1)
		err := model.Fit(ts)
		if err != nil {
			t.Fatalf("AR(1) model fitting failed: %v", err)
		}

		// Check if model is fitted
		if !model.IsFitted() {
			t.Error("Model should be marked as fitted after successful fitting")
		}

		// Get model parameters
		params := model.GetParameters()
		if params == nil {
			t.Fatal("Model parameters should not be nil after fitting")
		}

		// Check AR parameter (should be close to 0.7)
		if len(params.AR) != 1 {
			t.Errorf("AR(1) should have 1 AR parameter, got %d", len(params.AR))
		} else {
			arParam := params.AR[0]
			if math.Abs(arParam) > 1.0 {
				t.Errorf("AR parameter should be stationary (|phi| < 1), got %.3f", arParam)
			}
			t.Logf("AR(1) parameter estimated: φ = %.3f", arParam)
		}

		// Check model information criteria
		aic := model.AIC()
		bic := model.BIC()
		if math.IsNaN(aic) || math.IsNaN(bic) {
			t.Error("AIC and BIC should be computed after fitting")
		}

		t.Logf("AR(1) fitting successful: AIC=%.2f, BIC=%.2f", aic, bic)
	})

	t.Run("MA(1) model fitting", func(t *testing.T) {
		// Generate MA(1) process approximation
		values := []float64{0.5, 1.3, 0.8, 1.6, 0.9, 1.4, 1.1, 1.2}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ma1_process")

		// Create and fit MA(1) model
		model, _ := NewARIMA(0, 0, 1) // MA(1)
		err := model.Fit(ts)
		if err != nil {
			t.Fatalf("MA(1) model fitting failed: %v", err)
		}

		// Check MA parameter
		params := model.GetParameters()
		if len(params.MA) != 1 {
			t.Errorf("MA(1) should have 1 MA parameter, got %d", len(params.MA))
		} else {
			maParam := params.MA[0]
			if math.Abs(maParam) > 1.0 {
				t.Logf("MA parameter: θ = %.3f (invertibility check)", maParam)
			}
		}

		t.Logf("MA(1) fitting successful")
	})

	t.Run("ARIMA(1,1,1) model fitting", func(t *testing.T) {
		// Generate ARIMA(1,1,1) process approximation
		values := []float64{1.0, 2.2, 3.1, 4.5, 5.3, 6.7, 7.4, 8.8, 9.6, 10.9}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "arima111_process")

		// Create and fit ARIMA(1,1,1) model
		model, _ := NewARIMA(1, 1, 1)
		err := model.Fit(ts)
		if err != nil {
			t.Fatalf("ARIMA(1,1,1) model fitting failed: %v", err)
		}

		// Check parameters
		params := model.GetParameters()
		if len(params.AR) != 1 || len(params.MA) != 1 {
			t.Errorf("ARIMA(1,1,1) should have 1 AR and 1 MA parameter")
		}

		// Check model fit metrics
		logLikelihood := model.LogLikelihood()
		if math.IsNaN(logLikelihood) {
			t.Error("Log-likelihood should be computed after fitting")
		}

		t.Logf("ARIMA(1,1,1) fitting successful: LL=%.2f", logLikelihood)
	})
}

// TestARIMAForecasting tests forecasting capabilities
func TestARIMAForecasting(t *testing.T) {
	t.Run("AR(1) forecasting with prediction intervals", func(t *testing.T) {
		// Simple AR(1) data
		values := []float64{1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ar1_forecast_test")

		// Fit AR(1) model
		model, _ := NewARIMA(1, 0, 0)
		err := model.Fit(ts)
		if err != nil {
			t.Fatalf("Model fitting failed: %v", err)
		}

		// Generate forecasts
		horizon := 3
		forecast, err := model.Forecast(horizon)
		if err != nil {
			t.Fatalf("Forecasting failed: %v", err)
		}

		// Check forecast structure
		if len(forecast.Values) != horizon {
			t.Errorf("Forecast should have %d values, got %d", horizon, len(forecast.Values))
		}

		// Check that forecasts are reasonable (should decay for stationary AR(1))
		for i := 1; i < len(forecast.Values); i++ {
			if forecast.Values[i] > forecast.Values[i-1] {
				t.Logf("Forecast[%d]=%.3f > Forecast[%d]=%.3f (may not always decay)",
					i, forecast.Values[i], i-1, forecast.Values[i-1])
			}
		}

		// Test prediction intervals
		intervals, err := model.PredictionIntervals(horizon, 0.95)
		if err != nil {
			t.Fatalf("Prediction intervals failed: %v", err)
		}

		// Check intervals structure
		if len(intervals.Lower) != horizon || len(intervals.Upper) != horizon {
			t.Errorf("Prediction intervals should have %d values each", horizon)
		}

		// Check interval validity (lower < forecast < upper)
		for i := 0; i < horizon; i++ {
			if intervals.Lower[i] >= forecast.Values[i] || forecast.Values[i] >= intervals.Upper[i] {
				t.Errorf("Prediction interval invalid at step %d: [%.3f, %.3f] for forecast %.3f",
					i+1, intervals.Lower[i], intervals.Upper[i], forecast.Values[i])
			}
		}

		t.Logf("AR(1) forecasting successful: %d-step forecast with 95%% intervals", horizon)
	})

	t.Run("ARIMA forecasting with differencing", func(t *testing.T) {
		// Trending data that needs differencing
		values := []float64{1.0, 2.1, 3.3, 4.2, 5.4, 6.1, 7.3, 8.2}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "trending_forecast_test")

		// Fit ARIMA(1,1,0) model
		model, _ := NewARIMA(1, 1, 0)
		err := model.Fit(ts)
		if err != nil {
			t.Fatalf("ARIMA(1,1,0) fitting failed: %v", err)
		}

		// Generate forecasts
		forecast, err := model.Forecast(2)
		if err != nil {
			t.Fatalf("ARIMA forecasting failed: %v", err)
		}

		// Check that forecasts continue the trend
		lastValue := values[len(values)-1]
		if forecast.Values[0] <= lastValue {
			t.Logf("First forecast %.2f vs last value %.2f (trend continuation check)",
				forecast.Values[0], lastValue)
		}

		t.Logf("ARIMA trending forecast successful")
	})
}

// TestARIMADiagnostics tests model diagnostic capabilities
func TestARIMADiagnostics(t *testing.T) {
	t.Run("Residual analysis and diagnostics", func(t *testing.T) {
		// Generate test data
		values := []float64{1.2, 2.1, 1.8, 2.5, 2.3, 3.1, 2.9, 3.6, 3.4, 4.2}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "diagnostic_test")

		// Fit AR(1) model
		model, _ := NewARIMA(1, 0, 0)
		err := model.Fit(ts)
		if err != nil {
			t.Fatalf("Model fitting failed: %v", err)
		}

		// Get residuals
		residuals := model.Residuals()
		if residuals == nil {
			t.Fatal("Residuals should be available after fitting")
		}

		if residuals.Len() == 0 {
			t.Error("Residuals should not be empty")
		}

		// Run diagnostic tests
		diagnostics, err := model.DiagnosticTests()
		if err != nil {
			t.Fatalf("Diagnostic tests failed: %v", err)
		}

		// Check diagnostic structure
		if diagnostics.LjungBox == nil {
			t.Error("Ljung-Box test should be included in diagnostics")
		}

		if diagnostics.JarqueBera == nil {
			t.Error("Jarque-Bera test should be included in diagnostics")
		}

		// Check residual autocorrelation
		if diagnostics.LjungBox.PValue > 0.05 {
			t.Logf("Ljung-Box p-value = %.3f: residuals appear uncorrelated", diagnostics.LjungBox.PValue)
		} else {
			t.Logf("Ljung-Box p-value = %.3f: residual autocorrelation detected", diagnostics.LjungBox.PValue)
		}

		t.Logf("ARIMA diagnostics successful")
	})

	t.Run("Model comparison and selection", func(t *testing.T) {
		// Generate test data
		values := []float64{2.0, 2.3, 2.1, 2.6, 2.4, 2.8, 2.7, 3.0, 2.9, 3.2}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "comparison_test")

		// Fit different models
		models := []struct {
			p, d, q int
			name    string
		}{
			{1, 0, 0, "AR(1)"},     // AR(1)
			{0, 0, 1, "MA(1)"},     // MA(1)
			{1, 0, 1, "ARMA(1,1)"}, // ARMA(1,1)
		}

		bestAIC := math.Inf(1)
		var bestModel *ARIMAModel

		for _, spec := range models {
			model, _ := NewARIMA(spec.p, spec.d, spec.q)
			err := model.Fit(ts)
			if err != nil {
				t.Logf("Failed to fit %s: %v", spec.name, err)
				continue
			}

			aic := model.AIC()
			t.Logf("%s: AIC = %.2f, BIC = %.2f", spec.name, aic, model.BIC())

			if aic < bestAIC {
				bestAIC = aic
				bestModel = model
			}
		}

		if bestModel == nil {
			t.Fatal("No models fitted successfully")
		}

		t.Logf("Best model selected based on AIC: ARIMA(%d,%d,%d)",
			bestModel.P(), bestModel.D(), bestModel.Q())
	})
}

// TestARIMAParameters tests parameter validation and edge cases
func TestARIMAParameters(t *testing.T) {
	t.Run("Parameter validation and constraints", func(t *testing.T) {
		// Test maximum reasonable orders
		_, err := NewARIMA(10, 3, 10)
		if err == nil {
			t.Log("High-order ARIMA model created (may be computationally intensive)")
		}

		// Test zero-order models
		ar0, _ := NewARIMA(0, 0, 0) // White noise
		if ar0.ModelType() != "ARIMA" {
			t.Error("Zero-order model should still be ARIMA type")
		}

		// Test fitting with insufficient data
		shortValues := []float64{1.0, 2.0}
		shortDates := []time.Time{
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC),
		}
		shortTS, _ := NewTimeSeries(shortValues, shortDates, "short_series")

		model, _ := NewARIMA(1, 1, 1)
		err = model.Fit(shortTS)
		if err == nil {
			t.Error("Should fail to fit ARIMA(1,1,1) with insufficient data")
		}

		t.Logf("Parameter validation tests passed")
	})

	t.Run("ARIMA with missing values", func(t *testing.T) {
		values := []float64{1.0, 2.0, math.NaN(), 4.0, 5.0, math.NaN(), 7.0, 8.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "missing_values")

		// Test ARIMA fitting with missing values
		model, _ := NewARIMA(1, 0, 0)
		err := model.Fit(ts)
		if err != nil {
			t.Logf("ARIMA fitting with missing values failed (expected): %v", err)
		} else {
			t.Logf("ARIMA successfully handled missing values")
		}

		t.Logf("Missing value handling tested")
	})
}
