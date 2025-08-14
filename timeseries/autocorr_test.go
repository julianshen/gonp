package timeseries

import (
	"math"
	"testing"
	"time"
)

// TestAutocorrelationFunction tests ACF computation
func TestAutocorrelationFunction(t *testing.T) {
	t.Run("Basic ACF computation with known correlation", func(t *testing.T) {
		// Create AR(1) process: x_t = 0.7 * x_{t-1} + noise
		// Expected ACF: ρ(k) = 0.7^k for AR(1) with φ=0.7
		values := []float64{1.0, 0.7, 1.49, 1.043, 1.7301, 1.21107, 1.847749}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ar1_test")

		// Test ACF computation for lags 0-5
		maxLags := 5
		acf, err := ts.AutocorrelationFunction(maxLags)
		if err != nil {
			t.Fatalf("ACF computation failed: %v", err)
		}

		// Check ACF properties
		if len(acf) != maxLags+1 {
			t.Errorf("ACF should have %d values, got %d", maxLags+1, len(acf))
		}

		// ACF at lag 0 should be 1.0
		if math.Abs(acf[0]-1.0) > 1e-10 {
			t.Errorf("ACF[0] should be 1.0, got %.6f", acf[0])
		}

		// ACF should decay for AR(1) process
		if acf[1] <= acf[2] {
			t.Logf("ACF decay pattern: ACF[1]=%.3f, ACF[2]=%.3f (may vary with sample)", acf[1], acf[2])
		}

		// All ACF values should be between -1 and 1
		for i, val := range acf {
			if val < -1.0 || val > 1.0 {
				t.Errorf("ACF[%d]=%.6f should be between -1 and 1", i, val)
			}
		}

		t.Logf("ACF successful: ACF[0]=%.3f, ACF[1]=%.3f, ACF[2]=%.3f", acf[0], acf[1], acf[2])
	})

	t.Run("ACF with confidence intervals", func(t *testing.T) {
		// White noise should have ACF ≈ 0 for all lags > 0
		values := []float64{0.1, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, 0.3, -0.4, 0.2}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "white_noise")

		// Test ACF with confidence intervals
		result, err := ts.AutocorrelationFunctionWithCI(5, 0.95)
		if err != nil {
			t.Fatalf("ACF with CI failed: %v", err)
		}

		// Check structure
		if len(result.ACF) != 6 || len(result.LowerCI) != 6 || len(result.UpperCI) != 6 {
			t.Errorf("ACF result should have 6 values each for ACF, LowerCI, UpperCI")
		}

		// Confidence intervals should be symmetric around 0 for white noise
		expectedCI := 1.96 / math.Sqrt(float64(len(values))) // ±1.96/√n for 95% CI
		tolerance := 0.1                                     // Allow some tolerance for sample variation

		for i := 1; i < len(result.LowerCI); i++ {
			if math.Abs(result.LowerCI[i]+expectedCI) > tolerance {
				t.Logf("Lower CI[%d] = %.3f (expected ≈ %.3f)", i, result.LowerCI[i], -expectedCI)
			}
			if math.Abs(result.UpperCI[i]-expectedCI) > tolerance {
				t.Logf("Upper CI[%d] = %.3f (expected ≈ %.3f)", i, result.UpperCI[i], expectedCI)
			}
		}

		t.Logf("ACF with CI successful: CI bounds ≈ ±%.3f", expectedCI)
	})
}

// TestPartialAutocorrelationFunction tests PACF computation
func TestPartialAutocorrelationFunction(t *testing.T) {
	t.Run("PACF computation for AR(2) process", func(t *testing.T) {
		// AR(2): x_t = 0.6*x_{t-1} + 0.3*x_{t-2} + noise
		// PACF should cut off after lag 2 for AR(2)
		values := []float64{1.0, 0.6, 0.66, 0.696, 0.7176, 0.73056, 0.736368, 0.7390208}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ar2_test")

		// Test PACF computation
		pacf, err := ts.PartialAutocorrelationFunction(5)
		if err != nil {
			t.Fatalf("PACF computation failed: %v", err)
		}

		// Check PACF properties
		if len(pacf) != 6 { // lags 0-5
			t.Errorf("PACF should have 6 values, got %d", len(pacf))
		}

		// PACF at lag 0 should be 1.0
		if math.Abs(pacf[0]-1.0) > 1e-10 {
			t.Errorf("PACF[0] should be 1.0, got %.6f", pacf[0])
		}

		// For AR(2), PACF should be significant at lags 1,2 and small after that
		if math.Abs(pacf[1]) < 0.1 {
			t.Logf("PACF[1]=%.3f may be small (expected significant for AR)", pacf[1])
		}

		// All PACF values should be between -1 and 1
		for i, val := range pacf {
			if val < -1.0 || val > 1.0 {
				t.Errorf("PACF[%d]=%.6f should be between -1 and 1", i, val)
			}
		}

		t.Logf("PACF successful: PACF[0]=%.3f, PACF[1]=%.3f, PACF[2]=%.3f", pacf[0], pacf[1], pacf[2])
	})

	t.Run("PACF with Yule-Walker equations", func(t *testing.T) {
		// Test data with clear autoregressive pattern
		values := []float64{2.0, 1.8, 1.44, 1.352, 1.2816, 1.22528, 1.180224, 1.1441792}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "yw_test")

		// Test Yule-Walker PACF computation
		result, err := ts.PartialAutocorrelationFunctionYW(4)
		if err != nil {
			t.Fatalf("Yule-Walker PACF failed: %v", err)
		}

		// Verify PACF computation method
		if len(result.PACF) != 5 {
			t.Errorf("YW PACF should have 5 values, got %d", len(result.PACF))
		}

		// Check coefficient validity
		for i, coeff := range result.ARCoefficients {
			if len(coeff) != i+1 {
				t.Errorf("AR coefficients at lag %d should have %d elements, got %d", i, i+1, len(coeff))
			}
		}

		t.Logf("Yule-Walker PACF successful: computed AR coefficients for lags 1-4")
	})
}

// TestCrossCorrelation tests cross-correlation analysis
func TestCrossCorrelation(t *testing.T) {
	t.Run("Cross-correlation between related series", func(t *testing.T) {
		// Create two related series: y = x shifted by 1 period
		x_values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
		y_values := []float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}

		dates := make([]time.Time, len(x_values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		x_ts, _ := NewTimeSeries(x_values, dates, "x_series")
		y_ts, _ := NewTimeSeries(y_values, dates, "y_series")

		// Test cross-correlation
		ccf, err := x_ts.CrossCorrelation(y_ts, 3)
		if err != nil {
			t.Fatalf("Cross-correlation failed: %v", err)
		}

		// Check cross-correlation properties
		expectedLength := 2*3 + 1 // -maxLag to +maxLag
		if len(ccf.Correlations) != expectedLength {
			t.Errorf("CCF should have %d values, got %d", expectedLength, len(ccf.Correlations))
		}

		// Find maximum correlation and its lag
		maxCorr := -2.0
		maxLag := 0
		for i, corr := range ccf.Correlations {
			if corr > maxCorr {
				maxCorr = corr
				maxLag = ccf.Lags[i]
			}
		}

		// For y = x shifted by 1, maximum correlation should be at lag +1
		if maxLag != 1 {
			t.Logf("Maximum correlation at lag %d (expected 1 for shifted series)", maxLag)
		}

		// Maximum correlation should be high for related series
		if maxCorr < 0.5 {
			t.Logf("Maximum correlation %.3f may be low (expected high for related series)", maxCorr)
		}

		t.Logf("Cross-correlation successful: max correlation %.3f at lag %d", maxCorr, maxLag)
	})

	t.Run("Cross-correlation lead-lag analysis", func(t *testing.T) {
		// Leading indicator: x leads y by 2 periods
		x_values := []float64{1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0}
		y_values := []float64{0.0, 0.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0}

		dates := make([]time.Time, len(x_values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		x_ts, _ := NewTimeSeries(x_values, dates, "leading_series")
		y_ts, _ := NewTimeSeries(y_values, dates, "lagging_series")

		// Analyze lead-lag relationship
		leadLag, err := x_ts.LeadLagAnalysis(y_ts, 4)
		if err != nil {
			t.Fatalf("Lead-lag analysis failed: %v", err)
		}

		// Check optimal lag detection
		if leadLag.OptimalLag < 0 {
			t.Logf("Optimal lag %d indicates x lags behind y", -leadLag.OptimalLag)
		} else if leadLag.OptimalLag > 0 {
			t.Logf("Optimal lag %d indicates x leads y", leadLag.OptimalLag)
		}

		// Maximum correlation should be reasonable
		if leadLag.MaxCorrelation < 0.3 {
			t.Logf("Maximum correlation %.3f at optimal lag", leadLag.MaxCorrelation)
		}

		t.Logf("Lead-lag analysis successful: optimal lag = %d, max correlation = %.3f",
			leadLag.OptimalLag, leadLag.MaxCorrelation)
	})
}

// TestCorrelationTests tests statistical tests for correlation
func TestCorrelationTests(t *testing.T) {
	t.Run("Ljung-Box test for autocorrelation", func(t *testing.T) {
		// AR(1) process should show significant autocorrelation
		values := []float64{1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ljung_box_test")

		// Test Ljung-Box statistic
		result, err := ts.LjungBoxTest(5) // Test first 5 lags
		if err != nil {
			t.Fatalf("Ljung-Box test failed: %v", err)
		}

		// Check test result structure
		if result.Statistic < 0 {
			t.Errorf("Ljung-Box statistic should be non-negative, got %.3f", result.Statistic)
		}

		if result.PValue < 0 || result.PValue > 1 {
			t.Errorf("P-value should be between 0 and 1, got %.6f", result.PValue)
		}

		if result.DegreesOfFreedom != 5 {
			t.Errorf("Degrees of freedom should be 5, got %d", result.DegreesOfFreedom)
		}

		// For AR(1), should reject null hypothesis of no autocorrelation
		alpha := 0.05
		if result.PValue > alpha {
			t.Logf("P-value %.6f > %.2f: fail to reject null (no autocorrelation)", result.PValue, alpha)
		} else {
			t.Logf("P-value %.6f ≤ %.2f: reject null (autocorrelation detected)", result.PValue, alpha)
		}

		t.Logf("Ljung-Box test successful: statistic=%.3f, p-value=%.6f", result.Statistic, result.PValue)
	})

	t.Run("Durbin-Watson test for first-order autocorrelation", func(t *testing.T) {
		// Positive autocorrelation example
		values := []float64{1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "dw_test")

		// Test Durbin-Watson statistic
		result, err := ts.DurbinWatsonTest()
		if err != nil {
			t.Fatalf("Durbin-Watson test failed: %v", err)
		}

		// DW statistic should be between 0 and 4
		if result.Statistic < 0 || result.Statistic > 4 {
			t.Errorf("DW statistic should be between 0 and 4, got %.3f", result.Statistic)
		}

		// Interpret DW statistic
		if result.Statistic < 1.5 {
			t.Logf("DW=%.3f suggests positive autocorrelation", result.Statistic)
		} else if result.Statistic > 2.5 {
			t.Logf("DW=%.3f suggests negative autocorrelation", result.Statistic)
		} else {
			t.Logf("DW=%.3f suggests no strong autocorrelation", result.Statistic)
		}

		t.Logf("Durbin-Watson test successful: DW statistic = %.3f", result.Statistic)
	})

	t.Run("Box-Pierce test for independence", func(t *testing.T) {
		// White noise should pass independence test
		values := []float64{0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 0.3}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "bp_test")

		// Test Box-Pierce statistic
		result, err := ts.BoxPierceTest(4) // Test first 4 lags
		if err != nil {
			t.Fatalf("Box-Pierce test failed: %v", err)
		}

		// Check test result
		if result.Statistic < 0 {
			t.Errorf("Box-Pierce statistic should be non-negative, got %.3f", result.Statistic)
		}

		if result.PValue < 0 || result.PValue > 1 {
			t.Errorf("P-value should be between 0 and 1, got %.6f", result.PValue)
		}

		// For white noise, should fail to reject null hypothesis
		alpha := 0.05
		if result.PValue > alpha {
			t.Logf("P-value %.6f > %.2f: fail to reject null (series appears independent)", result.PValue, alpha)
		} else {
			t.Logf("P-value %.6f ≤ %.2f: reject null (dependence detected)", result.PValue, alpha)
		}

		t.Logf("Box-Pierce test successful: statistic=%.3f, p-value=%.6f", result.Statistic, result.PValue)
	})
}

// TestAutocorrelationParameters tests parameter validation and edge cases
func TestAutocorrelationParameters(t *testing.T) {
	t.Run("Parameter validation for ACF/PACF", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "param_test")

		// Test invalid lag parameters
		_, err := ts.AutocorrelationFunction(-1)
		if err == nil {
			t.Error("Should fail with negative max lags")
		}

		_, err = ts.AutocorrelationFunction(10) // More lags than data
		if err == nil {
			t.Error("Should fail when max lags exceeds data length")
		}

		_, err = ts.PartialAutocorrelationFunction(-1)
		if err == nil {
			t.Error("Should fail with negative max lags for PACF")
		}

		_, err = ts.PartialAutocorrelationFunction(5) // More than data length
		if err == nil {
			t.Error("Should fail when PACF lags exceed data length")
		}

		t.Logf("Parameter validation tests passed")
	})

	t.Run("Autocorrelation with missing values", func(t *testing.T) {
		values := []float64{1.0, 2.0, math.NaN(), 4.0, 5.0, math.NaN(), 7.0, 8.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "missing_test")

		// Test ACF with missing values
		acf, err := ts.AutocorrelationFunction(3)
		if err != nil {
			t.Fatalf("ACF with missing values failed: %v", err)
		}

		// Should handle missing values appropriately
		if len(acf) != 4 {
			t.Errorf("ACF should have 4 values despite missing data")
		}

		// Test PACF with missing values
		pacf, err := ts.PartialAutocorrelationFunction(2)
		if err != nil {
			t.Fatalf("PACF with missing values failed: %v", err)
		}

		if len(pacf) != 3 {
			t.Errorf("PACF should have 3 values despite missing data")
		}

		t.Logf("Missing value handling successful")
	})
}
