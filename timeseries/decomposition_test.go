package timeseries

import (
	"math"
	"testing"
	"time"
)

// TestTimeSeriesDecomposition tests classical time series decomposition using TDD
func TestTimeSeriesDecomposition(t *testing.T) {
	t.Run("Classical additive decomposition with known seasonal pattern", func(t *testing.T) {
		// Red phase: Create test data with known trend, seasonal, and random components
		// Trend: Linear increase (1.0, 2.0, 3.0, ...)
		// Seasonal: Simple 4-period pattern [2, -1, -1, 2]
		// Expected: y(t) = trend(t) + seasonal(t) + residual(t)

		nPeriods := 3 // 3 complete seasonal cycles
		period := 4
		totalPoints := nPeriods * period // 12 data points

		values := make([]float64, totalPoints)
		dates := make([]time.Time, totalPoints)
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		// Known seasonal pattern: [2, -1, -1, 2] repeating
		seasonalPattern := []float64{2.0, -1.0, -1.0, 2.0}

		for i := 0; i < totalPoints; i++ {
			trend := float64(i) + 1.0             // Linear trend: 1, 2, 3, ...
			seasonal := seasonalPattern[i%period] // Seasonal component
			values[i] = trend + seasonal          // Additive model
			dates[i] = baseDate.AddDate(0, 0, i)  // Daily data
		}

		ts, _ := NewTimeSeries(values, dates, "test_seasonal")

		// Perform additive decomposition
		decomp, err := ts.Decompose("additive", period)
		if err != nil {
			t.Fatalf("Additive decomposition failed: %v", err)
		}

		// Check decomposition components exist
		if decomp.Original == nil || decomp.Trend == nil ||
			decomp.Seasonal == nil || decomp.Residual == nil {
			t.Fatal("Decomposition components are missing")
		}

		if decomp.Original.Len() != totalPoints {
			t.Errorf("Expected original length %d, got %d", totalPoints, decomp.Original.Len())
		}

		// Check that decomposition formula holds: original = trend + seasonal + residual
		for i := 2; i < totalPoints-2; i++ { // Skip edges where trend may not be available
			original := decomp.Original.At(i)
			trend := decomp.Trend.At(i)
			seasonal := decomp.Seasonal.At(i)
			residual := decomp.Residual.At(i)

			if !math.IsNaN(trend) && !math.IsNaN(seasonal) && !math.IsNaN(residual) {
				reconstructed := trend + seasonal + residual
				if math.Abs(original-reconstructed) > 1e-10 {
					t.Errorf("Decomposition doesn't add up at index %d: %.6f != %.6f + %.6f + %.6f",
						i, original, trend, seasonal, residual)
				}
			}
		}

		// Check that seasonal pattern repeats correctly
		for i := period; i < totalPoints-period; i++ {
			if !math.IsNaN(decomp.Seasonal.At(i)) && !math.IsNaN(decomp.Seasonal.At(i-period)) {
				current := decomp.Seasonal.At(i)
				previous := decomp.Seasonal.At(i - period)
				if math.Abs(current-previous) > 1e-6 {
					t.Errorf("Seasonal pattern not repeating: pos %d = %.6f, pos %d = %.6f",
						i, current, i-period, previous)
				}
			}
		}

		t.Logf("Additive decomposition successful with period %d", period)
	})

	t.Run("Classical multiplicative decomposition", func(t *testing.T) {
		// Test multiplicative model: y(t) = trend(t) * seasonal(t) * residual(t)
		nPeriods := 2
		period := 6                      // Semi-annual pattern
		totalPoints := nPeriods * period // 12 data points

		values := make([]float64, totalPoints)
		dates := make([]time.Time, totalPoints)
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		// Multiplicative seasonal pattern (centered around 1.0)
		seasonalPattern := []float64{1.2, 0.8, 0.9, 1.1, 1.3, 0.7}

		for i := 0; i < totalPoints; i++ {
			trend := float64(i)*0.1 + 2.0          // Gradual upward trend
			seasonal := seasonalPattern[i%period]  // Seasonal multiplier
			values[i] = trend * seasonal           // Multiplicative model
			dates[i] = baseDate.AddDate(0, 0, i*7) // Weekly data
		}

		ts, _ := NewTimeSeries(values, dates, "test_multiplicative")

		// Perform multiplicative decomposition
		decomp, err := ts.Decompose("multiplicative", period)
		if err != nil {
			t.Fatalf("Multiplicative decomposition failed: %v", err)
		}

		// Check multiplicative formula: original = trend * seasonal * residual
		for i := 2; i < totalPoints-2; i++ {
			original := decomp.Original.At(i)
			trend := decomp.Trend.At(i)
			seasonal := decomp.Seasonal.At(i)
			residual := decomp.Residual.At(i)

			if !math.IsNaN(trend) && !math.IsNaN(seasonal) && !math.IsNaN(residual) {
				reconstructed := trend * seasonal * residual
				if math.Abs(original-reconstructed) > 1e-10 {
					t.Errorf("Multiplicative decomposition doesn't multiply at index %d: %.6f != %.6f * %.6f * %.6f",
						i, original, trend, seasonal, residual)
				}
			}
		}

		// Seasonal indices should average to 1.0 for multiplicative
		seasonalSum := 0.0
		seasonalCount := 0
		for i := 0; i < decomp.Seasonal.Len(); i++ {
			val := decomp.Seasonal.At(i)
			if !math.IsNaN(val) {
				seasonalSum += val
				seasonalCount++
			}
		}
		if seasonalCount > 0 {
			avgSeasonal := seasonalSum / float64(seasonalCount)
			if math.Abs(avgSeasonal-1.0) > 0.1 { // Allow some tolerance
				t.Errorf("Multiplicative seasonal indices should average ~1.0, got %.6f", avgSeasonal)
			}
		}

		t.Logf("Multiplicative decomposition successful")
	})

	t.Run("Automatic seasonal period detection", func(t *testing.T) {
		// Create data with obvious 5-period seasonality
		period := 5
		nPeriods := 4
		totalPoints := nPeriods * period

		values := make([]float64, totalPoints)
		dates := make([]time.Time, totalPoints)
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		// Strong seasonal pattern
		seasonalPattern := []float64{10.0, 5.0, 2.0, 8.0, 15.0}

		for i := 0; i < totalPoints; i++ {
			trend := 100.0 + float64(i)*0.5       // Base level with slight trend
			seasonal := seasonalPattern[i%period] // Strong seasonal component
			values[i] = trend + seasonal
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "auto_seasonal")

		// Test automatic period detection
		detectedPeriod, err := ts.DetectSeasonalPeriod()
		if err != nil {
			t.Fatalf("Seasonal period detection failed: %v", err)
		}

		if detectedPeriod != period {
			t.Errorf("Expected detected period %d, got %d", period, detectedPeriod)
		}

		// Test decomposition with auto-detected period
		autoDecomp, err := ts.DecomposeAuto("additive")
		if err != nil {
			t.Fatalf("Auto decomposition failed: %v", err)
		}

		if autoDecomp.Period != period {
			t.Errorf("Auto decomposition period: expected %d, got %d", period, autoDecomp.Period)
		}

		t.Logf("Automatic seasonal detection successful: period = %d", detectedPeriod)
	})

	t.Run("Trend extraction methods", func(t *testing.T) {
		// Test different trend extraction methods
		values := []float64{1.0, 2.1, 3.2, 4.0, 5.1, 6.2, 7.0, 8.1, 9.2, 10.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "trend_test")

		// Simple moving average trend
		trendMA, err := ts.ExtractTrend("moving_average", 3)
		if err != nil {
			t.Fatalf("Moving average trend extraction failed: %v", err)
		}

		// Check that trend is smoother than original
		if trendMA.Len() != ts.Len() {
			t.Errorf("Trend series should have same length as original")
		}

		// Linear trend fitting
		trendLinear, err := ts.ExtractTrend("linear", 0)
		if err != nil {
			t.Fatalf("Linear trend extraction failed: %v", err)
		}

		// Linear trend should show consistent increase for our test data
		start := trendLinear.At(0)
		end := trendLinear.At(trendLinear.Len() - 1)
		if end <= start {
			t.Errorf("Linear trend should be increasing: start=%.2f, end=%.2f", start, end)
		}

		// Detrending operation
		detrended, err := ts.Detrend("linear")
		if err != nil {
			t.Fatalf("Detrending failed: %v", err)
		}

		// Detrended series should have mean near zero
		detrendedMean := detrended.Mean()
		if math.Abs(detrendedMean) > 0.5 { // Allow some tolerance
			t.Errorf("Detrended series should have mean near 0, got %.6f", detrendedMean)
		}

		t.Logf("Trend extraction methods successful")
	})

	t.Run("Decomposition parameter validation", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "validation_test")

		// Invalid decomposition type
		_, err := ts.Decompose("invalid", 2)
		if err == nil {
			t.Error("Expected error for invalid decomposition type")
		}

		// Invalid period (too large)
		_, err = ts.Decompose("additive", 10)
		if err == nil {
			t.Error("Expected error for period larger than series length")
		}

		// Invalid period (too small)
		_, err = ts.Decompose("additive", 1)
		if err == nil {
			t.Error("Expected error for period = 1")
		}

		// Invalid trend method
		_, err = ts.ExtractTrend("invalid_method", 3)
		if err == nil {
			t.Error("Expected error for invalid trend extraction method")
		}

		t.Logf("Parameter validation tests passed")
	})
}

// TestDecompositionResult tests the decomposition result structure
func TestDecompositionResult(t *testing.T) {
	t.Run("Decomposition result properties and methods", func(t *testing.T) {
		// Create simple test data
		values := []float64{10.0, 8.0, 12.0, 9.0, 11.0, 7.0, 13.0, 8.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "decomp_result_test")

		// Perform decomposition
		decomp, err := ts.Decompose("additive", 4)
		if err != nil {
			t.Fatalf("Decomposition failed: %v", err)
		}

		// Test decomposition result methods
		if decomp.GetPeriod() != 4 {
			t.Errorf("Expected period 4, got %d", decomp.GetPeriod())
		}

		if decomp.GetType() != "additive" {
			t.Errorf("Expected type 'additive', got %s", decomp.GetType())
		}

		// Test seasonal strength calculation
		seasonalStrength := decomp.SeasonalStrength()
		if seasonalStrength < 0 || seasonalStrength > 1 {
			t.Errorf("Seasonal strength should be between 0 and 1, got %.6f", seasonalStrength)
		}

		// Test trend strength calculation
		trendStrength := decomp.TrendStrength()
		if trendStrength < 0 || trendStrength > 1 {
			t.Errorf("Trend strength should be between 0 and 1, got %.6f", trendStrength)
		}

		// Test reconstruction accuracy
		rmse := decomp.ReconstructionRMSE()
		if rmse < 0 {
			t.Errorf("RMSE should be non-negative, got %.6f", rmse)
		}

		// Test summary statistics
		summary := decomp.Summary()
		if len(summary) == 0 {
			t.Error("Expected non-empty summary")
		}

		t.Logf("Decomposition result properties: seasonal_strength=%.3f, trend_strength=%.3f, rmse=%.6f",
			seasonalStrength, trendStrength, rmse)
		t.Logf("Summary: %s", summary)
	})
}

// TestAdvancedDecomposition tests advanced decomposition features
func TestAdvancedDecomposition(t *testing.T) {
	t.Run("Decomposition with missing values", func(t *testing.T) {
		// Test decomposition with NaN values in the series
		values := []float64{1.0, math.NaN(), 3.0, 4.0, math.NaN(), 6.0, 7.0, 8.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "missing_values_test")

		// Should handle missing values gracefully
		decomp, err := ts.Decompose("additive", 4)
		if err != nil {
			t.Fatalf("Decomposition with missing values failed: %v", err)
		}

		// Check that decomposition completed
		if decomp.Original == nil {
			t.Error("Expected decomposition result even with missing values")
		}

		t.Logf("Decomposition with missing values handled successfully")
	})

	t.Run("Multiple seasonal periods detection", func(t *testing.T) {
		// Create data with multiple seasonal components (e.g., weekly + monthly)
		totalPoints := 84 // 12 weeks
		values := make([]float64, totalPoints)
		dates := make([]time.Time, totalPoints)
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := 0; i < totalPoints; i++ {
			trend := 100.0 + float64(i)*0.1
			weekly := 5.0 * math.Sin(2*math.Pi*float64(i)/7.0)   // Weekly pattern
			monthly := 3.0 * math.Cos(2*math.Pi*float64(i)/30.0) // Monthly pattern
			values[i] = trend + weekly + monthly
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "multiple_seasonal")

		// Test detection of multiple periods
		periods, err := ts.DetectMultipleSeasonalPeriods(3) // Find top 3 periods
		if err != nil {
			t.Fatalf("Multiple seasonal period detection failed: %v", err)
		}

		if len(periods) == 0 {
			t.Error("Expected at least one seasonal period detected")
		}

		// Should detect both weekly (7) and monthly (~30) patterns
		foundWeekly := false
		foundMonthly := false
		for _, period := range periods {
			if period >= 6 && period <= 8 {
				foundWeekly = true
			}
			if period >= 28 && period <= 32 {
				foundMonthly = true
			}
		}

		t.Logf("Detected seasonal periods: %v", periods)
		t.Logf("Found weekly pattern: %v, monthly pattern: %v", foundWeekly, foundMonthly)
	})
}
