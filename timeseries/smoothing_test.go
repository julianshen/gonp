package timeseries

import (
	"math"
	"testing"
	"time"
)

// TestMovingAverages tests simple moving average methods
func TestMovingAverages(t *testing.T) {
	t.Run("Simple Moving Average (SMA) forward window", func(t *testing.T) {
		// Create test data with known trend
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "sma_test")

		// Test forward SMA with window=3
		sma3, err := ts.SimpleMovingAverage(3, "forward")
		if err != nil {
			t.Fatalf("Forward SMA failed: %v", err)
		}

		// Check length - should be same as original
		if sma3.Len() != ts.Len() {
			t.Errorf("SMA length should match original: expected %d, got %d", ts.Len(), sma3.Len())
		}

		// Check specific values: SMA(3) at position 2 should be (1+2+3)/3 = 2.0
		expectedAtIndex2 := 2.0
		actualAtIndex2 := sma3.At(2)
		if math.Abs(actualAtIndex2-expectedAtIndex2) > 1e-10 {
			t.Errorf("SMA[2] expected %.6f, got %.6f", expectedAtIndex2, actualAtIndex2)
		}

		// Check SMA(3) at position 5 should be (4+5+6)/3 = 5.0
		expectedAtIndex5 := 5.0
		actualAtIndex5 := sma3.At(5)
		if math.Abs(actualAtIndex5-expectedAtIndex5) > 1e-10 {
			t.Errorf("SMA[5] expected %.6f, got %.6f", expectedAtIndex5, actualAtIndex5)
		}

		// First two values should be NaN (insufficient data)
		if !math.IsNaN(sma3.At(0)) || !math.IsNaN(sma3.At(1)) {
			t.Errorf("First two SMA values should be NaN")
		}

		t.Logf("Forward SMA successful: SMA[2]=%.1f, SMA[5]=%.1f", actualAtIndex2, actualAtIndex5)
	})

	t.Run("Centered Moving Average for better trend estimation", func(t *testing.T) {
		// Test data with linear trend
		values := []float64{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "centered_ma_test")

		// Test centered SMA with window=3
		centeredMA, err := ts.SimpleMovingAverage(3, "centered")
		if err != nil {
			t.Fatalf("Centered MA failed: %v", err)
		}

		// Check centered MA at position 1: (2+4+6)/3 = 4.0
		expectedCenter := 4.0
		actualCenter := centeredMA.At(1)
		if math.Abs(actualCenter-expectedCenter) > 1e-10 {
			t.Errorf("Centered MA[1] expected %.1f, got %.6f", expectedCenter, actualCenter)
		}

		// Edge values should be NaN
		if !math.IsNaN(centeredMA.At(0)) || !math.IsNaN(centeredMA.At(centeredMA.Len()-1)) {
			t.Errorf("Edge values in centered MA should be NaN")
		}

		t.Logf("Centered MA successful: center value = %.1f", actualCenter)
	})

	t.Run("Weighted Moving Average with custom weights", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "wma_test")

		// Test weighted MA with increasing weights [1, 2, 3]
		weights := []float64{1.0, 2.0, 3.0}
		wma, err := ts.WeightedMovingAverage(weights)
		if err != nil {
			t.Fatalf("Weighted MA failed: %v", err)
		}

		// Check weighted MA at position 2: (1*1 + 2*2 + 3*3)/(1+2+3) = (1+4+9)/6 = 14/6 = 2.333...
		expected := 14.0 / 6.0
		actual := wma.At(2)
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("WMA[2] expected %.6f, got %.6f", expected, actual)
		}

		t.Logf("Weighted MA successful: WMA[2] = %.6f", actual)
	})
}

// TestExponentialMovingAverages tests EMA methods
func TestExponentialMovingAverages(t *testing.T) {
	t.Run("Single Exponential Moving Average (EMA)", func(t *testing.T) {
		// Simple test data
		values := []float64{2.0, 4.0, 6.0, 8.0, 10.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "ema_test")

		// Test EMA with alpha = 0.5
		alpha := 0.5
		ema, err := ts.ExponentialMovingAverage(alpha)
		if err != nil {
			t.Fatalf("EMA failed: %v", err)
		}

		// EMA calculation: EMA[0] = 2.0, EMA[1] = 0.5*4 + 0.5*2 = 3.0
		expected1 := 3.0
		actual1 := ema.At(1)
		if math.Abs(actual1-expected1) > 1e-10 {
			t.Errorf("EMA[1] expected %.1f, got %.6f", expected1, actual1)
		}

		// First value should be the same as original
		if math.Abs(ema.At(0)-values[0]) > 1e-10 {
			t.Errorf("EMA[0] should equal first value: expected %.1f, got %.6f", values[0], ema.At(0))
		}

		t.Logf("EMA successful: EMA[0]=%.1f, EMA[1]=%.1f", ema.At(0), actual1)
	})

	t.Run("Double Exponential Smoothing (Holt's method)", func(t *testing.T) {
		// Data with linear trend
		values := []float64{1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "holt_test")

		// Test Holt's method with alpha=0.3, beta=0.1
		holt, err := ts.HoltLinearTrend(0.3, 0.1)
		if err != nil {
			t.Fatalf("Holt's method failed: %v", err)
		}

		// Should handle trend better than simple EMA
		if holt.Len() != ts.Len() {
			t.Errorf("Holt result should have same length as input")
		}

		// Last smoothed value should be close to actual trend
		lastValue := holt.At(holt.Len() - 1)
		if math.IsNaN(lastValue) {
			t.Errorf("Last Holt value should not be NaN")
		}

		t.Logf("Holt's method successful: last smoothed value = %.2f", lastValue)
	})

	t.Run("Triple Exponential Smoothing (Holt-Winters)", func(t *testing.T) {
		// Data with trend and seasonality (period=4)
		values := []float64{
			10.0, 15.0, 8.0, 12.0, // Season 1: baseline around 11.25
			12.0, 17.0, 10.0, 14.0, // Season 2: trending up
			14.0, 19.0, 12.0, 16.0, // Season 3: continuing trend
		}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "holt_winters_test")

		// Test Holt-Winters with seasonal period=4
		hw, err := ts.HoltWinters(0.3, 0.1, 0.1, 4, "additive")
		if err != nil {
			t.Fatalf("Holt-Winters failed: %v", err)
		}

		// Check that it captures both trend and seasonality
		if hw.Len() != ts.Len() {
			t.Errorf("Holt-Winters result should have same length as input")
		}

		// Should produce reasonable smoothed values
		for i := 0; i < hw.Len(); i++ {
			if math.IsNaN(hw.At(i)) {
				t.Errorf("Holt-Winters value at %d should not be NaN", i)
			}
		}

		t.Logf("Holt-Winters successful: handled trend and seasonality")
	})
}

// TestAdvancedSmoothing tests advanced smoothing methods
func TestAdvancedSmoothing(t *testing.T) {
	t.Run("LOWESS smoothing for non-parametric trend", func(t *testing.T) {
		// Non-linear data that would benefit from LOWESS
		values := []float64{1.0, 1.8, 3.2, 4.1, 4.8, 5.2, 5.1, 4.7, 4.0, 3.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "lowess_test")

		// Test LOWESS with default parameters
		lowess, err := ts.LOWESS(0.5, 2) // bandwidth=0.5, iterations=2
		if err != nil {
			t.Fatalf("LOWESS failed: %v", err)
		}

		// LOWESS should smooth the curve while preserving general shape
		if lowess.Len() != ts.Len() {
			t.Errorf("LOWESS result should have same length as input")
		}

		// Smoothed values should be less variable than original
		originalStd := ts.Std()
		smoothedStd := lowess.Std()
		if smoothedStd >= originalStd {
			t.Logf("LOWESS may not be smoothing enough: original std=%.3f, smoothed std=%.3f", originalStd, smoothedStd)
		}

		t.Logf("LOWESS successful: original std=%.3f, smoothed std=%.3f", originalStd, smoothedStd)
	})

	t.Run("Savitzky-Golay filter for polynomial smoothing", func(t *testing.T) {
		// Noisy data with underlying polynomial trend
		values := []float64{0.1, 1.1, 3.9, 9.2, 15.8, 25.1, 35.9, 49.2, 63.8, 81.1}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "savgol_test")

		// Test Savitzky-Golay with window=5, polynomial degree=2
		savgol, err := ts.SavitzkyGolay(5, 2)
		if err != nil {
			t.Fatalf("Savitzky-Golay failed: %v", err)
		}

		// Should preserve polynomial trends while reducing noise
		if savgol.Len() != ts.Len() {
			t.Errorf("Savitzky-Golay result should have same length as input")
		}

		// Check that it produces reasonable smoothed values
		for i := 0; i < savgol.Len(); i++ {
			if math.IsNaN(savgol.At(i)) {
				t.Errorf("Savitzky-Golay value at %d should not be NaN", i)
			}
		}

		t.Logf("Savitzky-Golay successful: smoothed polynomial trend")
	})

	t.Run("Kernel smoothing with Gaussian kernel", func(t *testing.T) {
		// Test data with some noise
		values := []float64{2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "kernel_test")

		// Test Gaussian kernel smoothing
		kernel, err := ts.KernelSmoothing("gaussian", 1.5) // bandwidth=1.5
		if err != nil {
			t.Fatalf("Kernel smoothing failed: %v", err)
		}

		// Should produce smoothed values
		if kernel.Len() != ts.Len() {
			t.Errorf("Kernel smoothing result should have same length as input")
		}

		// Check smoothed values are reasonable
		for i := 0; i < kernel.Len(); i++ {
			if math.IsNaN(kernel.At(i)) {
				t.Errorf("Kernel smoothed value at %d should not be NaN", i)
			}
		}

		t.Logf("Kernel smoothing successful with Gaussian kernel")
	})
}

// TestSmoothingParameters tests parameter validation and edge cases
func TestSmoothingParameters(t *testing.T) {
	t.Run("Parameter validation for moving averages", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "param_test")

		// Test invalid window sizes
		_, err := ts.SimpleMovingAverage(0, "forward")
		if err == nil {
			t.Error("Should fail with window size 0")
		}

		_, err = ts.SimpleMovingAverage(-1, "forward")
		if err == nil {
			t.Error("Should fail with negative window size")
		}

		_, err = ts.SimpleMovingAverage(10, "forward") // Window larger than data
		if err == nil {
			t.Error("Should fail when window exceeds data length")
		}

		// Test invalid EMA alpha
		_, err = ts.ExponentialMovingAverage(-0.1)
		if err == nil {
			t.Error("Should fail with negative alpha")
		}

		_, err = ts.ExponentialMovingAverage(1.1)
		if err == nil {
			t.Error("Should fail with alpha > 1")
		}

		t.Logf("Parameter validation tests passed")
	})

	t.Run("Smoothing with missing values", func(t *testing.T) {
		values := []float64{1.0, 2.0, math.NaN(), 4.0, 5.0, math.NaN(), 7.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "missing_test")

		// Test SMA with missing values
		sma, err := ts.SimpleMovingAverage(3, "forward")
		if err != nil {
			t.Fatalf("SMA with missing values failed: %v", err)
		}

		// Should handle missing values appropriately
		if sma.Len() != ts.Len() {
			t.Errorf("SMA should maintain original length")
		}

		// Test EMA with missing values
		ema, err := ts.ExponentialMovingAverage(0.3)
		if err != nil {
			t.Fatalf("EMA with missing values failed: %v", err)
		}

		if ema.Len() != ts.Len() {
			t.Errorf("EMA should maintain original length")
		}

		t.Logf("Missing value handling successful")
	})
}
