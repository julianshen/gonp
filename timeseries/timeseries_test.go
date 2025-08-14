package timeseries

import (
	"math"
	"testing"
	"time"

	"github.com/julianshen/gonp/array"
)

// TestTimeSeries tests the basic TimeSeries data structure using TDD
func TestTimeSeries(t *testing.T) {
	t.Run("Basic TimeSeries creation from array with time index", func(t *testing.T) {
		// Red phase: Write a failing test first
		// Create time series with daily data
		values := []float64{10.0, 12.0, 11.0, 13.0, 15.0}
		dates := []time.Time{
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 4, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 5, 0, 0, 0, 0, time.UTC),
		}

		// Should be able to create TimeSeries from values and dates
		ts, err := NewTimeSeries(values, dates, "daily_data")
		if err != nil {
			t.Fatalf("Failed to create TimeSeries: %v", err)
		}

		// Check basic properties
		if ts.Len() != 5 {
			t.Errorf("Expected length 5, got %d", ts.Len())
		}

		if ts.Name() != "daily_data" {
			t.Errorf("Expected name 'daily_data', got %s", ts.Name())
		}

		// Check values and index access
		if ts.At(0) != 10.0 {
			t.Errorf("Expected value 10.0 at index 0, got %f", ts.At(0))
		}

		expectedDate := time.Date(2024, 1, 3, 0, 0, 0, 0, time.UTC)
		if !ts.TimeAt(2).Equal(expectedDate) {
			t.Errorf("Expected date %v at index 2, got %v", expectedDate, ts.TimeAt(2))
		}

		t.Logf("TimeSeries created successfully with %d observations", ts.Len())
	})

	t.Run("TimeSeries creation from array with automatic time index", func(t *testing.T) {
		// Test creating TimeSeries with automatic sequential time index
		dataArray, _ := array.FromSlice([]float64{1.0, 2.0, 3.0, 4.0})

		ts, err := FromArray(dataArray, nil, "auto_index")
		if err != nil {
			t.Fatalf("Failed to create TimeSeries from array: %v", err)
		}

		// Should have sequential integer index converted to time
		if ts.Len() != 4 {
			t.Errorf("Expected length 4, got %d", ts.Len())
		}

		// Check that automatic indexing works
		if ts.At(2) != 3.0 {
			t.Errorf("Expected value 3.0 at index 2, got %f", ts.At(2))
		}

		t.Logf("Auto-indexed TimeSeries created with %d periods", ts.Len())
	})

	t.Run("TimeSeries slicing and subsetting", func(t *testing.T) {
		// Test time-based slicing operations
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i) // Daily data
		}

		ts, _ := NewTimeSeries(values, dates, "test_series")

		// Slice by index
		subset, err := ts.Slice(1, 4) // Should get indices 1, 2, 3
		if err != nil {
			t.Fatalf("Failed to slice TimeSeries: %v", err)
		}

		if subset.Len() != 3 {
			t.Errorf("Expected sliced length 3, got %d", subset.Len())
		}

		if subset.At(0) != 2.0 {
			t.Errorf("Expected first value 2.0 in slice, got %f", subset.At(0))
		}

		// Time-based slicing
		startDate := time.Date(2024, 1, 3, 0, 0, 0, 0, time.UTC)
		endDate := time.Date(2024, 1, 5, 0, 0, 0, 0, time.UTC)

		timeSlice, err := ts.SliceByTime(startDate, endDate)
		if err != nil {
			t.Fatalf("Failed to slice TimeSeries by time: %v", err)
		}

		if timeSlice.Len() != 3 { // Should include start and end dates
			t.Errorf("Expected time sliced length 3, got %d", timeSlice.Len())
		}

		t.Logf("TimeSeries slicing successful: index slice %d, time slice %d", subset.Len(), timeSlice.Len())
	})

	t.Run("TimeSeries arithmetic operations", func(t *testing.T) {
		// Test basic arithmetic operations on time series
		values1 := []float64{1.0, 2.0, 3.0, 4.0}
		values2 := []float64{2.0, 4.0, 6.0, 8.0}
		dates := []time.Time{
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 4, 0, 0, 0, 0, time.UTC),
		}

		ts1, _ := NewTimeSeries(values1, dates, "series1")
		ts2, _ := NewTimeSeries(values2, dates, "series2")

		// Addition
		sum, err := ts1.Add(ts2)
		if err != nil {
			t.Fatalf("Failed to add TimeSeries: %v", err)
		}

		expectedSum := []float64{3.0, 6.0, 9.0, 12.0}
		for i := 0; i < sum.Len(); i++ {
			if sum.At(i) != expectedSum[i] {
				t.Errorf("Expected sum[%d] = %f, got %f", i, expectedSum[i], sum.At(i))
			}
		}

		// Scalar multiplication
		scaled, err := ts1.Multiply(2.0)
		if err != nil {
			t.Fatalf("Failed to multiply TimeSeries by scalar: %v", err)
		}

		if scaled.At(1) != 4.0 {
			t.Errorf("Expected scaled[1] = 4.0, got %f", scaled.At(1))
		}

		t.Logf("TimeSeries arithmetic operations successful")
	})

	t.Run("TimeSeries missing value handling", func(t *testing.T) {
		// Test handling of missing values (NaN) in time series
		values := []float64{1.0, math.NaN(), 3.0, math.NaN(), 5.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "with_missing")

		// Count missing values
		missingCount := ts.CountMissing()
		if missingCount != 2 {
			t.Errorf("Expected 2 missing values, got %d", missingCount)
		}

		// Drop missing values
		cleaned, err := ts.DropNA()
		if err != nil {
			t.Fatalf("Failed to drop missing values: %v", err)
		}

		if cleaned.Len() != 3 {
			t.Errorf("Expected 3 values after dropping NaN, got %d", cleaned.Len())
		}

		// Forward fill missing values
		filled, err := ts.FillNA("forward")
		if err != nil {
			t.Fatalf("Failed to forward fill: %v", err)
		}

		if filled.At(1) != 1.0 { // Should be forward filled from index 0
			t.Errorf("Expected forward filled value 1.0, got %f", filled.At(1))
		}

		t.Logf("Missing value handling: %d missing, %d after cleanup, forward fill successful",
			missingCount, cleaned.Len())
	})

	t.Run("TimeSeries parameter validation", func(t *testing.T) {
		// Test error conditions and parameter validation

		// Mismatched lengths
		values := []float64{1.0, 2.0, 3.0}
		dates := []time.Time{time.Now(), time.Now()}

		_, err := NewTimeSeries(values, dates, "mismatched")
		if err == nil {
			t.Error("Expected error for mismatched lengths")
		}

		// Empty data
		_, err = NewTimeSeries([]float64{}, []time.Time{}, "empty")
		if err == nil {
			t.Error("Expected error for empty data")
		}

		// Nil values
		_, err = NewTimeSeries(nil, nil, "nil_data")
		if err == nil {
			t.Error("Expected error for nil data")
		}

		// Invalid slice bounds
		validValues := []float64{1.0, 2.0, 3.0}
		validDates := []time.Time{time.Now(), time.Now().Add(time.Hour), time.Now().Add(2 * time.Hour)}
		ts, _ := NewTimeSeries(validValues, validDates, "valid")

		_, err = ts.Slice(-1, 2)
		if err == nil {
			t.Error("Expected error for negative slice start")
		}

		_, err = ts.Slice(0, 10)
		if err == nil {
			t.Error("Expected error for slice end beyond bounds")
		}

		t.Logf("Parameter validation tests passed")
	})
}

// TestTimeSeriesFrequency tests frequency detection and resampling
func TestTimeSeriesFrequency(t *testing.T) {
	t.Run("Frequency detection for regular time series", func(t *testing.T) {
		// Test automatic frequency detection
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		dates := []time.Time{
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC), // Daily
			time.Date(2024, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 4, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 5, 0, 0, 0, 0, time.UTC),
		}

		ts, _ := NewTimeSeries(values, dates, "daily")

		freq, err := ts.InferFrequency()
		if err != nil {
			t.Fatalf("Failed to infer frequency: %v", err)
		}

		if freq != "D" { // Daily frequency
			t.Errorf("Expected frequency 'D', got %s", freq)
		}

		// Test monthly data
		monthlyDates := []time.Time{
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC),
		}
		monthlyValues := []float64{10.0, 20.0, 30.0}

		monthlyTS, _ := NewTimeSeries(monthlyValues, monthlyDates, "monthly")
		monthlyFreq, _ := monthlyTS.InferFrequency()

		if monthlyFreq != "M" { // Monthly frequency
			t.Errorf("Expected monthly frequency 'M', got %s", monthlyFreq)
		}

		t.Logf("Frequency detection: daily='%s', monthly='%s'", freq, monthlyFreq)
	})

	t.Run("Time series resampling operations", func(t *testing.T) {
		// Test resampling from daily to weekly data
		values := make([]float64, 14) // 2 weeks of daily data
		dates := make([]time.Time, 14)
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range values {
			values[i] = float64(i + 1)
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "daily_to_weekly")

		// Resample to weekly using mean aggregation
		weekly, err := ts.Resample("W", "mean")
		if err != nil {
			t.Fatalf("Failed to resample to weekly: %v", err)
		}

		if weekly.Len() != 2 { // Should have 2 weeks
			t.Errorf("Expected 2 weeks after resampling, got %d", weekly.Len())
		}

		// First week mean should be (1+2+3+4+5+6+7)/7 = 4.0
		expectedFirstWeekMean := 4.0
		if math.Abs(weekly.At(0)-expectedFirstWeekMean) > 1e-10 {
			t.Errorf("Expected first week mean %.1f, got %.6f", expectedFirstWeekMean, weekly.At(0))
		}

		// Test other aggregation methods
		weeklySum, _ := ts.Resample("W", "sum")
		weeklyMax, _ := ts.Resample("W", "max")

		if weeklySum.At(0) != 28.0 { // Sum of 1+2+...+7 = 28
			t.Errorf("Expected first week sum 28.0, got %.1f", weeklySum.At(0))
		}

		if weeklyMax.At(0) != 7.0 { // Max of first week
			t.Errorf("Expected first week max 7.0, got %.1f", weeklyMax.At(0))
		}

		t.Logf("Resampling successful: daily->weekly mean=%.1f, sum=%.1f, max=%.1f",
			weekly.At(0), weeklySum.At(0), weeklyMax.At(0))
	})
}

// TestTimeSeriesStats tests basic statistical operations on time series
func TestTimeSeriesStats(t *testing.T) {
	t.Run("Basic descriptive statistics", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "stats_test")

		// Mean
		mean := ts.Mean()
		expectedMean := 5.5
		if math.Abs(mean-expectedMean) > 1e-10 {
			t.Errorf("Expected mean %.1f, got %.6f", expectedMean, mean)
		}

		// Standard deviation
		std := ts.Std()
		if std <= 0 {
			t.Errorf("Expected positive standard deviation, got %.6f", std)
		}

		// Min and Max
		min := ts.Min()
		max := ts.Max()
		if min != 1.0 || max != 10.0 {
			t.Errorf("Expected min=1.0, max=10.0, got min=%.1f, max=%.1f", min, max)
		}

		// Quantiles
		median, _ := ts.Quantile(0.5)
		q25, _ := ts.Quantile(0.25)
		q75, _ := ts.Quantile(0.75)

		if median != 5.5 {
			t.Errorf("Expected median 5.5, got %.6f", median)
		}

		t.Logf("Stats: mean=%.1f, std=%.3f, median=%.1f, q25=%.1f, q75=%.1f",
			mean, std, median, q25, q75)
	})

	t.Run("Rolling window statistics", func(t *testing.T) {
		// Test rolling mean, std, min, max
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
		dates := make([]time.Time, len(values))
		baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

		for i := range dates {
			dates[i] = baseDate.AddDate(0, 0, i)
		}

		ts, _ := NewTimeSeries(values, dates, "rolling_test")

		// 3-period rolling mean
		rollingMean, err := ts.RollingMean(3)
		if err != nil {
			t.Fatalf("Failed to compute rolling mean: %v", err)
		}

		// First two values should be NaN, then start with (1+2+3)/3 = 2.0
		if !math.IsNaN(rollingMean.At(0)) || !math.IsNaN(rollingMean.At(1)) {
			t.Error("Expected first two rolling mean values to be NaN")
		}

		if rollingMean.At(2) != 2.0 {
			t.Errorf("Expected rolling mean[2] = 2.0, got %.6f", rollingMean.At(2))
		}

		// Rolling standard deviation
		rollingStd, _ := ts.RollingStd(3)
		if math.IsNaN(rollingStd.At(3)) {
			t.Error("Expected rolling std[3] to not be NaN")
		}

		t.Logf("Rolling stats: mean[2]=%.1f, mean[3]=%.1f, std[3]=%.3f",
			rollingMean.At(2), rollingMean.At(3), rollingStd.At(3))
	})
}
