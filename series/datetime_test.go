package series

import (
	"testing"
	"time"
)

func TestDateTimeIndexExtended(t *testing.T) {
	t.Run("Create DateTimeIndex", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
		}

		index := NewDateTimeIndex(dates)

		if index.Len() != 3 {
			t.Errorf("Expected length 3, got %d", index.Len())
		}

		// Check first date
		firstDate := index.Get(0).(time.Time)
		if !firstDate.Equal(dates[0]) {
			t.Errorf("Expected %v at index 0, got %v", dates[0], firstDate)
		}
	})

	t.Run("DateTimeIndex Slice", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
		}

		index := NewDateTimeIndex(dates)
		sliced := index.Slice(1, 3)

		if sliced.Len() != 2 {
			t.Errorf("Expected length 2, got %d", sliced.Len())
		}

		// Check sliced dates
		if !sliced.Get(0).(time.Time).Equal(dates[1]) {
			t.Errorf("Expected %v at index 0, got %v", dates[1], sliced.Get(0))
		}
		if !sliced.Get(1).(time.Time).Equal(dates[2]) {
			t.Errorf("Expected %v at index 1, got %v", dates[2], sliced.Get(1))
		}
	})

	t.Run("DateTimeIndex Loc", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
		}

		index := NewDateTimeIndex(dates)

		// Find by exact date
		pos, found := index.Loc(dates[1])
		if !found {
			t.Error("Expected to find date")
		}
		if pos != 1 {
			t.Errorf("Expected position 1, got %d", pos)
		}

		// Find non-existent date
		nonExistent := time.Date(2023, 1, 10, 0, 0, 0, 0, time.UTC)
		_, found = index.Loc(nonExistent)
		if found {
			t.Error("Should not find non-existent date")
		}
	})
}

func TestTimeSeriesSeries(t *testing.T) {
	t.Run("Create Time Series", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
		}
		values := []float64{100.0, 200.0, 300.0}

		ts, err := NewTimeSeriesFromSlices(dates, values)
		if err != nil {
			t.Fatalf("Failed to create time series: %v", err)
		}

		if ts.Len() != 3 {
			t.Errorf("Expected length 3, got %d", ts.Len())
		}

		if ts.Name() != "TimeSeries" {
			t.Errorf("Expected default name 'TimeSeries', got %s", ts.Name())
		}

		// Check values
		for i, expected := range values {
			val := ts.ILocSingle(i).(float64)
			if val != expected {
				t.Errorf("Expected %v at index %d, got %v", expected, i, val)
			}
		}
	})

	t.Run("Time Series Loc Access", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
		}
		values := []float64{100.0, 200.0}

		ts, _ := NewTimeSeriesFromSlices(dates, values)

		// Access by date
		val, err := ts.Loc(dates[1])
		if err != nil {
			t.Fatalf("Failed to access by date: %v", err)
		}
		if val.(float64) != 200.0 {
			t.Errorf("Expected 200.0, got %v", val)
		}

		// Access non-existent date
		nonExistent := time.Date(2023, 1, 10, 0, 0, 0, 0, time.UTC)
		_, err = ts.Loc(nonExistent)
		if err == nil {
			t.Error("Expected error for non-existent date")
		}
	})

	t.Run("Time Series Slice by Date Range", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
		}
		values := []float64{100.0, 200.0, 300.0, 400.0}

		ts, _ := NewTimeSeriesFromSlices(dates, values)

		// Slice from Jan 2 to Jan 3
		start := time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC)
		end := time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC)

		sliced, err := ts.SliceByDateRange(start, end)
		if err != nil {
			t.Fatalf("Failed to slice by date range: %v", err)
		}

		if sliced.Len() != 2 {
			t.Errorf("Expected length 2, got %d", sliced.Len())
		}

		// Check values
		if sliced.ILocSingle(0).(float64) != 200.0 {
			t.Errorf("Expected 200.0 at index 0, got %v", sliced.ILocSingle(0))
		}
		if sliced.ILocSingle(1).(float64) != 300.0 {
			t.Errorf("Expected 300.0 at index 1, got %v", sliced.ILocSingle(1))
		}
	})
}

func TestDateRange(t *testing.T) {
	t.Run("Generate Daily Date Range", func(t *testing.T) {
		start := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
		end := time.Date(2023, 1, 5, 0, 0, 0, 0, time.UTC)

		dates, err := DateRange(start, end, "D")
		if err != nil {
			t.Fatalf("Failed to generate date range: %v", err)
		}

		if len(dates) != 5 {
			t.Errorf("Expected 5 dates, got %d", len(dates))
		}

		// Check first and last dates
		if !dates[0].Equal(start) {
			t.Errorf("Expected first date %v, got %v", start, dates[0])
		}

		expectedLast := time.Date(2023, 1, 5, 0, 0, 0, 0, time.UTC)
		if !dates[4].Equal(expectedLast) {
			t.Errorf("Expected last date %v, got %v", expectedLast, dates[4])
		}
	})

	t.Run("Generate Hourly Date Range", func(t *testing.T) {
		start := time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC)
		end := time.Date(2023, 1, 1, 13, 0, 0, 0, time.UTC)

		dates, err := DateRange(start, end, "H")
		if err != nil {
			t.Fatalf("Failed to generate hourly range: %v", err)
		}

		if len(dates) != 4 {
			t.Errorf("Expected 4 dates, got %d", len(dates))
		}

		// Check progression
		for i, date := range dates {
			expectedHour := 10 + i
			if date.Hour() != expectedHour {
				t.Errorf("Expected hour %d at index %d, got %d", expectedHour, i, date.Hour())
			}
		}
	})

	t.Run("Generate Monthly Date Range", func(t *testing.T) {
		start := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
		end := time.Date(2023, 4, 1, 0, 0, 0, 0, time.UTC)

		dates, err := DateRange(start, end, "M")
		if err != nil {
			t.Fatalf("Failed to generate monthly range: %v", err)
		}

		if len(dates) != 4 {
			t.Errorf("Expected 4 dates, got %d", len(dates))
		}

		// Check months
		expectedMonths := []time.Month{time.January, time.February, time.March, time.April}
		for i, date := range dates {
			if date.Month() != expectedMonths[i] {
				t.Errorf("Expected month %v at index %d, got %v", expectedMonths[i], i, date.Month())
			}
		}
	})
}

func TestTimeSeriesOperations(t *testing.T) {
	t.Run("Time Series Resample Daily to Weekly", func(t *testing.T) {
		// Create daily time series for one week
		dates := make([]time.Time, 7)
		values := make([]float64, 7)
		for i := 0; i < 7; i++ {
			dates[i] = time.Date(2023, 1, 1+i, 0, 0, 0, 0, time.UTC)
			values[i] = float64(i + 1) // values 1, 2, 3, 4, 5, 6, 7
		}

		ts, _ := NewTimeSeriesFromSlices(dates, values)

		// Resample to weekly with sum aggregation
		resampled, err := ts.Resample("W", "sum")
		if err != nil {
			t.Fatalf("Failed to resample: %v", err)
		}

		if resampled.Len() != 1 {
			t.Errorf("Expected length 1 after weekly resampling, got %d", resampled.Len())
		}

		// Sum of 1+2+3+4+5+6+7 = 28
		expectedSum := 28.0
		if resampled.ILocSingle(0).(float64) != expectedSum {
			t.Errorf("Expected sum %v, got %v", expectedSum, resampled.ILocSingle(0))
		}
	})

	t.Run("Time Series Shift", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
		}
		values := []float64{10.0, 20.0, 30.0}

		ts, _ := NewTimeSeriesFromSlices(dates, values)

		// Shift forward by 1 period
		shifted, err := ts.Shift(1)
		if err != nil {
			t.Fatalf("Failed to shift: %v", err)
		}

		if shifted.Len() != 3 {
			t.Errorf("Expected length 3, got %d", shifted.Len())
		}

		// First value should be nil after shift
		firstVal := shifted.ILocSingle(0)
		if firstVal != nil {
			t.Errorf("Expected nil at index 0 after shift, got %v", firstVal)
		}

		// Second value should be original first value
		if shifted.ILocSingle(1).(float64) != 10.0 {
			t.Errorf("Expected 10.0 at index 1, got %v", shifted.ILocSingle(1))
		}
	})

	t.Run("Time Series Rolling Window", func(t *testing.T) {
		dates := []time.Time{
			time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
			time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
		}
		values := []float64{1.0, 2.0, 3.0, 4.0}

		ts, _ := NewTimeSeriesFromSlices(dates, values)

		// Rolling mean with window size 2
		rolled, err := ts.Rolling(2, "mean")
		if err != nil {
			t.Fatalf("Failed to calculate rolling mean: %v", err)
		}

		if rolled.Len() != 4 {
			t.Errorf("Expected length 4, got %d", rolled.Len())
		}

		// First value should be nil (insufficient data)
		if rolled.ILocSingle(0) != nil {
			t.Errorf("Expected nil at index 0, got %v", rolled.ILocSingle(0))
		}

		// Second value should be mean of 1,2 = 1.5
		if rolled.ILocSingle(1).(float64) != 1.5 {
			t.Errorf("Expected 1.5 at index 1, got %v", rolled.ILocSingle(1))
		}

		// Third value should be mean of 2,3 = 2.5
		if rolled.ILocSingle(2).(float64) != 2.5 {
			t.Errorf("Expected 2.5 at index 2, got %v", rolled.ILocSingle(2))
		}
	})
}

func TestTimeSeriesErrors(t *testing.T) {
	t.Run("Mismatched Lengths", func(t *testing.T) {
		dates := []time.Time{time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)}
		values := []float64{1.0, 2.0} // Different length

		_, err := NewTimeSeriesFromSlices(dates, values)
		if err == nil {
			t.Error("Expected error for mismatched lengths")
		}
	})

	t.Run("Invalid Frequency", func(t *testing.T) {
		start := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
		end := time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC)

		_, err := DateRange(start, end, "X") // Invalid frequency
		if err == nil {
			t.Error("Expected error for invalid frequency")
		}
	})

	t.Run("Invalid Resample Aggregation", func(t *testing.T) {
		dates := []time.Time{time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)}
		values := []float64{1.0}

		ts, _ := NewTimeSeriesFromSlices(dates, values)

		_, err := ts.Resample("D", "invalid")
		if err == nil {
			t.Error("Expected error for invalid aggregation")
		}
	})
}
