package series

import (
	"fmt"
	"time"

	"github.com/julianshen/gonp/array"
)

// NewTimeSeriesFromSlices creates a time series from separate date and value slices
func NewTimeSeriesFromSlices(dates []time.Time, values interface{}) (*Series, error) {
	if len(dates) == 0 {
		return nil, fmt.Errorf("dates cannot be empty")
	}

	// Create array from values
	arr, err := array.FromSlice(values)
	if err != nil {
		return nil, fmt.Errorf("failed to create array from values: %v", err)
	}

	if arr.Size() != len(dates) {
		return nil, fmt.Errorf("dates length (%d) does not match values length (%d)", len(dates), arr.Size())
	}

	// Create DateTimeIndex
	index := NewDateTimeIndex(dates)

	return NewSeries(arr, index, "TimeSeries")
}

// SliceByDateRange returns a slice of the time series between start and end dates (inclusive)
func (s *Series) SliceByDateRange(start, end time.Time) (*Series, error) {
	dtIndex, ok := s.index.(*DateTimeIndex)
	if !ok {
		return nil, fmt.Errorf("series must have DateTimeIndex for date range slicing")
	}

	startIdx := -1
	endIdx := -1

	// Find start and end positions
	for i := 0; i < dtIndex.Len(); i++ {
		date := dtIndex.Get(i).(time.Time)
		if startIdx == -1 && (date.Equal(start) || date.After(start)) {
			startIdx = i
		}
		if date.Equal(end) || date.Before(end) {
			endIdx = i
		}
	}

	if startIdx == -1 {
		return nil, fmt.Errorf("start date %v not found in series", start)
	}
	if endIdx == -1 || endIdx < startIdx {
		return nil, fmt.Errorf("end date %v not found or before start date", end)
	}

	// Slice the series
	return s.ILocSlice(startIdx, endIdx+1)
}

// Shift shifts the time series data by the specified number of periods
// Positive periods shift forward (introduce NaN at the beginning)
// Negative periods shift backward (introduce NaN at the end)
func (s *Series) Shift(periods int) (*Series, error) {
	if periods == 0 {
		return s.Copy(), nil
	}

	size := s.Len()
	newData := make([]interface{}, size)

	if periods > 0 {
		// Shift forward: fill first 'periods' positions with nil
		for i := 0; i < periods; i++ {
			newData[i] = nil
		}
		// Copy remaining data
		for i := periods; i < size; i++ {
			newData[i] = s.ILocSingle(i - periods)
		}
	} else {
		// Shift backward: copy data, then fill end with nil
		absPeroids := -periods
		for i := 0; i < size-absPeroids; i++ {
			newData[i] = s.ILocSingle(i + absPeroids)
		}
		// Fill end with nil
		for i := size - absPeroids; i < size; i++ {
			newData[i] = nil
		}
	}

	// Create new series with same index
	newArr, err := array.FromSlice(newData)
	if err != nil {
		return nil, fmt.Errorf("failed to create shifted array: %v", err)
	}

	return NewSeries(newArr, s.index.Copy(), s.name)
}

// Resample resamples the time series to a different frequency with aggregation
func (s *Series) Resample(freq string, aggFunc string) (*Series, error) {
	dtIndex, ok := s.index.(*DateTimeIndex)
	if !ok {
		return nil, fmt.Errorf("series must have DateTimeIndex for resampling")
	}

	if dtIndex.Len() == 0 {
		return s.Copy(), nil
	}

	// Get time bounds
	startTime := dtIndex.Get(0).(time.Time)
	endTime := dtIndex.Get(dtIndex.Len() - 1).(time.Time)

	// Generate new time grid based on frequency
	newDates, err := generateResampleDates(startTime, endTime, freq)
	if err != nil {
		return nil, fmt.Errorf("failed to generate resample dates: %v", err)
	}

	if len(newDates) == 0 {
		return nil, fmt.Errorf("no dates generated for resampling")
	}

	// Aggregate data into new periods
	newValues := make([]float64, len(newDates)-1) // Periods between dates

	for i := 0; i < len(newDates)-1; i++ {
		periodStart := newDates[i]
		periodEnd := newDates[i+1]

		// Find all data points in this period
		var periodData []float64
		for j := 0; j < dtIndex.Len(); j++ {
			date := dtIndex.Get(j).(time.Time)
			if (date.Equal(periodStart) || date.After(periodStart)) && date.Before(periodEnd) {
				val := s.ILocSingle(j)
				if val != nil {
					if floatVal, ok := val.(float64); ok {
						periodData = append(periodData, floatVal)
					}
				}
			}
		}

		// Apply aggregation function
		if len(periodData) == 0 {
			newValues[i] = 0 // or NaN
		} else {
			switch aggFunc {
			case "sum":
				sum := 0.0
				for _, v := range periodData {
					sum += v
				}
				newValues[i] = sum
			case "mean":
				sum := 0.0
				for _, v := range periodData {
					sum += v
				}
				newValues[i] = sum / float64(len(periodData))
			case "count":
				newValues[i] = float64(len(periodData))
			default:
				return nil, fmt.Errorf("unsupported aggregation function: %s", aggFunc)
			}
		}
	}

	// Use period start times as new index
	newIndex := NewDateTimeIndex(newDates[:len(newDates)-1])
	newArr, err := array.FromSlice(newValues)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampled array: %v", err)
	}

	return NewSeries(newArr, newIndex, s.name)
}

// Rolling calculates rolling window statistics
func (s *Series) Rolling(window int, aggFunc string) (*Series, error) {
	if window <= 0 {
		return nil, fmt.Errorf("window size must be positive, got %d", window)
	}

	size := s.Len()
	newData := make([]interface{}, size)

	for i := 0; i < size; i++ {
		if i < window-1 {
			// Not enough data for window
			newData[i] = nil
			continue
		}

		// Collect window data
		var windowData []float64
		for j := i - window + 1; j <= i; j++ {
			val := s.ILocSingle(j)
			if val != nil {
				if floatVal, ok := val.(float64); ok {
					windowData = append(windowData, floatVal)
				}
			}
		}

		if len(windowData) == 0 {
			newData[i] = nil
			continue
		}

		// Apply aggregation
		switch aggFunc {
		case "sum":
			sum := 0.0
			for _, v := range windowData {
				sum += v
			}
			newData[i] = sum
		case "mean":
			sum := 0.0
			for _, v := range windowData {
				sum += v
			}
			newData[i] = sum / float64(len(windowData))
		case "min":
			min := windowData[0]
			for _, v := range windowData[1:] {
				if v < min {
					min = v
				}
			}
			newData[i] = min
		case "max":
			max := windowData[0]
			for _, v := range windowData[1:] {
				if v > max {
					max = v
				}
			}
			newData[i] = max
		default:
			return nil, fmt.Errorf("unsupported rolling aggregation function: %s", aggFunc)
		}
	}

	newArr, err := array.FromSlice(newData)
	if err != nil {
		return nil, fmt.Errorf("failed to create rolling array: %v", err)
	}

	return NewSeries(newArr, s.index.Copy(), s.name)
}

// DateRange generates a range of dates between start and end with given frequency
func DateRange(start, end time.Time, freq string) ([]time.Time, error) {
	var dates []time.Time
	current := start

	for !current.After(end) {
		dates = append(dates, current)

		switch freq {
		case "D": // Daily
			current = current.AddDate(0, 0, 1)
		case "H": // Hourly
			current = current.Add(time.Hour)
		case "M": // Monthly
			current = current.AddDate(0, 1, 0)
		case "Y": // Yearly
			current = current.AddDate(1, 0, 0)
		case "W": // Weekly
			current = current.AddDate(0, 0, 7)
		default:
			return nil, fmt.Errorf("unsupported frequency: %s", freq)
		}
	}

	return dates, nil
}

// Helper function to generate dates for resampling
func generateResampleDates(start, end time.Time, freq string) ([]time.Time, error) {
	switch freq {
	case "W": // Weekly - align to week boundaries
		// Find start of week (Sunday)
		weekStart := start
		for weekStart.Weekday() != time.Sunday {
			weekStart = weekStart.AddDate(0, 0, -1)
		}

		var dates []time.Time
		current := weekStart
		for !current.After(end.AddDate(0, 0, 7)) { // Include one period past end
			dates = append(dates, current)
			current = current.AddDate(0, 0, 7)
		}
		return dates, nil

	case "M": // Monthly
		// Find start of month
		monthStart := time.Date(start.Year(), start.Month(), 1, 0, 0, 0, 0, start.Location())

		var dates []time.Time
		current := monthStart
		for current.Before(end.AddDate(0, 1, 0)) {
			dates = append(dates, current)
			current = current.AddDate(0, 1, 0)
		}
		return dates, nil

	case "D": // Daily
		return DateRange(start, end.AddDate(0, 0, 1), "D")

	default:
		return nil, fmt.Errorf("unsupported resample frequency: %s", freq)
	}
}
