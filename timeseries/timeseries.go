package timeseries

import (
	"errors"
	"math"
	"sort"
	"time"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// TimeSeries represents a time-indexed sequence of numerical data
// following pandas Series-like functionality with temporal indexing
type TimeSeries struct {
	data  *array.Array // Underlying numerical data
	index []time.Time  // Time index for each data point
	name  string       // Optional name for the series
	freq  string       // Inferred or specified frequency (D, W, M, etc.)
}

// NewTimeSeries creates a new TimeSeries from values and time index
func NewTimeSeries(values []float64, timeIndex []time.Time, name string) (*TimeSeries, error) {
	if values == nil || timeIndex == nil {
		return nil, errors.New("values and timeIndex cannot be nil")
	}

	if len(values) == 0 {
		return nil, errors.New("cannot create TimeSeries from empty data")
	}

	if len(values) != len(timeIndex) {
		return nil, errors.New("values and timeIndex must have the same length")
	}

	// Create array from values
	data, err := array.FromSlice(values)
	if err != nil {
		return nil, err
	}

	// Copy time index
	indexCopy := make([]time.Time, len(timeIndex))
	copy(indexCopy, timeIndex)

	return &TimeSeries{
		data:  data,
		index: indexCopy,
		name:  name,
		freq:  "", // Will be inferred later if needed
	}, nil
}

// FromArray creates a TimeSeries from an array with optional time index
func FromArray(data *array.Array, timeIndex []time.Time, name string) (*TimeSeries, error) {
	if data == nil {
		return nil, errors.New("data array cannot be nil")
	}

	if data.Ndim() != 1 {
		return nil, errors.New("data array must be 1-dimensional")
	}

	length := data.Size()
	if length == 0 {
		return nil, errors.New("cannot create TimeSeries from empty array")
	}

	// Create time index if not provided
	var index []time.Time
	if timeIndex == nil {
		// Create sequential time index (daily starting from 2024-01-01)
		baseTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
		index = make([]time.Time, length)
		for i := 0; i < length; i++ {
			index[i] = baseTime.AddDate(0, 0, i)
		}
	} else {
		if len(timeIndex) != length {
			return nil, errors.New("timeIndex length must match data array length")
		}
		index = make([]time.Time, len(timeIndex))
		copy(index, timeIndex)
	}

	// Copy the array data
	dataCopy := array.Empty(internal.Shape{length}, data.DType())
	for i := 0; i < length; i++ {
		dataCopy.Set(data.At(i), i)
	}

	return &TimeSeries{
		data:  dataCopy,
		index: index,
		name:  name,
		freq:  "",
	}, nil
}

// Basic access methods
func (ts *TimeSeries) Len() int {
	if ts.data == nil {
		return 0
	}
	return ts.data.Size()
}

func (ts *TimeSeries) Name() string {
	return ts.name
}

func (ts *TimeSeries) At(index int) float64 {
	if index < 0 || index >= ts.Len() {
		return math.NaN()
	}
	value := ts.data.At(index)
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return math.NaN()
	}
}

func (ts *TimeSeries) TimeAt(index int) time.Time {
	if index < 0 || index >= len(ts.index) {
		return time.Time{}
	}
	return ts.index[index]
}

// Slicing operations
func (ts *TimeSeries) Slice(start, end int) (*TimeSeries, error) {
	if start < 0 {
		return nil, errors.New("start index cannot be negative")
	}
	if end > ts.Len() {
		return nil, errors.New("end index cannot exceed series length")
	}
	if start >= end {
		return nil, errors.New("start index must be less than end index")
	}

	length := end - start
	values := make([]float64, length)
	times := make([]time.Time, length)

	for i := 0; i < length; i++ {
		values[i] = ts.At(start + i)
		times[i] = ts.TimeAt(start + i)
	}

	return NewTimeSeries(values, times, ts.name)
}

func (ts *TimeSeries) SliceByTime(startTime, endTime time.Time) (*TimeSeries, error) {
	if startTime.After(endTime) {
		return nil, errors.New("start time must be before or equal to end time")
	}

	values := make([]float64, 0)
	times := make([]time.Time, 0)

	for i := 0; i < ts.Len(); i++ {
		t := ts.TimeAt(i)
		// Include if time is within range (inclusive)
		if (t.Equal(startTime) || t.After(startTime)) && (t.Equal(endTime) || t.Before(endTime)) {
			values = append(values, ts.At(i))
			times = append(times, t)
		}
	}

	if len(values) == 0 {
		return NewTimeSeries([]float64{}, []time.Time{}, ts.name)
	}

	return NewTimeSeries(values, times, ts.name)
}

// Arithmetic operations
func (ts *TimeSeries) Add(other *TimeSeries) (*TimeSeries, error) {
	if other == nil {
		return nil, errors.New("other TimeSeries cannot be nil")
	}
	if ts.Len() != other.Len() {
		return nil, errors.New("TimeSeries must have the same length for addition")
	}

	// For simplicity, assume same time index (more advanced alignment could be added later)
	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		val1 := ts.At(i)
		val2 := other.At(i)

		if math.IsNaN(val1) || math.IsNaN(val2) {
			values[i] = math.NaN()
		} else {
			values[i] = val1 + val2
		}
	}

	return NewTimeSeries(values, ts.index, ts.name)
}

func (ts *TimeSeries) Multiply(scalar float64) (*TimeSeries, error) {
	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		val := ts.At(i)
		if math.IsNaN(val) {
			values[i] = math.NaN()
		} else {
			values[i] = val * scalar
		}
	}

	return NewTimeSeries(values, ts.index, ts.name)
}

// Missing value handling
func (ts *TimeSeries) CountMissing() int {
	count := 0
	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if math.IsNaN(value) {
			count++
		}
	}
	return count
}

func (ts *TimeSeries) DropNA() (*TimeSeries, error) {
	validValues := make([]float64, 0)
	validTimes := make([]time.Time, 0)

	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			validValues = append(validValues, value)
			validTimes = append(validTimes, ts.TimeAt(i))
		}
	}

	return NewTimeSeries(validValues, validTimes, ts.name)
}

func (ts *TimeSeries) FillNA(method string) (*TimeSeries, error) {
	values := make([]float64, ts.Len())

	switch method {
	case "forward":
		lastValid := math.NaN()
		for i := 0; i < ts.Len(); i++ {
			value := ts.At(i)
			if !math.IsNaN(value) {
				lastValid = value
				values[i] = value
			} else if !math.IsNaN(lastValid) {
				values[i] = lastValid
			} else {
				values[i] = value // Keep NaN if no valid value found yet
			}
		}
	case "backward":
		// First pass: find the values
		for i := 0; i < ts.Len(); i++ {
			values[i] = ts.At(i)
		}
		// Second pass: backward fill
		for i := ts.Len() - 2; i >= 0; i-- {
			if math.IsNaN(values[i]) && !math.IsNaN(values[i+1]) {
				values[i] = values[i+1]
			}
		}
	default:
		return nil, errors.New("unsupported fill method: " + method)
	}

	return NewTimeSeries(values, ts.index, ts.name)
}

// Frequency detection and resampling
func (ts *TimeSeries) InferFrequency() (string, error) {
	if ts.Len() < 2 {
		return "", errors.New("need at least 2 observations to infer frequency")
	}

	// Calculate time differences
	diff := ts.TimeAt(1).Sub(ts.TimeAt(0))

	// Check common frequencies
	switch {
	case diff.Hours() >= 23 && diff.Hours() <= 25: // Daily (24 ± 1 hour)
		return "D", nil
	case diff.Hours() >= 167 && diff.Hours() <= 169: // Weekly (168 ± 1 hour)
		return "W", nil
	case diff.Hours() >= 719 && diff.Hours() <= 745: // Monthly (approx 30-31 days)
		return "M", nil
	case diff.Hours() >= 8759 && diff.Hours() <= 8785: // Yearly (365 ± 1 day)
		return "Y", nil
	case diff.Hours() >= 0.9 && diff.Hours() <= 1.1: // Hourly
		return "H", nil
	default:
		return "IRREGULAR", nil
	}
}

func (ts *TimeSeries) Resample(freq, aggMethod string) (*TimeSeries, error) {
	if freq != "W" {
		return nil, errors.New("only weekly resampling ('W') is implemented for now")
	}

	if ts.Len() == 0 {
		return NewTimeSeries([]float64{}, []time.Time{}, ts.name)
	}

	// Simple weekly grouping - group by weeks starting from first date
	startDate := ts.TimeAt(0)
	groups := make(map[int][]float64)
	groupTimes := make(map[int]time.Time)

	for i := 0; i < ts.Len(); i++ {
		t := ts.TimeAt(i)
		// Calculate week number from start
		daysDiff := int(t.Sub(startDate).Hours() / 24)
		weekNum := daysDiff / 7

		if _, exists := groups[weekNum]; !exists {
			groups[weekNum] = make([]float64, 0)
			// Use the start of the week as group time
			groupTimes[weekNum] = startDate.AddDate(0, 0, weekNum*7)
		}

		val := ts.At(i)
		if !math.IsNaN(val) {
			groups[weekNum] = append(groups[weekNum], val)
		}
	}

	// Aggregate each group
	resultValues := make([]float64, 0)
	resultTimes := make([]time.Time, 0)

	// Sort week numbers to maintain order
	weekNums := make([]int, 0, len(groups))
	for weekNum := range groups {
		weekNums = append(weekNums, weekNum)
	}
	sort.Ints(weekNums)

	for _, weekNum := range weekNums {
		values := groups[weekNum]
		if len(values) == 0 {
			continue
		}

		var aggValue float64
		switch aggMethod {
		case "mean":
			sum := 0.0
			for _, v := range values {
				sum += v
			}
			aggValue = sum / float64(len(values))
		case "sum":
			sum := 0.0
			for _, v := range values {
				sum += v
			}
			aggValue = sum
		case "max":
			aggValue = values[0]
			for _, v := range values {
				if v > aggValue {
					aggValue = v
				}
			}
		case "min":
			aggValue = values[0]
			for _, v := range values {
				if v < aggValue {
					aggValue = v
				}
			}
		default:
			return nil, errors.New("unsupported aggregation method: " + aggMethod)
		}

		resultValues = append(resultValues, aggValue)
		resultTimes = append(resultTimes, groupTimes[weekNum])
	}

	return NewTimeSeries(resultValues, resultTimes, ts.name+"_resampled")
}

// Basic statistical operations
func (ts *TimeSeries) Mean() float64 {
	sum := 0.0
	count := 0
	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			sum += value
			count++
		}
	}
	if count == 0 {
		return math.NaN()
	}
	return sum / float64(count)
}

func (ts *TimeSeries) Std() float64 {
	mean := ts.Mean()
	if math.IsNaN(mean) {
		return math.NaN()
	}

	sumSquares := 0.0
	count := 0
	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			diff := value - mean
			sumSquares += diff * diff
			count++
		}
	}
	if count <= 1 {
		return math.NaN()
	}
	return math.Sqrt(sumSquares / float64(count-1)) // Sample standard deviation
}

// Helper methods for decomposition

// copy creates a deep copy of the TimeSeries
func (ts *TimeSeries) copy() *TimeSeries {
	if ts == nil {
		return nil
	}

	values := make([]float64, ts.Len())
	times := make([]time.Time, ts.Len())

	for i := 0; i < ts.Len(); i++ {
		values[i] = ts.At(i)
		times[i] = ts.TimeAt(i)
	}

	newTS, _ := NewTimeSeries(values, times, ts.name)
	return newTS
}

// subtractSeries subtracts another time series (element-wise)
func (ts *TimeSeries) subtractSeries(other *TimeSeries) (*TimeSeries, error) {
	if other == nil {
		return nil, errors.New("other time series cannot be nil")
	}
	if ts.Len() != other.Len() {
		return nil, errors.New("time series must have same length for subtraction")
	}

	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		val1 := ts.At(i)
		val2 := other.At(i)

		if math.IsNaN(val1) || math.IsNaN(val2) {
			values[i] = math.NaN()
		} else {
			values[i] = val1 - val2
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_subtracted")
}

// divideSeries divides by another time series (element-wise)
func (ts *TimeSeries) divideSeries(other *TimeSeries) (*TimeSeries, error) {
	if other == nil {
		return nil, errors.New("other time series cannot be nil")
	}
	if ts.Len() != other.Len() {
		return nil, errors.New("time series must have same length for division")
	}

	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		val1 := ts.At(i)
		val2 := other.At(i)

		if math.IsNaN(val1) || math.IsNaN(val2) || val2 == 0.0 {
			values[i] = math.NaN()
		} else {
			values[i] = val1 / val2
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_divided")
}

// multiplySeries multiplies with another time series (element-wise)
func (ts *TimeSeries) multiplySeries(other *TimeSeries) (*TimeSeries, error) {
	if other == nil {
		return nil, errors.New("other time series cannot be nil")
	}
	if ts.Len() != other.Len() {
		return nil, errors.New("time series must have same length for multiplication")
	}

	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		val1 := ts.At(i)
		val2 := other.At(i)

		if math.IsNaN(val1) || math.IsNaN(val2) {
			values[i] = math.NaN()
		} else {
			values[i] = val1 * val2
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_multiplied")
}

func (ts *TimeSeries) Min() float64 {
	minVal := math.Inf(1) // Positive infinity
	found := false
	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			if value < minVal {
				minVal = value
			}
			found = true
		}
	}
	if !found {
		return math.NaN()
	}
	return minVal
}

func (ts *TimeSeries) Max() float64 {
	maxVal := math.Inf(-1) // Negative infinity
	found := false
	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			if value > maxVal {
				maxVal = value
			}
			found = true
		}
	}
	if !found {
		return math.NaN()
	}
	return maxVal
}

func (ts *TimeSeries) Quantile(q float64) (float64, error) {
	if q < 0.0 || q > 1.0 {
		return math.NaN(), errors.New("quantile must be between 0 and 1")
	}

	// Collect non-NaN values
	values := make([]float64, 0, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			values = append(values, value)
		}
	}

	if len(values) == 0 {
		return math.NaN(), nil
	}

	// Sort values
	sort.Float64s(values)

	// Calculate quantile position
	pos := q * float64(len(values)-1)
	if pos == float64(int(pos)) {
		// Exact position
		return values[int(pos)], nil
	} else {
		// Interpolate between two values
		lower := int(pos)
		upper := lower + 1
		fraction := pos - float64(lower)
		return values[lower]*(1-fraction) + values[upper]*fraction, nil
	}
}

// Rolling window operations
func (ts *TimeSeries) RollingMean(window int) (*TimeSeries, error) {
	if window <= 0 {
		return nil, errors.New("window size must be positive")
	}
	if window > ts.Len() {
		return nil, errors.New("window size cannot exceed series length")
	}

	values := make([]float64, ts.Len())

	for i := 0; i < ts.Len(); i++ {
		if i < window-1 {
			// Not enough data for full window
			values[i] = math.NaN()
		} else {
			// Calculate mean for window ending at position i
			sum := 0.0
			count := 0
			for j := i - window + 1; j <= i; j++ {
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

	return NewTimeSeries(values, ts.index, ts.name+"_rolling_mean")
}

func (ts *TimeSeries) RollingStd(window int) (*TimeSeries, error) {
	if window <= 0 {
		return nil, errors.New("window size must be positive")
	}
	if window > ts.Len() {
		return nil, errors.New("window size cannot exceed series length")
	}

	values := make([]float64, ts.Len())

	for i := 0; i < ts.Len(); i++ {
		if i < window-1 {
			// Not enough data for full window
			values[i] = math.NaN()
		} else {
			// Calculate std for window ending at position i
			windowValues := make([]float64, 0, window)
			for j := i - window + 1; j <= i; j++ {
				val := ts.At(j)
				if !math.IsNaN(val) {
					windowValues = append(windowValues, val)
				}
			}

			if len(windowValues) <= 1 {
				values[i] = math.NaN()
			} else {
				// Calculate mean
				sum := 0.0
				for _, v := range windowValues {
					sum += v
				}
				mean := sum / float64(len(windowValues))

				// Calculate variance
				sumSquares := 0.0
				for _, v := range windowValues {
					diff := v - mean
					sumSquares += diff * diff
				}
				variance := sumSquares / float64(len(windowValues)-1) // Sample std
				values[i] = math.Sqrt(variance)
			}
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_rolling_std")
}

// variance calculates the sample variance of the time series
func (ts *TimeSeries) variance() float64 {
	if ts == nil || ts.Len() < 2 {
		return math.NaN()
	}

	// Calculate mean first
	mean := ts.Mean()
	if math.IsNaN(mean) {
		return math.NaN()
	}

	// Calculate variance
	sumSquares := 0.0
	count := 0

	for i := 0; i < ts.Len(); i++ {
		value := ts.At(i)
		if !math.IsNaN(value) {
			diff := value - mean
			sumSquares += diff * diff
			count++
		}
	}

	if count < 2 {
		return math.NaN()
	}

	return sumSquares / float64(count-1) // Sample variance
}

// addSeries adds another TimeSeries to this one element-wise
func (ts *TimeSeries) addSeries(other *TimeSeries) (*TimeSeries, error) {
	if ts == nil || other == nil {
		return nil, errors.New("time series cannot be nil")
	}
	if ts.Len() != other.Len() {
		return nil, errors.New("time series must have the same length")
	}

	values := make([]float64, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		val1 := ts.At(i)
		val2 := other.At(i)

		if math.IsNaN(val1) || math.IsNaN(val2) {
			values[i] = math.NaN()
		} else {
			values[i] = val1 + val2
		}
	}

	return NewTimeSeries(values, ts.index, ts.name+"_added")
}
