//go:build ignore
// +build ignore

// Package examples demonstrates error handling patterns with GoNP
//
// This example shows comprehensive error handling including:
// - Error type identification
// - Graceful error recovery
// - Error reporting and logging
// - Best practices for error handling
//
// Run with: go run examples/error_handling.go
package main

import (
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
	"github.com/julianshen/gonp/stats"
)

func main() {
	fmt.Println("=== GoNP Error Handling Examples ===\n")

	// Demonstrate different types of errors
	demonstrateArrayErrors()
	demonstrateSeriesErrors()
	demonstrateDataFrameErrors()
	demonstrateStatisticalErrors()
	demonstrateErrorRecovery()
}

// demonstrateArrayErrors shows array-related error scenarios
func demonstrateArrayErrors() {
	fmt.Println("1. Array Error Handling")
	fmt.Println("-----------------------")

	// Shape mismatch errors
	fmt.Println("Shape Mismatch Errors:")
	arr1, _ := array.FromSlice([]float64{1, 2, 3})
	arr2, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

	_, err := arr1.Add(arr2)
	if err != nil {
		fmt.Printf("❌ Addition failed: %v\n", err)
		if internal.IsShapeError(err) {
			fmt.Println("   ℹ️  This is a shape mismatch error - arrays must have compatible shapes")
		}
	}

	// Index out of bounds errors
	fmt.Println("\nIndex Out of Bounds Errors:")
	arr, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

	// Try to access invalid index
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("❌ Index access panicked: %v\n", r)
			fmt.Println("   ℹ️  Always validate indices before accessing array elements")
		}
	}()

	// This would panic - demonstrating defensive programming
	fmt.Println("Attempting to access index 10 in array of length 5...")
	safeArrayAccess(arr, 10)

	// Valid access
	value := safeArrayAccess(arr, 2)
	if value != nil {
		fmt.Printf("✅ Safe access: arr[2] = %.1f\n", *value)
	}

	fmt.Println()
}

// demonstrateSeriesErrors shows series-related error scenarios
func demonstrateSeriesErrors() {
	fmt.Println("2. Series Error Handling")
	fmt.Println("------------------------")

	// Invalid slice ranges
	fmt.Println("Invalid Slice Operations:")
	data, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
	s, _ := series.NewSeries(data, nil, "test_series")

	// Invalid slice range
	_, err := s.Slice(10, 15)
	if err != nil {
		fmt.Printf("❌ Slice failed: %v\n", err)
		fmt.Println("   ℹ️  Slice bounds must be within series length")
	}

	// Valid slice with error handling
	result, err := safeSlice(s, 1, 4)
	if err != nil {
		fmt.Printf("❌ Safe slice failed: %v\n", err)
	} else {
		fmt.Printf("✅ Safe slice successful: length = %d\n", result.Len())
	}

	// Empty series operations
	fmt.Println("\nEmpty Series Operations:")

	// Create a very small series to demonstrate the concept
	smallData, _ := array.FromSlice([]float64{42.0})
	smallSeries, _ := series.NewSeries(smallData, nil, "small")

	if smallSeries.Len() == 1 {
		fmt.Println("✅ Small series created successfully")
		fmt.Println("   ℹ️  Always validate data before creating series")
	}

	// Demonstrate checking for minimum data requirements
	if smallSeries.Len() < 2 {
		fmt.Println("⚠️  Warning: Series has insufficient data for certain operations")
		fmt.Println("   ℹ️  Some statistical operations require multiple data points")
	}

	fmt.Println()
}

// demonstrateDataFrameErrors shows DataFrame-related error scenarios
func demonstrateDataFrameErrors() {
	fmt.Println("3. DataFrame Error Handling")
	fmt.Println("---------------------------")

	// Mismatched series lengths
	fmt.Println("Mismatched Series Lengths:")
	shortData, _ := array.FromSlice([]float64{1, 2, 3})
	longData, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

	shortSeries, _ := series.NewSeries(shortData, nil, "short")
	longSeries, _ := series.NewSeries(longData, nil, "long")

	_, err := dataframe.FromSeries([]*series.Series{shortSeries, longSeries})
	if err != nil {
		fmt.Printf("❌ DataFrame creation failed: %v\n", err)
		fmt.Println("   ℹ️  All series must have the same length")
	}

	// Valid DataFrame with error handling
	validSeries1, _ := series.NewSeries(shortData, nil, "col1")
	validSeries2, _ := series.NewSeries(shortData, nil, "col2")

	df, err := dataframe.FromSeries([]*series.Series{validSeries1, validSeries2})
	if err != nil {
		fmt.Printf("❌ DataFrame creation failed: %v\n", err)
		return
	}

	fmt.Printf("✅ DataFrame created successfully: %d×%d\n", df.Len(), len(df.Columns()))

	// Invalid indexing
	fmt.Println("\nInvalid DataFrame Indexing:")
	value, err := safeDataFrameAccess(df, 5, 0) // Row 5 doesn't exist
	if err != nil {
		fmt.Printf("❌ Index access failed: %v\n", err)
		fmt.Println("   ℹ️  Always validate row and column indices")
	}

	// Valid indexing
	value, err = safeDataFrameAccess(df, 1, 0)
	if err != nil {
		fmt.Printf("❌ Safe access failed: %v\n", err)
	} else {
		fmt.Printf("✅ Safe access: df[1,0] = %.1f\n", value)
	}

	// Non-existent columns
	fmt.Println("\nNon-existent Column Access:")
	_, err = df.GetColumn("nonexistent")
	if err != nil {
		fmt.Printf("❌ Column access failed: %v\n", err)
		fmt.Printf("   ℹ️  Available columns: %v\n", df.Columns())
	}

	fmt.Println()
}

// demonstrateStatisticalErrors shows statistical computation error scenarios
func demonstrateStatisticalErrors() {
	fmt.Println("4. Statistical Error Handling")
	fmt.Println("-----------------------------")

	// Empty data statistics
	fmt.Println("Empty Data Statistics:")
	emptyData := array.Empty(internal.Shape{0}, internal.Float64)

	_, err := stats.Mean(emptyData)
	if err != nil {
		fmt.Printf("❌ Mean calculation failed: %v\n", err)
		fmt.Println("   ℹ️  Cannot calculate statistics on empty data")
	}

	// Invalid correlation (different lengths)
	fmt.Println("\nInvalid Correlation:")
	arr1, _ := array.FromSlice([]float64{1, 2, 3})
	arr2, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})

	_, err = stats.Correlation(arr1, arr2)
	if err != nil {
		fmt.Printf("❌ Correlation failed: %v\n", err)
		fmt.Println("   ℹ️  Arrays must have the same length for correlation")
	}

	// Valid statistics with error handling
	validData, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
	mean, err := safeStatistics(validData)
	if err != nil {
		fmt.Printf("❌ Statistics failed: %v\n", err)
	} else {
		fmt.Printf("✅ Safe statistics: mean = %.2f\n", mean)
	}

	fmt.Println()
}

// demonstrateErrorRecovery shows patterns for error recovery and graceful degradation
func demonstrateErrorRecovery() {
	fmt.Println("5. Error Recovery Patterns")
	fmt.Println("--------------------------")

	// Graceful degradation
	fmt.Println("Graceful Degradation Example:")

	// Simulate a data processing pipeline with potential failures
	results := processDataPipeline()

	fmt.Printf("Pipeline completed with %d successful operations out of %d attempted\n",
		results.successful, results.total)

	if results.errors > 0 {
		fmt.Printf("⚠️  %d operations failed but pipeline continued\n", results.errors)
		fmt.Println("   ℹ️  Consider implementing retry mechanisms for transient failures")
	} else {
		fmt.Println("✅ All operations completed successfully")
	}

	// Error aggregation
	fmt.Println("\nError Aggregation Example:")
	errors := validateMultipleInputs()
	if len(errors) > 0 {
		fmt.Printf("Found %d validation errors:\n", len(errors))
		for i, err := range errors {
			fmt.Printf("  %d. %v\n", i+1, err)
		}
		fmt.Println("   ℹ️  Collect all validation errors before failing")
	} else {
		fmt.Println("✅ All inputs are valid")
	}
}

// Helper functions for safe operations

func safeArrayAccess(arr *array.Array, index int) *float64 {
	if index < 0 || index >= arr.Size() {
		fmt.Printf("⚠️  Index %d out of bounds (array size: %d)\n", index, arr.Size())
		return nil
	}

	value := arr.At(index).(float64)
	return &value
}

func safeSlice(s *series.Series, start, end int) (*series.Series, error) {
	if start < 0 {
		return nil, fmt.Errorf("start index cannot be negative: %d", start)
	}
	if end > s.Len() {
		return nil, fmt.Errorf("end index %d exceeds series length %d", end, s.Len())
	}
	if start >= end {
		return nil, fmt.Errorf("start index %d must be less than end index %d", start, end)
	}

	return s.Slice(start, end)
}

func safeDataFrameAccess(df *dataframe.DataFrame, row, col int) (float64, error) {
	if row < 0 || row >= df.Len() {
		return 0, fmt.Errorf("row index %d out of bounds (DataFrame has %d rows)", row, df.Len())
	}
	if col < 0 || col >= len(df.Columns()) {
		return 0, fmt.Errorf("column index %d out of bounds (DataFrame has %d columns)", col, len(df.Columns()))
	}

	value, err := df.IAt(row, col)
	if err != nil {
		return 0, err
	}

	return value.(float64), nil
}

func safeStatistics(arr *array.Array) (float64, error) {
	if arr.Size() == 0 {
		return 0, fmt.Errorf("cannot calculate statistics on empty array")
	}

	return stats.Mean(arr)
}

// Pipeline result tracking
type PipelineResult struct {
	successful int
	errors     int
	total      int
}

func processDataPipeline() PipelineResult {
	result := PipelineResult{}

	// Simulate multiple operations with some failures
	operations := []struct {
		name string
		data []float64
	}{
		{"Operation 1", []float64{1, 2, 3, 4, 5}},
		{"Operation 2", []float64{42}}, // Small but valid
		{"Operation 3", []float64{10, 20, 30}},
		{"Operation 4", []float64{-1, -2, -3}},
		{"Operation 5", []float64{100, 200}}, // Valid operation
	}

	for _, op := range operations {
		result.total++

		arr, err := array.FromSlice(op.data)
		if err != nil {
			fmt.Printf("  ❌ %s failed: %v\n", op.name, err)
			result.errors++
			continue
		}

		// All operations now have valid data, but we can check for other conditions
		if arr.Size() < 2 {
			fmt.Printf("  ⚠️  %s: minimal data (size: %d)\n", op.name, arr.Size())
		}

		mean, err := stats.Mean(arr)
		if err != nil {
			fmt.Printf("  ❌ %s failed: %v\n", op.name, err)
			result.errors++
			continue
		}

		fmt.Printf("  ✅ %s succeeded: mean = %.2f\n", op.name, mean)
		result.successful++
	}

	return result
}

func validateMultipleInputs() []error {
	var errors []error

	// Simulate validation of multiple inputs
	inputs := []struct {
		name  string
		value interface{}
	}{
		{"temperature", -300.0}, // Invalid: below absolute zero
		{"humidity", 150.0},     // Invalid: over 100%
		{"pressure", 1013.25},   // Valid
		{"wind_speed", -5.0},    // Invalid: negative wind speed
		{"visibility", 10.0},    // Valid
	}

	for _, input := range inputs {
		switch input.name {
		case "temperature":
			if temp := input.value.(float64); temp < -273.15 {
				errors = append(errors, fmt.Errorf("temperature %.2f°C is below absolute zero", temp))
			}
		case "humidity":
			if humidity := input.value.(float64); humidity < 0 || humidity > 100 {
				errors = append(errors, fmt.Errorf("humidity %.1f%% must be between 0 and 100", humidity))
			}
		case "wind_speed":
			if speed := input.value.(float64); speed < 0 {
				errors = append(errors, fmt.Errorf("wind speed %.1f cannot be negative", speed))
			}
		}
	}

	return errors
}
