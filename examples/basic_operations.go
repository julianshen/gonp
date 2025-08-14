// Package examples demonstrates basic GoNP operations
//
// This example shows fundamental array, series, and DataFrame operations
// that form the foundation of numerical computing with GoNP.
//
// Run with: go run examples/basic_operations.go
package main

import (
	"fmt"
	"log"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/series"
	"github.com/julianshen/gonp/stats"
)

func main() {
	fmt.Println("=== GoNP Basic Operations Example ===")

	// Demonstrate array operations
	arrayOperations()

	// Demonstrate series operations
	seriesOperations()

	// Demonstrate DataFrame operations
	dataFrameOperations()

	// Demonstrate statistical operations
	statisticalOperations()
}

// arrayOperations demonstrates basic array creation and manipulation
func arrayOperations() {
	fmt.Println("1. Array Operations")
	fmt.Println("-------------------")

	// Create arrays from Go slices
	data1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	data2 := []float64{2.0, 4.0, 6.0, 8.0, 10.0}

	arr1, err := array.FromSlice(data1)
	if err != nil {
		log.Fatal("Failed to create array1:", err)
	}

	arr2, err := array.FromSlice(data2)
	if err != nil {
		log.Fatal("Failed to create array2:", err)
	}

	fmt.Printf("Array 1: %v\n", data1)
	fmt.Printf("Array 2: %v\n", data2)

	// Element-wise addition
	sum, err := arr1.Add(arr2)
	if err != nil {
		log.Fatal("Failed to add arrays:", err)
	}
	fmt.Printf("Sum:     %v\n", formatArray(sum))

	// Element-wise multiplication
	product, err := arr1.Mul(arr2)
	if err != nil {
		log.Fatal("Failed to multiply arrays:", err)
	}
	fmt.Printf("Product: %v\n", formatArray(product))

	// Access individual elements
	fmt.Printf("arr1[2] = %.1f\n", arr1.At(2).(float64))
	fmt.Printf("arr2[4] = %.1f\n", arr2.At(4).(float64))

	fmt.Printf("Array1 size: %d, Array2 size: %d\n\n", arr1.Size(), arr2.Size())
}

// seriesOperations demonstrates Series creation and manipulation
func seriesOperations() {
	fmt.Println("2. Series Operations")
	fmt.Println("--------------------")

	// Create sample temperature data
	temperatures := []float64{20.1, 22.3, 19.8, 25.2, 23.7, 21.9, 24.1}
	tempArr, err := array.FromSlice(temperatures)
	if err != nil {
		log.Fatal("Failed to create temperature array:", err)
	}

	// Create Series with meaningful name
	tempSeries, err := series.NewSeries(tempArr, nil, "temperature_celsius")
	if err != nil {
		log.Fatal("Failed to create temperature series:", err)
	}

	fmt.Printf("Temperature series: %s\n", tempSeries.Name())
	fmt.Printf("Length: %d measurements\n", tempSeries.Len())

	// Access elements
	fmt.Printf("First temperature: %.1f°C\n", tempSeries.At(0).(float64))
	fmt.Printf("Last temperature:  %.1f°C\n", tempSeries.At(tempSeries.Len()-1).(float64))

	// Slice operations
	firstThree, err := tempSeries.Slice(0, 3)
	if err != nil {
		log.Fatal("Failed to slice series:", err)
	}
	fmt.Printf("First 3 temperatures: %v\n", formatSeriesValues(firstThree))

	lastTwo := tempSeries.Tail(2)
	fmt.Printf("Last 2 temperatures:  %v\n", formatSeriesValues(lastTwo))

	fmt.Println()
}

// dataFrameOperations demonstrates DataFrame creation and manipulation
func dataFrameOperations() {
	fmt.Println("3. DataFrame Operations")
	fmt.Println("-----------------------")

	// Create sample weather data
	temperatures := []float64{20.1, 22.3, 19.8, 25.2, 23.7}
	humidity := []float64{45.2, 50.1, 42.8, 55.3, 48.9}
	windSpeed := []float64{5.2, 3.8, 7.1, 2.9, 4.5}

	// Create arrays
	tempArr, _ := array.FromSlice(temperatures)
	humidityArr, _ := array.FromSlice(humidity)
	windArr, _ := array.FromSlice(windSpeed)

	// Create Series
	tempSeries, _ := series.NewSeries(tempArr, nil, "temperature")
	humiditySeries, _ := series.NewSeries(humidityArr, nil, "humidity")
	windSeries, _ := series.NewSeries(windArr, nil, "wind_speed")

	// Create DataFrame
	df, err := dataframe.FromSeries([]*series.Series{tempSeries, humiditySeries, windSeries})
	if err != nil {
		log.Fatal("Failed to create DataFrame:", err)
	}

	fmt.Printf("Weather DataFrame: %d rows × %d columns\n", df.Len(), len(df.Columns()))
	fmt.Printf("Columns: %v\n", df.Columns())

	// Access specific values
	temp0, _ := df.IAt(0, 0)     // First temperature
	humidity1, _ := df.IAt(1, 1) // Second humidity reading
	wind2, _ := df.IAt(2, 2)     // Third wind speed

	fmt.Printf("Day 1 temperature: %.1f°C\n", temp0.(float64))
	fmt.Printf("Day 2 humidity: %.1f%%\n", humidity1.(float64))
	fmt.Printf("Day 3 wind speed: %.1f m/s\n", wind2.(float64))

	// Get entire columns
	tempColumn, _ := df.GetColumn("temperature")
	fmt.Printf("All temperatures: %v\n", formatSeriesValues(tempColumn))

	// Slice DataFrame (first 3 days)
	firstThreeDays, err := df.Slice(0, 3)
	if err != nil {
		log.Fatal("Failed to slice DataFrame:", err)
	}
	fmt.Printf("First 3 days: %d rows\n", firstThreeDays.Len())

	fmt.Println()
}

// statisticalOperations demonstrates statistical analysis
func statisticalOperations() {
	fmt.Println("4. Statistical Analysis")
	fmt.Println("-----------------------")

	// Sample data: exam scores
	scores := []float64{85.5, 92.0, 78.5, 96.0, 89.5, 82.0, 94.5, 87.0, 91.5, 88.0}
	scoresArr, err := array.FromSlice(scores)
	if err != nil {
		log.Fatal("Failed to create scores array:", err)
	}

	fmt.Printf("Exam scores: %v\n", scores)

	// Calculate descriptive statistics
	mean, err := stats.Mean(scoresArr)
	if err != nil {
		log.Fatal("Failed to calculate mean:", err)
	}

	median, err := stats.Median(scoresArr)
	if err != nil {
		log.Fatal("Failed to calculate median:", err)
	}

	stdDev, err := stats.Std(scoresArr)
	if err != nil {
		log.Fatal("Failed to calculate standard deviation:", err)
	}

	variance, err := stats.Var(scoresArr)
	if err != nil {
		log.Fatal("Failed to calculate variance:", err)
	}

	min, err := stats.Min(scoresArr)
	if err != nil {
		log.Fatal("Failed to calculate minimum:", err)
	}

	max, err := stats.Max(scoresArr)
	if err != nil {
		log.Fatal("Failed to calculate maximum:", err)
	}

	fmt.Printf("Mean:               %.2f\n", mean)
	fmt.Printf("Median:             %.2f\n", median)
	fmt.Printf("Standard Deviation: %.2f\n", stdDev)
	fmt.Printf("Variance:           %.2f\n", variance)
	fmt.Printf("Minimum:            %.2f\n", min)
	fmt.Printf("Maximum:            %.2f\n", max)
	fmt.Printf("Range:              %.2f\n", max-min)

	// Correlation analysis
	studyHours := []float64{8.0, 10.0, 6.0, 12.0, 9.0, 7.0, 11.0, 8.5, 10.5, 9.5}
	studyArr, _ := array.FromSlice(studyHours)

	correlation, err := stats.Correlation(scoresArr, studyArr)
	if err != nil {
		log.Fatal("Failed to calculate correlation:", err)
	}

	fmt.Printf("\nCorrelation between study hours and scores: %.3f\n", correlation)
	if correlation > 0.7 {
		fmt.Println("Strong positive correlation - more study time correlates with higher scores!")
	} else if correlation > 0.3 {
		fmt.Println("Moderate positive correlation between study time and scores.")
	} else {
		fmt.Println("Weak correlation between study time and scores.")
	}

	fmt.Println()
}

// Helper functions for formatting output

func formatArray(arr *array.Array) []float64 {
	result := make([]float64, arr.Size())
	for i := 0; i < arr.Size(); i++ {
		result[i] = arr.At(i).(float64)
	}
	return result
}

func formatSeriesValues(s *series.Series) []float64 {
	result := make([]float64, s.Len())
	for i := 0; i < s.Len(); i++ {
		result[i] = s.At(i).(float64)
	}
	return result
}
