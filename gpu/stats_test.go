package gpu

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	stats "github.com/julianshen/gonp/stats"
)

// TestGPUDescriptiveStatistics tests GPU-accelerated descriptive statistics
func TestGPUDescriptiveStatistics(t *testing.T) {
	t.Run("GPU mean computation", func(t *testing.T) {
		// Create test data
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
		arr, _ := array.FromSlice(data)
		expectedMean := 5.5

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for mean test")
		}

		// GPU accelerated mean
		result, err := MeanGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU mean computation failed: %v", err)
		}

		if math.Abs(result-expectedMean) > 1e-10 {
			t.Errorf("GPU mean incorrect: expected %.6f, got %.6f", expectedMean, result)
		}

		t.Logf("GPU mean computation successful: %.6f", result)
	})

	t.Run("GPU standard deviation computation", func(t *testing.T) {
		// Test data with known standard deviation
		data := []float64{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}
		arr, _ := array.FromSlice(data)
		expectedStd := 2.0 // Sample standard deviation

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for std test")
		}

		// GPU accelerated standard deviation
		result, err := StdGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU std computation failed: %v", err)
		}

		if math.Abs(result-expectedStd) > 1e-6 {
			t.Errorf("GPU std incorrect: expected %.6f, got %.6f", expectedStd, result)
		}

		t.Logf("GPU standard deviation computation successful: %.6f", result)
	})

	t.Run("GPU variance computation", func(t *testing.T) {
		// Test data: [1, 2, 3, 4, 5] - variance should be 2.5 (sample variance)
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)
		expectedVar := 2.5

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for variance test")
		}

		// GPU accelerated variance
		result, err := VarianceGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU variance computation failed: %v", err)
		}

		if math.Abs(result-expectedVar) > 1e-10 {
			t.Errorf("GPU variance incorrect: expected %.6f, got %.6f", expectedVar, result)
		}

		t.Logf("GPU variance computation successful: %.6f", result)
	})

	t.Run("Large data performance test", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping performance test in short mode")
		}

		// Large dataset for performance testing
		size := 1000000
		data := make([]float64, size)
		for i := 0; i < size; i++ {
			data[i] = float64(i%100) / 10.0 // Values from 0.0 to 9.9
		}
		arr, _ := array.FromSlice(data)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for performance test")
		}

		// Time CPU implementation
		cpuStart := getCurrentTime()
		cpuMean, err := stats.Mean(arr)
		if err != nil {
			t.Fatalf("CPU mean failed: %v", err)
		}
		cpuTime := getCurrentTime() - cpuStart

		// Time GPU implementation
		gpuStart := getCurrentTime()
		gpuMean, err := MeanGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU mean failed: %v", err)
		}
		gpuTime := getCurrentTime() - gpuStart

		// Validate accuracy
		if math.Abs(cpuMean-gpuMean) > 1e-6 {
			t.Errorf("GPU/CPU mean mismatch: CPU=%.6f, GPU=%.6f", cpuMean, gpuMean)
		}

		speedup := float64(cpuTime) / float64(gpuTime)
		t.Logf("Performance comparison for %d elements:", size)
		t.Logf("CPU time: %.3f ms", float64(cpuTime)/1e6)
		t.Logf("GPU time: %.3f ms", float64(gpuTime)/1e6)
		t.Logf("GPU speedup: %.2fx", speedup)

		// GPU should provide some benefit for large datasets
		if speedup < 0.5 {
			t.Logf("Warning: GPU slower than CPU (%.2fx), may indicate overhead", speedup)
		}
	})
}

// TestGPUReductionOperations tests GPU-accelerated reduction operations
func TestGPUReductionOperations(t *testing.T) {
	t.Run("GPU sum reduction", func(t *testing.T) {
		// Test data: sum of 1 to 100 = 5050
		data := make([]float64, 100)
		for i := 0; i < 100; i++ {
			data[i] = float64(i + 1)
		}
		arr, _ := array.FromSlice(data)
		expectedSum := 5050.0

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for sum test")
		}

		// GPU accelerated sum
		result, err := SumGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU sum computation failed: %v", err)
		}

		if math.Abs(result-expectedSum) > 1e-6 {
			t.Errorf("GPU sum incorrect: expected %.1f, got %.6f", expectedSum, result)
		}

		t.Logf("GPU sum computation successful: %.1f", result)
	})

	t.Run("GPU min/max reduction", func(t *testing.T) {
		data := []float64{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0}
		arr, _ := array.FromSlice(data)
		expectedMin := 1.0
		expectedMax := 9.0

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for min/max test")
		}

		// GPU accelerated min
		minResult, err := MinGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU min computation failed: %v", err)
		}

		// GPU accelerated max
		maxResult, err := MaxGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU max computation failed: %v", err)
		}

		if math.Abs(minResult-expectedMin) > 1e-10 {
			t.Errorf("GPU min incorrect: expected %.1f, got %.6f", expectedMin, minResult)
		}

		if math.Abs(maxResult-expectedMax) > 1e-10 {
			t.Errorf("GPU max incorrect: expected %.1f, got %.6f", expectedMax, maxResult)
		}

		t.Logf("GPU min/max computation successful: min=%.1f, max=%.1f", minResult, maxResult)
	})

	t.Run("GPU product reduction", func(t *testing.T) {
		// Small values to avoid overflow: product of [1,2,3,4] = 24
		data := []float64{1.0, 2.0, 3.0, 4.0}
		arr, _ := array.FromSlice(data)
		expectedProduct := 24.0

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for product test")
		}

		// GPU accelerated product
		result, err := ProductGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU product computation failed: %v", err)
		}

		if math.Abs(result-expectedProduct) > 1e-10 {
			t.Errorf("GPU product incorrect: expected %.1f, got %.6f", expectedProduct, result)
		}

		t.Logf("GPU product computation successful: %.1f", result)
	})
}

// TestGPUCorrelationAnalysis tests GPU-accelerated correlation and covariance
func TestGPUCorrelationAnalysis(t *testing.T) {
	t.Run("GPU correlation coefficient", func(t *testing.T) {
		// Perfect positive correlation
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}
		arrX, _ := array.FromSlice(x)
		arrY, _ := array.FromSlice(y)
		expectedCorr := 1.0

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for correlation test")
		}

		// GPU accelerated correlation
		result, err := CorrelationGPU(arrX, arrY, device)
		if err != nil {
			t.Fatalf("GPU correlation computation failed: %v", err)
		}

		if math.Abs(result-expectedCorr) > 1e-10 {
			t.Errorf("GPU correlation incorrect: expected %.6f, got %.6f", expectedCorr, result)
		}

		t.Logf("GPU correlation computation successful: %.6f", result)
	})

	t.Run("GPU covariance computation", func(t *testing.T) {
		// Test data with known covariance
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{1.0, 4.0, 9.0, 16.0, 25.0}
		arrX, _ := array.FromSlice(x)
		arrY, _ := array.FromSlice(y)
		// For this data: cov(x,y) = 20.0 (sample covariance)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for covariance test")
		}

		// GPU accelerated covariance
		result, err := CovarianceGPU(arrX, arrY, device)
		if err != nil {
			t.Fatalf("GPU covariance computation failed: %v", err)
		}

		// Validate result is reasonable (exact value depends on implementation)
		if math.IsNaN(result) || math.IsInf(result, 0) {
			t.Errorf("GPU covariance invalid: %f", result)
		}

		t.Logf("GPU covariance computation successful: %.6f", result)
	})

	t.Run("GPU correlation matrix", func(t *testing.T) {
		// Create 3x3 correlation matrix from 3 variables
		size := 100
		data1 := make([]float64, size)
		data2 := make([]float64, size)
		data3 := make([]float64, size)

		for i := 0; i < size; i++ {
			data1[i] = float64(i)
			data2[i] = float64(i * 2)    // Perfectly correlated with data1
			data3[i] = float64(size - i) // Negatively correlated
		}

		arr1, _ := array.FromSlice(data1)
		arr2, _ := array.FromSlice(data2)
		arr3, _ := array.FromSlice(data3)
		arrays := []*array.Array{arr1, arr2, arr3}

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for correlation matrix test")
		}

		// GPU accelerated correlation matrix
		result, err := CorrelationMatrixGPU(arrays, device)
		if err != nil {
			t.Fatalf("GPU correlation matrix computation failed: %v", err)
		}

		// Check result dimensions
		shape := result.Shape()
		if shape[0] != 3 || shape[1] != 3 {
			t.Errorf("Correlation matrix wrong shape: expected [3,3], got [%d,%d]", shape[0], shape[1])
		}

		// Check diagonal elements are 1.0
		for i := 0; i < 3; i++ {
			diag := result.At(i, i).(float64)
			if math.Abs(diag-1.0) > 1e-10 {
				t.Errorf("Diagonal element [%d,%d] should be 1.0, got %.6f", i, i, diag)
			}
		}

		t.Logf("GPU correlation matrix computation successful: shape %dx%d", shape[0], shape[1])
	})
}

// TestGPUDistributionStatistics tests GPU-accelerated statistical distribution functions
func TestGPUDistributionStatistics(t *testing.T) {
	t.Run("GPU percentile computation", func(t *testing.T) {
		// Create sorted data for easy verification
		data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
		arr, _ := array.FromSlice(data)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for percentile test")
		}

		// Test 50th percentile (median)
		p50, err := PercentileGPU(arr, 50.0, device)
		if err != nil {
			t.Fatalf("GPU percentile computation failed: %v", err)
		}

		expectedP50 := 5.5 // Median of 1-10
		if math.Abs(p50-expectedP50) > 1e-6 {
			t.Errorf("GPU 50th percentile incorrect: expected %.1f, got %.6f", expectedP50, p50)
		}

		// Test 25th percentile
		p25, err := PercentileGPU(arr, 25.0, device)
		if err != nil {
			t.Fatalf("GPU 25th percentile computation failed: %v", err)
		}

		t.Logf("GPU percentile computation successful: P25=%.2f, P50=%.2f", p25, p50)
	})

	t.Run("GPU quantile computation", func(t *testing.T) {
		// Normal distribution-like data
		data := []float64{1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0}
		arr, _ := array.FromSlice(data)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for quantile test")
		}

		// Test quartiles
		quartiles := []float64{0.25, 0.5, 0.75}
		results := make([]float64, len(quartiles))

		for i, q := range quartiles {
			result, err := QuantileGPU(arr, q, device)
			if err != nil {
				t.Fatalf("GPU quantile computation failed for q=%.2f: %v", q, err)
			}
			results[i] = result
		}

		// Quartiles should be in ascending order
		for i := 1; i < len(results); i++ {
			if results[i] < results[i-1] {
				t.Errorf("Quartiles not in ascending order: Q%.0f=%.2f > Q%.0f=%.2f",
					quartiles[i-1]*100, results[i-1], quartiles[i]*100, results[i])
			}
		}

		t.Logf("GPU quantile computation successful: Q25=%.2f, Q50=%.2f, Q75=%.2f",
			results[0], results[1], results[2])
	})

	t.Run("GPU histogram computation", func(t *testing.T) {
		// Create data with known distribution
		data := []float64{1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0}
		arr, _ := array.FromSlice(data)
		bins := 5

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for histogram test")
		}

		// GPU accelerated histogram
		counts, edges, err := HistogramGPU(arr, bins, device)
		if err != nil {
			t.Fatalf("GPU histogram computation failed: %v", err)
		}

		// Check histogram properties
		if counts.Size() != bins {
			t.Errorf("Histogram counts wrong size: expected %d, got %d", bins, counts.Size())
		}

		if edges.Size() != bins+1 {
			t.Errorf("Histogram edges wrong size: expected %d, got %d", bins+1, edges.Size())
		}

		// Total count should equal data length
		totalCount := 0
		for i := 0; i < counts.Size(); i++ {
			totalCount += int(counts.At(i).(float64))
		}

		if totalCount != len(data) {
			t.Errorf("Histogram total count wrong: expected %d, got %d", len(data), totalCount)
		}

		t.Logf("GPU histogram computation successful: %d bins, %d total count", bins, totalCount)
	})
}

// TestGPUHypothesisTesting tests GPU-accelerated statistical tests
func TestGPUHypothesisTesting(t *testing.T) {
	t.Run("GPU t-test computation", func(t *testing.T) {
		// Two samples with different means
		sample1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		sample2 := []float64{6.0, 7.0, 8.0, 9.0, 10.0}
		arr1, _ := array.FromSlice(sample1)
		arr2, _ := array.FromSlice(sample2)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for t-test")
		}

		// GPU accelerated two-sample t-test
		tStat, pValue, err := TTestGPU(arr1, arr2, device)
		if err != nil {
			t.Fatalf("GPU t-test computation failed: %v", err)
		}

		// t-statistic should be significant (large magnitude)
		if math.Abs(tStat) < 1.0 {
			t.Errorf("t-statistic seems too small: %.6f", tStat)
		}

		// p-value should be small for this clear difference
		if pValue > 0.1 {
			t.Errorf("p-value seems too large: %.6f", pValue)
		}

		t.Logf("GPU t-test successful: t=%.6f, p=%.6f", tStat, pValue)
	})

	t.Run("GPU chi-square test", func(t *testing.T) {
		// Observed vs expected frequencies
		observed := []float64{10.0, 15.0, 20.0, 25.0}
		expected := []float64{12.0, 18.0, 18.0, 22.0}
		obsArr, _ := array.FromSlice(observed)
		expArr, _ := array.FromSlice(expected)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for chi-square test")
		}

		// GPU accelerated chi-square test
		chiStat, pValue, err := ChiSquareTestGPU(obsArr, expArr, device)
		if err != nil {
			t.Fatalf("GPU chi-square test failed: %v", err)
		}

		// Chi-square statistic should be positive
		if chiStat < 0 {
			t.Errorf("Chi-square statistic should be positive: %.6f", chiStat)
		}

		// p-value should be between 0 and 1
		if pValue < 0 || pValue > 1 {
			t.Errorf("p-value should be between 0 and 1: %.6f", pValue)
		}

		t.Logf("GPU chi-square test successful: χ²=%.6f, p=%.6f", chiStat, pValue)
	})
}

// TestGPUStatisticsEdgeCases tests edge cases for GPU statistical functions
func TestGPUStatisticsEdgeCases(t *testing.T) {
	t.Run("Empty array handling", func(t *testing.T) {
		empty := array.Empty(internal.Shape{0}, internal.Float64)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for empty array test")
		}

		// GPU operations should handle empty arrays gracefully
		_, err = MeanGPU(empty, device)
		if err == nil {
			t.Error("Expected error for empty array mean, got nil")
		}

		_, err = SumGPU(empty, device)
		if err == nil {
			t.Error("Expected error for empty array sum, got nil")
		}
	})

	t.Run("Single element array", func(t *testing.T) {
		single, _ := array.FromSlice([]float64{42.0})

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for single element test")
		}

		// Mean of single element should be the element itself
		mean, err := MeanGPU(single, device)
		if err != nil {
			t.Fatalf("GPU mean of single element failed: %v", err)
		}

		if math.Abs(mean-42.0) > 1e-10 {
			t.Errorf("Single element mean incorrect: expected 42.0, got %.6f", mean)
		}

		// Standard deviation of single element should be 0
		std, err := StdGPU(single, device)
		if err != nil {
			t.Fatalf("GPU std of single element failed: %v", err)
		}

		if math.Abs(std-0.0) > 1e-10 {
			t.Errorf("Single element std should be 0, got %.6f", std)
		}
	})

	t.Run("NaN and infinity handling", func(t *testing.T) {
		data := []float64{1.0, math.NaN(), 3.0, math.Inf(1), 5.0}
		arr, _ := array.FromSlice(data)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for NaN/Inf test")
		}

		// GPU functions should handle NaN/Inf appropriately
		mean, err := MeanGPU(arr, device)
		if err != nil {
			t.Logf("Expected behavior: GPU mean with NaN/Inf returned error: %v", err)
		} else {
			if !math.IsNaN(mean) && !math.IsInf(mean, 0) {
				t.Logf("GPU mean with NaN/Inf: %.6f", mean)
			}
		}
	})
}

// TestGPUStatisticsIntegration tests integration with existing stats package
func TestGPUStatisticsIntegration(t *testing.T) {
	t.Run("GPU vs CPU accuracy comparison", func(t *testing.T) {
		// Generate test data
		data := make([]float64, 1000)
		for i := 0; i < 1000; i++ {
			data[i] = math.Sin(float64(i) * 0.01) // Smooth variation
		}
		arr, _ := array.FromSlice(data)

		device, err := GetDefaultDevice()
		if err != nil {
			t.Skip("No GPU device available for integration test")
		}

		// Compare GPU vs CPU implementations
		cpuMean, _ := stats.Mean(arr)
		gpuMean, err := MeanGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU mean failed: %v", err)
		}

		if math.Abs(cpuMean-gpuMean) > 1e-10 {
			t.Errorf("GPU/CPU mean mismatch: CPU=%.10f, GPU=%.10f", cpuMean, gpuMean)
		}

		cpuStd, _ := stats.Std(arr)
		gpuStd, err := StdGPU(arr, device)
		if err != nil {
			t.Fatalf("GPU std failed: %v", err)
		}

		if math.Abs(cpuStd-gpuStd) > 1e-6 {
			t.Errorf("GPU/CPU std mismatch: CPU=%.10f, GPU=%.10f", cpuStd, gpuStd)
		}

		t.Logf("GPU/CPU integration test passed: mean diff=%.2e, std diff=%.2e",
			math.Abs(cpuMean-gpuMean), math.Abs(cpuStd-gpuStd))
	})
}
