package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
)

func TestAdvancedDescriptiveStats(t *testing.T) {
	// Create test data with known statistical properties
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	arr, _ := array.FromSlice(data)

	// Test Skewness (should be 0 for symmetric data)
	skewness, err := Skewness(arr)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(skewness) > 1e-10 {
		t.Errorf("Skewness of symmetric data should be ~0, got %.6f", skewness)
	}

	// Test Kurtosis
	kurtosis, err := Kurtosis(arr)
	if err != nil {
		t.Fatal(err)
	}
	// For uniform distribution, kurtosis should be less than normal (3)
	if kurtosis > 3.5 {
		t.Errorf("Kurtosis seems too high for uniform data: %.6f", kurtosis)
	}

	// Test Range
	rangeVal, err := Range(arr)
	if err != nil {
		t.Fatal(err)
	}
	expected := 9.0 // 10 - 1
	if math.Abs(rangeVal-expected) > 1e-10 {
		t.Errorf("Range: got %.6f, expected %.6f", rangeVal, expected)
	}

	// Test Mode
	modeData := []float64{1, 2, 2, 3, 3, 3, 4, 4}
	modeArr, _ := array.FromSlice(modeData)
	mode, err := Mode(modeArr)
	if err != nil {
		t.Fatal(err)
	}
	if mode != 3.0 {
		t.Errorf("Mode: got %.1f, expected 3.0", mode)
	}

	// Test Percentile
	p90, err := Percentile(arr, 90)
	if err != nil {
		t.Fatal(err)
	}
	// 90th percentile of 1-10 should be 9.1
	if math.Abs(p90-9.1) > 0.1 {
		t.Errorf("90th percentile: got %.2f, expected ~9.1", p90)
	}

	t.Logf("Advanced descriptive stats test passed")
}

func TestDistanceMetrics(t *testing.T) {
	// Create test vectors
	x, _ := array.FromSlice([]float64{1, 2, 3})
	y, _ := array.FromSlice([]float64{4, 5, 6})

	// Test Euclidean distance
	euclidean, err := EuclideanDistance(x, y)
	if err != nil {
		t.Fatal(err)
	}
	expected := math.Sqrt(9 + 9 + 9) // sqrt(27)
	if math.Abs(euclidean-expected) > 1e-10 {
		t.Errorf("Euclidean distance: got %.6f, expected %.6f", euclidean, expected)
	}

	// Test Manhattan distance
	manhattan, err := ManhattanDistance(x, y)
	if err != nil {
		t.Fatal(err)
	}
	expected = 9.0 // |1-4| + |2-5| + |3-6| = 3+3+3
	if math.Abs(manhattan-expected) > 1e-10 {
		t.Errorf("Manhattan distance: got %.6f, expected %.6f", manhattan, expected)
	}

	// Test Cosine similarity
	cosine, err := CosineSimilarity(x, y)
	if err != nil {
		t.Fatal(err)
	}
	// Cosine similarity of (1,2,3) and (4,5,6)
	// dot product = 4+10+18 = 32
	// magnitude x = sqrt(1+4+9) = sqrt(14)
	// magnitude y = sqrt(16+25+36) = sqrt(77)
	expectedCosine := 32.0 / (math.Sqrt(14) * math.Sqrt(77))
	if math.Abs(cosine-expectedCosine) > 1e-10 {
		t.Errorf("Cosine similarity: got %.6f, expected %.6f", cosine, expectedCosine)
	}

	// Test Hamming distance
	binary1, _ := array.FromSlice([]float64{1, 0, 1, 0})
	binary2, _ := array.FromSlice([]float64{1, 1, 0, 0})
	hamming, err := HammingDistance(binary1, binary2)
	if err != nil {
		t.Fatal(err)
	}
	expected_hamming := 2 // positions 1 and 2 differ
	if hamming != expected_hamming {
		t.Errorf("Hamming distance: got %d, expected %d", hamming, expected_hamming)
	}

	t.Logf("Distance metrics test passed")
}

func TestStatisticalTests(t *testing.T) {
	// Test one-sample t-test
	// Sample from a normal distribution with mean=10
	data := []float64{9.5, 10.2, 10.1, 9.8, 10.3, 9.9, 10.0, 10.4, 9.7, 10.1}
	arr, _ := array.FromSlice(data)

	result, err := OneSampleTTest(arr, 10.0)
	if err != nil {
		t.Fatal(err)
	}

	// Check that we get reasonable values
	if math.Abs(result.Statistic) > 5.0 {
		t.Errorf("t-statistic seems too large: %.6f", result.Statistic)
	}
	if result.PValue < 0 || result.PValue > 1 {
		t.Errorf("p-value should be between 0 and 1: %.6f", result.PValue)
	}
	if result.DegreesOfFreedom != 9 {
		t.Errorf("Degrees of freedom: got %.0f, expected 9", result.DegreesOfFreedom)
	}

	// Test two-sample t-test
	group1, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
	group2, _ := array.FromSlice([]float64{3, 4, 5, 6, 7})

	twoSampleResult, err := TwoSampleTTest(group1, group2)
	if err != nil {
		t.Fatal(err)
	}

	// Group2 should have higher mean, so t-statistic should be negative
	if twoSampleResult.Statistic > 0 {
		t.Errorf("Expected negative t-statistic, got %.6f", twoSampleResult.Statistic)
	}

	t.Logf("Statistical tests passed: t=%.3f, p=%.3f", result.Statistic, result.PValue)
}

func TestHistogram(t *testing.T) {
	// Create data for histogram
	data := []float64{1, 2, 2, 3, 3, 3, 4, 4, 5}
	arr, _ := array.FromSlice(data)

	// Test with 4 bins
	hist, err := Histogram(arr, 4)
	if err != nil {
		t.Fatal(err)
	}

	// Check that we have the right number of bins
	if hist.Counts.Size() != 4 {
		t.Errorf("Expected 4 bins, got %d", hist.Counts.Size())
	}

	// Check that bin edges are correct
	if hist.BinEdges.Size() != 5 { // n+1 edges for n bins
		t.Errorf("Expected 5 bin edges, got %d", hist.BinEdges.Size())
	}

	// Check that total count matches data size
	totalCount := 0.0
	for i := 0; i < hist.Counts.Size(); i++ {
		totalCount += convertToFloat64(hist.Counts.At(i))
	}
	if int(totalCount) != len(data) {
		t.Errorf("Total count mismatch: got %.0f, expected %d", totalCount, len(data))
	}

	t.Logf("Histogram test passed")
}

func TestMovingAverage(t *testing.T) {
	// Create simple test data
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	arr, _ := array.FromSlice(data)

	// Test moving average with window size 3
	ma, err := MovingAverage(arr, 3)
	if err != nil {
		t.Fatal(err)
	}

	// Should have 8 values (10 - 3 + 1)
	if ma.Size() != 8 {
		t.Errorf("Expected 8 moving averages, got %d", ma.Size())
	}

	// First moving average should be (1+2+3)/3 = 2
	firstMA := convertToFloat64(ma.At(0))
	if math.Abs(firstMA-2.0) > 1e-10 {
		t.Errorf("First moving average: got %.6f, expected 2.0", firstMA)
	}

	// Last moving average should be (8+9+10)/3 = 9
	lastMA := convertToFloat64(ma.At(7))
	if math.Abs(lastMA-9.0) > 1e-10 {
		t.Errorf("Last moving average: got %.6f, expected 9.0", lastMA)
	}

	t.Logf("Moving average test passed")
}

func TestErrorHandling(t *testing.T) {
	// Test nil array handling
	_, err := Skewness(nil)
	if err == nil {
		t.Error("Expected error for nil array in Skewness")
	}

	_, err = EuclideanDistance(nil, nil)
	if err == nil {
		t.Error("Expected error for nil arrays in EuclideanDistance")
	}

	// Test dimension mismatch
	x, _ := array.FromSlice([]float64{1, 2, 3})
	y, _ := array.FromSlice([]float64{1, 2})
	_, err = EuclideanDistance(x, y)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}

	// Test empty array
	empty, _ := array.FromSlice([]float64{})
	_, err = Histogram(empty, 5)
	if err == nil {
		t.Error("Expected error for empty array in Histogram")
	}

	// Test invalid window size
	arr, _ := array.FromSlice([]float64{1, 2, 3})
	_, err = MovingAverage(arr, 0)
	if err == nil {
		t.Error("Expected error for invalid window size")
	}

	_, err = MovingAverage(arr, 5) // Window larger than array
	if err == nil {
		t.Error("Expected error for window size larger than array")
	}

	t.Logf("Error handling tests passed")
}

func TestAdvancedMeans(t *testing.T) {
	// Test geometric mean
	data := []float64{2, 8} // geometric mean should be 4
	arr, _ := array.FromSlice(data)

	geomMean, err := GeometricMean(arr)
	if err != nil {
		t.Fatal(err)
	}
	expected := 4.0 // sqrt(2*8) = sqrt(16) = 4
	if math.Abs(geomMean-expected) > 1e-10 {
		t.Errorf("Geometric mean: got %.6f, expected %.6f", geomMean, expected)
	}

	// Test harmonic mean
	harmMean, err := HarmonicMean(arr)
	if err != nil {
		t.Fatal(err)
	}
	expected = 3.2 // 2 / (1/2 + 1/8) = 2 / (5/8) = 16/5 = 3.2
	if math.Abs(harmMean-expected) > 1e-10 {
		t.Errorf("Harmonic mean: got %.6f, expected %.6f", harmMean, expected)
	}

	// Test RMS
	rms, err := RootMeanSquare(arr)
	if err != nil {
		t.Fatal(err)
	}
	expected = math.Sqrt((4 + 64) / 2) // sqrt((2^2 + 8^2)/2) = sqrt(34)
	if math.Abs(rms-expected) > 1e-10 {
		t.Errorf("RMS: got %.6f, expected %.6f", rms, expected)
	}

	t.Logf("Advanced means test passed")
}

// Benchmark tests
func BenchmarkEuclideanDistance(b *testing.B) {
	size := 1000
	data1 := make([]float64, size)
	data2 := make([]float64, size)
	for i := 0; i < size; i++ {
		data1[i] = float64(i)
		data2[i] = float64(i + 1)
	}

	arr1, _ := array.FromSlice(data1)
	arr2, _ := array.FromSlice(data2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EuclideanDistance(arr1, arr2)
	}
}

func BenchmarkHistogram(b *testing.B) {
	size := 10000
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = math.Sin(float64(i)) * 100
	}

	arr, _ := array.FromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Histogram(arr, 50)
	}
}

func BenchmarkMovingAverage(b *testing.B) {
	size := 10000
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = math.Sin(float64(i) * 0.1)
	}

	arr, _ := array.FromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MovingAverage(arr, 20)
	}
}
