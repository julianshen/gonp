package stats

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// Test correlation functions
func TestCorrelation(t *testing.T) {
	t.Run("PearsonCorrelation", func(t *testing.T) {
		// Perfect positive correlation
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		corr, err := Correlation(xArr, yArr)
		if err != nil {
			t.Fatalf("Correlation failed: %v", err)
		}

		expected := 1.0
		if math.Abs(corr-expected) > 1e-10 {
			t.Errorf("Expected correlation %f, got %f", expected, corr)
		}
	})

	t.Run("NegativeCorrelation", func(t *testing.T) {
		// Perfect negative correlation
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{5.0, 4.0, 3.0, 2.0, 1.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		corr, err := Correlation(xArr, yArr)
		if err != nil {
			t.Fatalf("Correlation failed: %v", err)
		}

		expected := -1.0
		if math.Abs(corr-expected) > 1e-10 {
			t.Errorf("Expected correlation %f, got %f", expected, corr)
		}
	})

	t.Run("ZeroCorrelation", func(t *testing.T) {
		// No correlation - the actual correlation for this data is 0.8, so let's test that
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{1.0, 3.0, 2.0, 5.0, 4.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		corr, err := Correlation(xArr, yArr)
		if err != nil {
			t.Fatalf("Correlation failed: %v", err)
		}

		// Expected correlation is 0.8 for this data
		expected := 0.8
		if math.Abs(corr-expected) > 1e-10 {
			t.Errorf("Expected correlation %f, got %f", expected, corr)
		}
	})
}

// Test covariance functions
func TestCovariance(t *testing.T) {
	t.Run("Covariance", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		cov, err := Covariance(xArr, yArr)
		if err != nil {
			t.Fatalf("Covariance failed: %v", err)
		}

		// Covariance should be positive for positive correlation
		if cov <= 0 {
			t.Errorf("Expected positive covariance, got %f", cov)
		}

		// Test known covariance calculation
		// For x = [1,2,3,4,5] and y = [2,4,6,8,10]
		// Cov(x,y) = 5.0
		expected := 5.0
		if math.Abs(cov-expected) > 1e-10 {
			t.Errorf("Expected covariance %f, got %f", expected, cov)
		}
	})

	t.Run("CovarianceMatrix", func(t *testing.T) {
		// Create multiple series
		s1, _ := series.FromSlice([]float64{1.0, 2.0, 3.0, 4.0}, nil, "s1")
		s2, _ := series.FromSlice([]float64{2.0, 4.0, 6.0, 8.0}, nil, "s2")
		s3, _ := series.FromSlice([]float64{1.0, 3.0, 2.0, 4.0}, nil, "s3")

		seriesList := []*series.Series{s1, s2, s3}

		covMatrix, err := CovarianceMatrix(seriesList)
		if err != nil {
			t.Fatalf("CovarianceMatrix failed: %v", err)
		}

		// Verify matrix dimensions
		expectedShape := []int{3, 3}
		if !covMatrix.Shape().Equal(expectedShape) {
			t.Errorf("Expected covariance matrix shape %v, got %v", expectedShape, covMatrix.Shape())
		}

		// Diagonal elements should be variances (positive)
		if covMatrix.At(0, 0).(float64) <= 0 {
			t.Errorf("Expected positive variance on diagonal, got %f", covMatrix.At(0, 0))
		}

		// Matrix should be symmetric
		if covMatrix.At(0, 1) != covMatrix.At(1, 0) {
			t.Errorf("Covariance matrix should be symmetric")
		}
	})
}

// Test correlation matrix
func TestCorrelationMatrix(t *testing.T) {
	t.Run("CorrelationMatrix", func(t *testing.T) {
		// Create test series
		s1, _ := series.FromSlice([]float64{1.0, 2.0, 3.0, 4.0, 5.0}, nil, "s1")
		s2, _ := series.FromSlice([]float64{2.0, 4.0, 6.0, 8.0, 10.0}, nil, "s2")
		s3, _ := series.FromSlice([]float64{5.0, 4.0, 3.0, 2.0, 1.0}, nil, "s3")

		seriesList := []*series.Series{s1, s2, s3}

		corrMatrix, err := CorrelationMatrix(seriesList)
		if err != nil {
			t.Fatalf("CorrelationMatrix failed: %v", err)
		}

		// Verify matrix dimensions
		expectedShape := []int{3, 3}
		if !corrMatrix.Shape().Equal(expectedShape) {
			t.Errorf("Expected correlation matrix shape %v, got %v", expectedShape, corrMatrix.Shape())
		}

		// Diagonal elements should be 1.0
		for i := 0; i < 3; i++ {
			diag := corrMatrix.At(i, i).(float64)
			if math.Abs(diag-1.0) > 1e-10 {
				t.Errorf("Expected diagonal element %d to be 1.0, got %f", i, diag)
			}
		}

		// s1 and s2 should have perfect positive correlation
		corr12 := corrMatrix.At(0, 1).(float64)
		if math.Abs(corr12-1.0) > 1e-10 {
			t.Errorf("Expected s1-s2 correlation 1.0, got %f", corr12)
		}

		// s1 and s3 should have perfect negative correlation
		corr13 := corrMatrix.At(0, 2).(float64)
		if math.Abs(corr13-(-1.0)) > 1e-10 {
			t.Errorf("Expected s1-s3 correlation -1.0, got %f", corr13)
		}

		// Matrix should be symmetric
		if corrMatrix.At(0, 1) != corrMatrix.At(1, 0) {
			t.Errorf("Correlation matrix should be symmetric")
		}
	})
}

// Test rank correlation
func TestRankCorrelation(t *testing.T) {
	t.Run("SpearmanCorrelation", func(t *testing.T) {
		// Non-linear but monotonic relationship
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{1.0, 4.0, 9.0, 16.0, 25.0} // y = x^2

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		spearman, err := SpearmanCorrelation(xArr, yArr)
		if err != nil {
			t.Fatalf("SpearmanCorrelation failed: %v", err)
		}

		// Should be perfect rank correlation (1.0)
		expected := 1.0
		if math.Abs(spearman-expected) > 1e-10 {
			t.Errorf("Expected Spearman correlation %f, got %f", expected, spearman)
		}
	})

	t.Run("KendallTau", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{1.0, 3.0, 2.0, 4.0, 5.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		tau, err := KendallTau(xArr, yArr)
		if err != nil {
			t.Fatalf("KendallTau failed: %v", err)
		}

		// Should be positive but less than 1 due to one inversion
		if tau <= 0 || tau >= 1 {
			t.Errorf("Expected 0 < tau < 1, got %f", tau)
		}
	})
}

// Test error conditions
func TestCorrelationErrors(t *testing.T) {
	t.Run("DifferentLengths", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0}
		y := []float64{1.0, 2.0, 3.0, 4.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := Correlation(xArr, yArr)
		if err == nil {
			t.Error("Expected error for arrays of different lengths")
		}
	})

	t.Run("ConstantValues", func(t *testing.T) {
		x := []float64{1.0, 1.0, 1.0, 1.0}
		y := []float64{2.0, 3.0, 4.0, 5.0}

		xArr, _ := array.FromSlice(x)
		yArr, _ := array.FromSlice(y)

		_, err := Correlation(xArr, yArr)
		if err == nil {
			t.Error("Expected error for constant values (zero variance)")
		}
	})

	t.Run("EmptyArrays", func(t *testing.T) {
		// Create empty arrays using array.Empty instead of FromSlice for empty data
		xArr := array.Empty(internal.Shape{0}, internal.Float64)
		yArr := array.Empty(internal.Shape{0}, internal.Float64)

		_, err := Correlation(xArr, yArr)
		if err == nil {
			t.Error("Expected error for empty arrays")
		}
	})
}
