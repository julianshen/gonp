package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestConjugateGradient(t *testing.T) {
	t.Run("Simple symmetric positive definite system", func(t *testing.T) {
		// Solve: [4 1] [x1] = [1]
		//        [1 3] [x2]   [2]
		// Solution: x1 = 1/11, x2 = 7/11
		A := array.Empty(internal.Shape{2, 2}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(3.0, 1, 1)

		b := array.Empty(internal.Shape{2}, internal.Float64)
		b.Set(1.0, 0)
		b.Set(2.0, 1)

		options := &IterativeSolverOptions{
			MaxIterations: 100,
			Tolerance:     1e-10,
			Verbose:       false,
		}

		result := ConjugateGradient(A, b, options)

		if result.Error != nil {
			t.Fatalf("CG failed: %v", result.Error)
		}

		if !result.Converged {
			t.Errorf("CG did not converge after %d iterations", result.Iterations)
		}

		// Check solution
		expectedX1 := 1.0 / 11.0
		expectedX2 := 7.0 / 11.0

		x1 := result.Solution.At(0).(float64)
		x2 := result.Solution.At(1).(float64)

		if math.Abs(x1-expectedX1) > 1e-10 {
			t.Errorf("x1: expected %f, got %f", expectedX1, x1)
		}
		if math.Abs(x2-expectedX2) > 1e-10 {
			t.Errorf("x2: expected %f, got %f", expectedX2, x2)
		}

		t.Logf("CG converged in %d iterations with residual %e", result.Iterations, result.Residual)
	})

	t.Run("Larger symmetric system", func(t *testing.T) {
		// Create a 5x5 symmetric positive definite matrix
		n := 5
		A := createSPDMatrix(n, t)

		// Create a known solution and compute b = A*x
		xTrue := array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			xTrue.Set(float64(i+1), i) // x = [1, 2, 3, 4, 5]
		}

		b, err := Dot(A, xTrue.Reshape(internal.Shape{n, 1}))
		if err != nil {
			t.Fatalf("Failed to compute b: %v", err)
		}
		bVec := b.Reshape(internal.Shape{n})

		options := &IterativeSolverOptions{
			MaxIterations: 1000,
			Tolerance:     1e-8,
			Verbose:       false,
		}

		result := ConjugateGradient(A, bVec, options)

		if result.Error != nil {
			t.Fatalf("CG failed: %v", result.Error)
		}

		if !result.Converged {
			t.Errorf("CG did not converge after %d iterations", result.Iterations)
		}

		// Check solution accuracy
		for i := 0; i < n; i++ {
			expected := xTrue.At(i).(float64)
			actual := result.Solution.At(i).(float64)
			if math.Abs(actual-expected) > 1e-6 {
				t.Errorf("x[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		t.Logf("CG converged in %d iterations with residual %e", result.Iterations, result.Residual)
	})

	t.Run("Error handling", func(t *testing.T) {
		// Test with nil inputs
		result := ConjugateGradient(nil, nil, nil)
		if result.Error == nil {
			t.Error("Expected error for nil inputs")
		}

		// Test with non-square matrix
		A := array.Empty(internal.Shape{2, 3}, internal.Float64)
		b := array.Empty(internal.Shape{2}, internal.Float64)
		result = ConjugateGradient(A, b, nil)
		if result.Error == nil {
			t.Error("Expected error for non-square matrix")
		}

		// Test with mismatched dimensions
		A = array.Empty(internal.Shape{2, 2}, internal.Float64)
		b = array.Empty(internal.Shape{3}, internal.Float64)
		result = ConjugateGradient(A, b, nil)
		if result.Error == nil {
			t.Error("Expected error for mismatched dimensions")
		}
	})
}

func TestGMRES(t *testing.T) {
	t.Run("Simple non-symmetric system", func(t *testing.T) {
		// Solve: [2 1] [x1] = [3]
		//        [1 1] [x2]   [2]
		// Solution: x1 = 1, x2 = 1
		A := array.Empty(internal.Shape{2, 2}, internal.Float64)
		A.Set(2.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(1.0, 1, 1)

		b := array.Empty(internal.Shape{2}, internal.Float64)
		b.Set(3.0, 0)
		b.Set(2.0, 1)

		options := &IterativeSolverOptions{
			MaxIterations: 100,
			Tolerance:     1e-1,
			Verbose:       false,
		}

		result := GMRES(A, b, 10, options) // restart=10

		if result.Error != nil {
			t.Fatalf("GMRES failed: %v", result.Error)
		}

		if !result.Converged {
			t.Errorf("GMRES did not converge after %d iterations", result.Iterations)
		}

		// Check solution
		x1 := result.Solution.At(0).(float64)
		x2 := result.Solution.At(1).(float64)

		if math.Abs(x1-1.0) > 1e-1 {
			t.Errorf("x1: expected 1.0, got %f", x1)
		}
		if math.Abs(x2-1.0) > 1e-1 {
			t.Errorf("x2: expected 1.0, got %f", x2)
		}

		t.Logf("GMRES converged in %d iterations with residual %e", result.Iterations, result.Residual)
	})

	t.Run("Larger general system", func(t *testing.T) {
		// Create a 4x4 general matrix
		n := 4
		A := array.Empty(internal.Shape{n, n}, internal.Float64)

		// Create a non-symmetric matrix
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					A.Set(float64(n+1), i, j) // Diagonal dominant
				} else {
					A.Set(float64(i-j+1)*0.5, i, j)
				}
			}
		}

		// Create a known solution and compute b = A*x
		xTrue := array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			xTrue.Set(float64(i+1)*0.5, i)
		}

		b, err := Dot(A, xTrue.Reshape(internal.Shape{n, 1}))
		if err != nil {
			t.Fatalf("Failed to compute b: %v", err)
		}
		bVec := b.Reshape(internal.Shape{n})

		options := &IterativeSolverOptions{
			MaxIterations: 200,
			Tolerance:     1e-3,
			Verbose:       false,
		}

		result := GMRES(A, bVec, 20, options)

		if result.Error != nil {
			t.Fatalf("GMRES failed: %v", result.Error)
		}

		if !result.Converged {
			t.Errorf("GMRES did not converge after %d iterations", result.Iterations)
		}

		// Check solution accuracy
		for i := 0; i < n; i++ {
			expected := xTrue.At(i).(float64)
			actual := result.Solution.At(i).(float64)
			if math.Abs(actual-expected) > 1e-6 {
				t.Errorf("x[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		t.Logf("GMRES converged in %d iterations with residual %e", result.Iterations, result.Residual)
	})

	t.Run("Error handling", func(t *testing.T) {
		// Test with nil inputs
		result := GMRES(nil, nil, 10, nil)
		if result.Error == nil {
			t.Error("Expected error for nil inputs")
		}

		// Test with invalid restart parameter
		A := array.Empty(internal.Shape{2, 2}, internal.Float64)
		b := array.Empty(internal.Shape{2}, internal.Float64)
		result = GMRES(A, b, 0, nil) // Invalid restart
		// Should use default restart=30, so no error expected here
	})
}

func TestBiCGSTAB(t *testing.T) {
	t.Run("Simple non-symmetric system", func(t *testing.T) {
		// Solve: [3 1] [x1] = [4]
		//        [1 2] [x2]   [3]
		// Solution: x1 = 1, x2 = 1
		A := array.Empty(internal.Shape{2, 2}, internal.Float64)
		A.Set(3.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(2.0, 1, 1)

		b := array.Empty(internal.Shape{2}, internal.Float64)
		b.Set(4.0, 0)
		b.Set(3.0, 1)

		options := &IterativeSolverOptions{
			MaxIterations: 100,
			Tolerance:     1e-8,
			Verbose:       false,
		}

		result := BiCGSTAB(A, b, options)

		if result.Error != nil {
			t.Fatalf("BiCGSTAB failed: %v", result.Error)
		}

		if !result.Converged {
			t.Errorf("BiCGSTAB did not converge after %d iterations", result.Iterations)
		}

		// Check solution
		x1 := result.Solution.At(0).(float64)
		x2 := result.Solution.At(1).(float64)

		if math.Abs(x1-1.0) > 1e-8 {
			t.Errorf("x1: expected 1.0, got %f", x1)
		}
		if math.Abs(x2-1.0) > 1e-8 {
			t.Errorf("x2: expected 1.0, got %f", x2)
		}

		t.Logf("BiCGSTAB converged in %d iterations with residual %e", result.Iterations, result.Residual)
	})

	t.Run("Larger general system", func(t *testing.T) {
		// Create a 6x6 general matrix
		n := 6
		A := array.Empty(internal.Shape{n, n}, internal.Float64)

		// Create a diagonally dominant matrix
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					A.Set(10.0, i, j)
				} else if math.Abs(float64(i-j)) == 1 {
					A.Set(1.0, i, j)
				} else {
					A.Set(0.1*float64(i+j+1), i, j)
				}
			}
		}

		// Create a known solution and compute b = A*x
		xTrue := array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			xTrue.Set(math.Sin(float64(i+1)), i)
		}

		b, err := Dot(A, xTrue.Reshape(internal.Shape{n, 1}))
		if err != nil {
			t.Fatalf("Failed to compute b: %v", err)
		}
		bVec := b.Reshape(internal.Shape{n})

		options := &IterativeSolverOptions{
			MaxIterations: 500,
			Tolerance:     1e-8,
			Verbose:       false,
		}

		result := BiCGSTAB(A, bVec, options)

		if result.Error != nil {
			t.Fatalf("BiCGSTAB failed: %v", result.Error)
		}

		if !result.Converged {
			t.Errorf("BiCGSTAB did not converge after %d iterations", result.Iterations)
		}

		// Check solution accuracy
		for i := 0; i < n; i++ {
			expected := xTrue.At(i).(float64)
			actual := result.Solution.At(i).(float64)
			if math.Abs(actual-expected) > 1e-6 {
				t.Errorf("x[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		t.Logf("BiCGSTAB converged in %d iterations with residual %e", result.Iterations, result.Residual)
	})

	t.Run("Error handling", func(t *testing.T) {
		// Test with nil inputs
		result := BiCGSTAB(nil, nil, nil)
		if result.Error == nil {
			t.Error("Expected error for nil inputs")
		}

		// Test with non-square matrix
		A := array.Empty(internal.Shape{3, 2}, internal.Float64)
		b := array.Empty(internal.Shape{3}, internal.Float64)
		result = BiCGSTAB(A, b, nil)
		if result.Error == nil {
			t.Error("Expected error for non-square matrix")
		}
	})
}

func TestIterativeSolverComparison(t *testing.T) {
	t.Run("Compare solver performance", func(t *testing.T) {
		// Create a moderately sized test system
		n := 10
		A := createDiagonallyDominantMatrix(n, t)

		// Create a known solution
		xTrue := array.Empty(internal.Shape{n}, internal.Float64)
		for i := 0; i < n; i++ {
			xTrue.Set(math.Cos(float64(i)), i)
		}

		b, err := Dot(A, xTrue.Reshape(internal.Shape{n, 1}))
		if err != nil {
			t.Fatalf("Failed to compute b: %v", err)
		}
		bVec := b.Reshape(internal.Shape{n})

		options := &IterativeSolverOptions{
			MaxIterations: 1000,
			Tolerance:     1e-6,
			Verbose:       false,
		}

		// Test Conjugate Gradient (should work if matrix is SPD)
		cgResult := ConjugateGradient(A, bVec, options)
		if cgResult.Error != nil {
			t.Logf("CG failed (expected for non-SPD): %v", cgResult.Error)
		} else {
			t.Logf("CG: %d iterations, residual %e", cgResult.Iterations, cgResult.Residual)
		}

		// Test GMRES
		gmresResult := GMRES(A, bVec, 20, options)
		if gmresResult.Error != nil {
			t.Logf("GMRES failed (expected for some systems): %v", gmresResult.Error)
		} else {
			t.Logf("GMRES: %d iterations, residual %e", gmresResult.Iterations, gmresResult.Residual)
		}

		// Test BiCGSTAB
		bicgstabResult := BiCGSTAB(A, bVec, options)
		if bicgstabResult.Error != nil {
			t.Errorf("BiCGSTAB failed: %v", bicgstabResult.Error)
		} else {
			t.Logf("BiCGSTAB: %d iterations, residual %e", bicgstabResult.Iterations, bicgstabResult.Residual)
		}

		// Compare solutions (all successful methods should give similar results)
		if gmresResult.Converged && bicgstabResult.Converged {
			for i := 0; i < n; i++ {
				gmresVal := gmresResult.Solution.At(i).(float64)
				bicgstabVal := bicgstabResult.Solution.At(i).(float64)
				if math.Abs(gmresVal-bicgstabVal) > 1e-8 {
					t.Errorf("Solution mismatch at index %d: GMRES=%f, BiCGSTAB=%f", i, gmresVal, bicgstabVal)
				}
			}
		}
	})
}

func TestIterativeSolverOptions(t *testing.T) {
	t.Run("Custom options", func(t *testing.T) {
		A := array.Empty(internal.Shape{3, 3}, internal.Float64)
		A.Set(5.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(0.0, 0, 2)
		A.Set(1.0, 1, 0)
		A.Set(4.0, 1, 1)
		A.Set(1.0, 1, 2)
		A.Set(0.0, 2, 0)
		A.Set(1.0, 2, 1)
		A.Set(3.0, 2, 2)

		b := array.Empty(internal.Shape{3}, internal.Float64)
		b.Set(6.0, 0)
		b.Set(6.0, 1)
		b.Set(4.0, 2)

		// Test with very strict tolerance
		options := &IterativeSolverOptions{
			MaxIterations: 5,     // Very few iterations
			Tolerance:     1e-15, // Very strict tolerance
			Verbose:       false,
		}

		result := ConjugateGradient(A, b, options)
		// Should not converge with so few iterations
		if result.Converged {
			t.Logf("Unexpectedly converged in %d iterations", result.Iterations)
		} else {
			t.Logf("As expected, did not converge in %d iterations", options.MaxIterations)
		}
	})

	t.Run("Default options", func(t *testing.T) {
		// Test that default options work
		A := array.Empty(internal.Shape{2, 2}, internal.Float64)
		A.Set(2.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(2.0, 1, 1)

		b := array.Empty(internal.Shape{2}, internal.Float64)
		b.Set(3.0, 0)
		b.Set(3.0, 1)

		result := ConjugateGradient(A, b, nil) // Use default options
		if result.Error != nil {
			t.Errorf("Failed with default options: %v", result.Error)
		}
	})
}

func TestHelperFunctions(t *testing.T) {
	t.Run("iterativeDot", func(t *testing.T) {
		a := array.Empty(internal.Shape{3}, internal.Float64)
		a.Set(1.0, 0)
		a.Set(2.0, 1)
		a.Set(3.0, 2)

		b := array.Empty(internal.Shape{3}, internal.Float64)
		b.Set(4.0, 0)
		b.Set(5.0, 1)
		b.Set(6.0, 2)

		result, err := iterativeDot(a, b)
		if err != nil {
			t.Fatalf("iterativeDot failed: %v", err)
		}

		expected := 1*4 + 2*5 + 3*6 // = 4 + 10 + 18 = 32
		if math.Abs(result-float64(expected)) > 1e-15 {
			t.Errorf("Expected %d, got %f", expected, result)
		}
	})

	t.Run("iterativeNorm", func(t *testing.T) {
		v := array.Empty(internal.Shape{3}, internal.Float64)
		v.Set(3.0, 0)
		v.Set(4.0, 1)
		v.Set(0.0, 2)

		result, err := iterativeNorm(v)
		if err != nil {
			t.Fatalf("iterativeNorm failed: %v", err)
		}

		expected := 5.0 // sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5
		if math.Abs(result-expected) > 1e-15 {
			t.Errorf("Expected %f, got %f", expected, result)
		}
	})

	t.Run("vectorAdd", func(t *testing.T) {
		a := array.Empty(internal.Shape{2}, internal.Float64)
		a.Set(1.0, 0)
		a.Set(2.0, 1)

		b := array.Empty(internal.Shape{2}, internal.Float64)
		b.Set(3.0, 0)
		b.Set(4.0, 1)

		result, err := vectorAdd(a, b)
		if err != nil {
			t.Fatalf("vectorAdd failed: %v", err)
		}

		if result.At(0).(float64) != 4.0 || result.At(1).(float64) != 6.0 {
			t.Errorf("Expected [4, 6], got [%f, %f]", result.At(0).(float64), result.At(1).(float64))
		}
	})

	t.Run("vectorScale", func(t *testing.T) {
		v := array.Empty(internal.Shape{3}, internal.Float64)
		v.Set(1.0, 0)
		v.Set(2.0, 1)
		v.Set(3.0, 2)

		result, err := vectorScale(2.5, v)
		if err != nil {
			t.Fatalf("vectorScale failed: %v", err)
		}

		expected := []float64{2.5, 5.0, 7.5}
		for i, exp := range expected {
			if math.Abs(result.At(i).(float64)-exp) > 1e-15 {
				t.Errorf("At index %d: expected %f, got %f", i, exp, result.At(i).(float64))
			}
		}
	})
}

// Helper function to create a symmetric positive definite matrix
func createSPDMatrix(n int, t *testing.T) *array.Array {
	// Create A = B^T * B where B is random, ensuring A is SPD
	A := array.Empty(internal.Shape{n, n}, internal.Float64)

	// Simple approach: create diagonally dominant symmetric matrix
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				A.Set(float64(n+10), i, j) // Strong diagonal
			} else {
				val := 1.0 / float64(math.Abs(float64(i-j))+2) // Small off-diagonal
				A.Set(val, i, j)
				A.Set(val, j, i) // Ensure symmetry
			}
		}
	}

	return A
}

// Helper function to create a diagonally dominant matrix
func createDiagonallyDominantMatrix(n int, t *testing.T) *array.Array {
	A := array.Empty(internal.Shape{n, n}, internal.Float64)

	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			if i != j {
				val := 0.1 * float64(i+j+1)
				A.Set(val, i, j)
				sum += math.Abs(val)
			}
		}
		// Make diagonal element larger than sum of off-diagonal elements
		A.Set(sum+10.0, i, i)
	}

	return A
}

// Benchmark tests for comparing iterative solver performance
func BenchmarkIterativeSolvers(b *testing.B) {
	// Create a test system
	n := 100
	A := createDiagonallyDominantMatrix(n, nil)

	xTrue := array.Empty(internal.Shape{n}, internal.Float64)
	for i := 0; i < n; i++ {
		xTrue.Set(float64(i+1), i)
	}

	bMat, _ := Dot(A, xTrue.Reshape(internal.Shape{n, 1}))
	bVec := bMat.Reshape(internal.Shape{n})

	options := &IterativeSolverOptions{
		MaxIterations: 1000,
		Tolerance:     1e-8,
		Verbose:       false,
	}

	b.Run("ConjugateGradient", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			result := ConjugateGradient(A, bVec, options)
			if result.Error != nil && result.Converged {
				b.Logf("CG: %d iterations, residual %e", result.Iterations, result.Residual)
			}
		}
	})

	b.Run("GMRES", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			result := GMRES(A, bVec, 30, options)
			if result.Converged {
				b.Logf("GMRES: %d iterations, residual %e", result.Iterations, result.Residual)
			}
		}
	})

	b.Run("BiCGSTAB", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			result := BiCGSTAB(A, bVec, options)
			if result.Converged {
				b.Logf("BiCGSTAB: %d iterations, residual %e", result.Iterations, result.Residual)
			}
		}
	})
}
