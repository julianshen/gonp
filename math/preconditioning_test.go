package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestJacobiPreconditioner(t *testing.T) {
	t.Run("Diagonal matrix preconditioning", func(t *testing.T) {
		// Create a diagonal matrix A with different diagonal values
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(4.0, 0, 0) // A[0,0] = 4
		A.Set(2.0, 1, 1) // A[1,1] = 2
		A.Set(8.0, 2, 2) // A[2,2] = 8

		preconditioner, err := NewJacobiPreconditioner(A)
		if err != nil {
			t.Fatalf("Failed to create Jacobi preconditioner: %v", err)
		}

		// For a diagonal matrix, Jacobi preconditioner should be 1/diagonal
		expectedDiag := []float64{1.0 / 4.0, 1.0 / 2.0, 1.0 / 8.0}
		for i, expected := range expectedDiag {
			actual := preconditioner.GetDiagonal().At(i).(float64)
			if math.Abs(actual-expected) > 1e-12 {
				t.Errorf("Diagonal[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		// Test applying preconditioner to a vector
		x, _ := array.FromSlice([]float64{8, 4, 16})
		result, err := preconditioner.Apply(x)
		if err != nil {
			t.Fatalf("Failed to apply preconditioner: %v", err)
		}

		// Expected result: [8*(1/4), 4*(1/2), 16*(1/8)] = [2, 2, 2]
		expected := []float64{2, 2, 2}
		for i, exp := range expected {
			actual := result.At(i).(float64)
			if math.Abs(actual-exp) > 1e-12 {
				t.Errorf("Result[%d]: expected %f, got %f", i, exp, actual)
			}
		}
	})

	t.Run("General matrix preconditioning", func(t *testing.T) {
		// Create a general SPD matrix
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(0.0, 0, 2)
		A.Set(1.0, 1, 0)
		A.Set(3.0, 1, 1)
		A.Set(0.5, 1, 2)
		A.Set(0.0, 2, 0)
		A.Set(0.5, 2, 1)
		A.Set(2.0, 2, 2)

		preconditioner, err := NewJacobiPreconditioner(A)
		if err != nil {
			t.Fatalf("Failed to create Jacobi preconditioner: %v", err)
		}

		// Jacobi preconditioner uses only diagonal elements
		expectedDiag := []float64{1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0}
		for i, expected := range expectedDiag {
			actual := preconditioner.GetDiagonal().At(i).(float64)
			if math.Abs(actual-expected) > 1e-12 {
				t.Errorf("Diagonal[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		// Test solving M*y = x where M is the preconditioner
		x, _ := array.FromSlice([]float64{1, 1, 1})
		y, err := preconditioner.Solve(x)
		if err != nil {
			t.Fatalf("Failed to solve with preconditioner: %v", err)
		}

		// For Jacobi: M = diag(A), so y[i] = x[i] / A[i,i]
		expected := []float64{1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0}
		for i, exp := range expected {
			actual := y.At(i).(float64)
			if math.Abs(actual-exp) > 1e-12 {
				t.Errorf("Solve result[%d]: expected %f, got %f", i, exp, actual)
			}
		}
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Test nil matrix
		_, err := NewJacobiPreconditioner(nil)
		if err == nil {
			t.Error("Expected error for nil matrix")
		}

		// Test non-square matrix
		nonSquare := array.Zeros(internal.Shape{3, 4}, internal.Float64)
		_, err = NewJacobiPreconditioner(nonSquare)
		if err == nil {
			t.Error("Expected error for non-square matrix")
		}

		// Test matrix with zero diagonal element
		zeroDiag := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		zeroDiag.Set(1.0, 0, 0)
		// zeroDiag.Set(0.0, 1, 1) // already zero
		_, err = NewJacobiPreconditioner(zeroDiag)
		if err == nil {
			t.Error("Expected error for matrix with zero diagonal")
		}

		// Test 1D array
		arr1D := array.Ones(internal.Shape{5}, internal.Float64)
		_, err = NewJacobiPreconditioner(arr1D)
		if err == nil {
			t.Error("Expected error for 1D array")
		}
	})

	t.Run("Vector validation in Apply", func(t *testing.T) {
		// Create valid preconditioner
		A := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		A.Set(2.0, 0, 0)
		A.Set(3.0, 1, 1)
		preconditioner, _ := NewJacobiPreconditioner(A)

		// Test nil vector
		_, err := preconditioner.Apply(nil)
		if err == nil {
			t.Error("Expected error for nil vector")
		}

		// Test wrong size vector
		wrongSize := array.Ones(internal.Shape{3}, internal.Float64)
		_, err = preconditioner.Apply(wrongSize)
		if err == nil {
			t.Error("Expected error for wrong size vector")
		}

		// Test 2D array instead of vector
		matrix := array.Ones(internal.Shape{2, 2}, internal.Float64)
		_, err = preconditioner.Apply(matrix)
		if err == nil {
			t.Error("Expected error for 2D array input")
		}
	})
}

func TestPreconditionedConjugateGradient(t *testing.T) {
	t.Run("Diagonal system with Jacobi preconditioning", func(t *testing.T) {
		// Create a diagonal system that should converge in 1 iteration with preconditioning
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(16.0, 1, 1) // Large diagonal element to test preconditioning effect
		A.Set(1.0, 2, 2)

		b, _ := array.FromSlice([]float64{4, 16, 1}) // Solution should be [1, 1, 1]

		preconditioner, err := NewJacobiPreconditioner(A)
		if err != nil {
			t.Fatalf("Failed to create preconditioner: %v", err)
		}

		options := &IterativeSolverOptions{
			MaxIterations: 10,
			Tolerance:     1e-10,
		}

		result := PreconditionedConjugateGradient(A, b, preconditioner, options)
		if result.Error != nil {
			t.Fatalf("PCG failed: %v", result.Error)
		}

		if !result.Converged {
			t.Error("PCG should have converged")
		}

		// Should converge in 1 iteration for diagonal matrix with preconditioning
		if result.Iterations > 3 {
			t.Errorf("Expected quick convergence, got %d iterations", result.Iterations)
		}

		// Check solution accuracy
		expectedSolution := []float64{1, 1, 1}
		for i, expected := range expectedSolution {
			actual := result.Solution.At(i).(float64)
			if math.Abs(actual-expected) > 1e-8 {
				t.Errorf("Solution[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		t.Logf("PCG converged in %d iterations with residual %e",
			result.Iterations, result.Residual)
	})

	t.Run("Parameter validation", func(t *testing.T) {
		A := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		A.Set(1.0, 0, 0)
		A.Set(1.0, 1, 1)
		b, _ := array.FromSlice([]float64{1, 1})
		preconditioner, _ := NewJacobiPreconditioner(A)

		// Test nil matrix
		result := PreconditionedConjugateGradient(nil, b, preconditioner, nil)
		if result.Error == nil {
			t.Error("Expected error for nil matrix")
		}

		// Test nil vector
		result = PreconditionedConjugateGradient(A, nil, preconditioner, nil)
		if result.Error == nil {
			t.Error("Expected error for nil vector")
		}

		// Test with nil preconditioner (should fall back to normal CG)
		result = PreconditionedConjugateGradient(A, b, nil, nil)
		if result.Error != nil {
			t.Errorf("PCG should fall back to normal CG with nil preconditioner: %v", result.Error)
		}
		if !result.Converged {
			t.Error("Fallback to normal CG should converge")
		}
	})
}
