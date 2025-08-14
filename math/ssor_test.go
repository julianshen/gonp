package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestSSORPreconditioner(t *testing.T) {
	t.Run("Simple symmetric matrix SSOR", func(t *testing.T) {
		// Create a simple symmetric positive definite matrix
		//   4  1  1
		//   1  4  1
		//   1  1  4
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 0, 2)
		A.Set(1.0, 1, 0)
		A.Set(4.0, 1, 1)
		A.Set(1.0, 1, 2)
		A.Set(1.0, 2, 0)
		A.Set(1.0, 2, 1)
		A.Set(4.0, 2, 2)

		// Use optimal relaxation parameter ω = 1.0 for this test
		omega := 1.0
		preconditioner, err := NewSSORPreconditioner(A, omega)
		if err != nil {
			t.Fatalf("Failed to create SSOR preconditioner: %v", err)
		}

		// Check that the preconditioner was created successfully
		if preconditioner.GetOmega() != omega {
			t.Errorf("Expected omega %f, got %f", omega, preconditioner.GetOmega())
		}

		// Test applying the preconditioner to a vector
		b, _ := array.FromSlice([]float64{6, 6, 6}) // For solution [1, 1, 1]
		y, err := preconditioner.Apply(b)
		if err != nil {
			t.Fatalf("Failed to apply SSOR preconditioner: %v", err)
		}

		// Result should be reasonable (not testing exact values due to iterative nature)
		if y == nil || y.Size() != 3 {
			t.Error("Applied preconditioner should return vector of size 3")
		}

		t.Logf("SSOR(ω=%.1f) applied to [6,6,6], result: [%.3f, %.3f, %.3f]",
			omega, y.At(0).(float64), y.At(1).(float64), y.At(2).(float64))
	})

	t.Run("SSOR with different relaxation parameters", func(t *testing.T) {
		// Create a test matrix
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(3.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(3.0, 1, 1)
		A.Set(1.0, 1, 2)
		A.Set(1.0, 2, 1)
		A.Set(3.0, 2, 2)

		b, _ := array.FromSlice([]float64{1, 2, 3})

		// Test different omega values
		omegaValues := []float64{0.5, 1.0, 1.5}
		results := make([]*array.Array, len(omegaValues))

		for i, omega := range omegaValues {
			preconditioner, err := NewSSORPreconditioner(A, omega)
			if err != nil {
				t.Fatalf("Failed to create SSOR preconditioner with ω=%f: %v", omega, err)
			}

			result, err := preconditioner.Apply(b)
			if err != nil {
				t.Fatalf("Failed to apply SSOR(ω=%f): %v", omega, err)
			}
			results[i] = result

			t.Logf("SSOR(ω=%.1f) result: [%.3f, %.3f, %.3f]", omega,
				result.At(0).(float64), result.At(1).(float64), result.At(2).(float64))
		}

		// Results should be different for different omega values
		for i := 0; i < len(results)-1; i++ {
			different := false
			for j := 0; j < 3; j++ {
				if math.Abs(results[i].At(j).(float64)-results[i+1].At(j).(float64)) > 1e-10 {
					different = true
					break
				}
			}
			if !different {
				t.Errorf("SSOR results should differ for different omega values")
			}
		}
	})

	t.Run("SSOR convergence properties", func(t *testing.T) {
		// Test SSOR as an iterative method (multiple applications)
		A := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(4.0, 1, 1)

		omega := 1.2 // Slightly over-relaxed
		preconditioner, err := NewSSORPreconditioner(A, omega)
		if err != nil {
			t.Fatalf("Failed to create SSOR preconditioner: %v", err)
		}

		// Test convergence by applying multiple times
		x, _ := array.FromSlice([]float64{5, 5}) // Target solution [1, 1]

		for iter := 0; iter < 3; iter++ {
			newX, err := preconditioner.Apply(x)
			if err != nil {
				t.Fatalf("SSOR application failed at iteration %d: %v", iter, err)
			}

			t.Logf("SSOR iteration %d: [%.3f, %.3f]", iter,
				newX.At(0).(float64), newX.At(1).(float64))

			x = newX
		}

		// Should be moving towards the solution
		x0 := x.At(0).(float64)
		x1 := x.At(1).(float64)
		if math.Abs(x0-1.0) > 2.0 || math.Abs(x1-1.0) > 2.0 {
			t.Errorf("SSOR should be converging towards [1, 1], got [%.3f, %.3f]", x0, x1)
		}
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Test nil matrix
		_, err := NewSSORPreconditioner(nil, 1.0)
		if err == nil {
			t.Error("Expected error for nil matrix")
		}

		// Test non-square matrix
		nonSquare := array.Zeros(internal.Shape{3, 4}, internal.Float64)
		_, err = NewSSORPreconditioner(nonSquare, 1.0)
		if err == nil {
			t.Error("Expected error for non-square matrix")
		}

		// Test invalid omega values
		validMatrix := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		validMatrix.Set(2.0, 0, 0)
		validMatrix.Set(1.0, 0, 1)
		validMatrix.Set(1.0, 1, 0)
		validMatrix.Set(2.0, 1, 1)

		// Test omega <= 0
		_, err = NewSSORPreconditioner(validMatrix, 0.0)
		if err == nil {
			t.Error("Expected error for omega = 0")
		}

		_, err = NewSSORPreconditioner(validMatrix, -0.5)
		if err == nil {
			t.Error("Expected error for negative omega")
		}

		// Test omega >= 2 (should work but warn it may not converge)
		_, err = NewSSORPreconditioner(validMatrix, 2.0)
		if err == nil {
			t.Logf("SSOR with ω=2.0 created (may not converge in practice)")
		}

		// Test zero diagonal
		zeroDiag := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		zeroDiag.Set(0.0, 0, 0)
		zeroDiag.Set(1.0, 0, 1)
		zeroDiag.Set(1.0, 1, 0)
		zeroDiag.Set(2.0, 1, 1)
		_, err = NewSSORPreconditioner(zeroDiag, 1.0)
		if err == nil {
			t.Error("Expected error for zero diagonal element")
		}

		// Test 1D array
		arr1D := array.Ones(internal.Shape{5}, internal.Float64)
		_, err = NewSSORPreconditioner(arr1D, 1.0)
		if err == nil {
			t.Error("Expected error for 1D array")
		}
	})

	t.Run("Vector validation in Apply", func(t *testing.T) {
		// Create valid SSOR preconditioner
		A := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		A.Set(3.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(3.0, 1, 1)
		preconditioner, _ := NewSSORPreconditioner(A, 1.0)

		// Test nil vector
		_, err := preconditioner.Apply(nil)
		if err == nil {
			t.Error("Expected error for nil vector")
		}

		// Test wrong size vector
		wrongSize, _ := array.FromSlice([]float64{1, 2, 3})
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

func TestSSORPreconditionedCG(t *testing.T) {
	t.Run("SSOR vs other preconditioners comparison", func(t *testing.T) {
		// Create a symmetric positive definite test system
		A := array.Zeros(internal.Shape{4, 4}, internal.Float64)
		A.Set(8.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(0.0, 0, 2)
		A.Set(1.0, 0, 3)
		A.Set(1.0, 1, 0)
		A.Set(6.0, 1, 1)
		A.Set(1.0, 1, 2)
		A.Set(0.0, 1, 3)
		A.Set(0.0, 2, 0)
		A.Set(1.0, 2, 1)
		A.Set(5.0, 2, 2)
		A.Set(1.0, 2, 3)
		A.Set(1.0, 3, 0)
		A.Set(0.0, 3, 1)
		A.Set(1.0, 3, 2)
		A.Set(4.0, 3, 3)

		b, _ := array.FromSlice([]float64{10, 8, 7, 6})

		options := &IterativeSolverOptions{
			MaxIterations: 20,
			Tolerance:     1e-8,
		}

		// Solve with Jacobi preconditioning
		jacobiPrec, _ := NewJacobiPreconditioner(A)
		jacobiResult := PreconditionedConjugateGradient(A, b, jacobiPrec, options)

		// Solve with ILU preconditioning
		iluPrec, _ := NewILUPreconditioner(A, 0)
		iluResult := PreconditionedConjugateGradient(A, b, iluPrec, options)

		// Solve with SSOR preconditioning
		ssorePrec, err := NewSSORPreconditioner(A, 1.0)
		if err != nil {
			t.Fatalf("Failed to create SSOR preconditioner: %v", err)
		}
		ssorResult := PreconditionedConjugateGradient(A, b, ssorePrec, options)

		// Check that all methods converged
		if jacobiResult.Error != nil {
			t.Fatalf("Jacobi PCG failed: %v", jacobiResult.Error)
		}
		if iluResult.Error != nil {
			t.Fatalf("ILU PCG failed: %v", iluResult.Error)
		}
		if ssorResult.Error != nil {
			t.Fatalf("SSOR PCG failed: %v", ssorResult.Error)
		}

		// All should converge for this SPD system
		if !jacobiResult.Converged {
			t.Error("Jacobi PCG should have converged")
		}
		if !iluResult.Converged {
			t.Error("ILU PCG should have converged")
		}
		if !ssorResult.Converged {
			t.Error("SSOR PCG should have converged")
		}

		// Log performance comparison
		t.Logf("Jacobi PCG: %d iterations, residual %.2e",
			jacobiResult.Iterations, jacobiResult.Residual)
		t.Logf("ILU PCG:    %d iterations, residual %.2e",
			iluResult.Iterations, iluResult.Residual)
		t.Logf("SSOR PCG:   %d iterations, residual %.2e",
			ssorResult.Iterations, ssorResult.Residual)

		// Solutions should be similar
		if jacobiResult.Converged && iluResult.Converged && ssorResult.Converged {
			for i := 0; i < 4; i++ {
				jacobiSol := jacobiResult.Solution.At(i).(float64)
				iluSol := iluResult.Solution.At(i).(float64)
				ssorSol := ssorResult.Solution.At(i).(float64)

				if math.Abs(jacobiSol-iluSol) > 1e-6 {
					t.Errorf("Jacobi and ILU solutions differ at [%d]: %.6f vs %.6f",
						i, jacobiSol, iluSol)
				}
				if math.Abs(jacobiSol-ssorSol) > 1e-6 {
					t.Errorf("Jacobi and SSOR solutions differ at [%d]: %.6f vs %.6f",
						i, jacobiSol, ssorSol)
				}
			}
		}
	})

	t.Run("SSOR optimal omega selection", func(t *testing.T) {
		// Test different omega values to find optimal performance
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(5.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 0, 2)
		A.Set(1.0, 1, 0)
		A.Set(5.0, 1, 1)
		A.Set(1.0, 1, 2)
		A.Set(1.0, 2, 0)
		A.Set(1.0, 2, 1)
		A.Set(5.0, 2, 2)

		b, _ := array.FromSlice([]float64{7, 7, 7}) // Solution should be [1, 1, 1]

		options := &IterativeSolverOptions{
			MaxIterations: 10,
			Tolerance:     1e-10,
		}

		omegaValues := []float64{0.8, 1.0, 1.2, 1.5}
		bestIterations := 1000
		bestOmega := 1.0

		for _, omega := range omegaValues {
			preconditioner, err := NewSSORPreconditioner(A, omega)
			if err != nil {
				t.Errorf("Failed to create SSOR preconditioner with ω=%f: %v", omega, err)
				continue
			}

			result := PreconditionedConjugateGradient(A, b, preconditioner, options)
			if result.Error != nil {
				t.Logf("SSOR PCG with ω=%f failed: %v", omega, result.Error)
				continue
			}

			t.Logf("SSOR(ω=%.1f): %d iterations, residual %.2e, converged=%t",
				omega, result.Iterations, result.Residual, result.Converged)

			if result.Converged && result.Iterations < bestIterations {
				bestIterations = result.Iterations
				bestOmega = omega
			}
		}

		t.Logf("Best performance: ω=%.1f with %d iterations", bestOmega, bestIterations)
	})
}
