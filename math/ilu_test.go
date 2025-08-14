package math

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestILUPreconditioner(t *testing.T) {
	t.Run("Simple tridiagonal matrix ILU(0)", func(t *testing.T) {
		// Create a simple tridiagonal matrix
		//   2 -1  0
		//  -1  2 -1
		//   0 -1  2
		A := array.Zeros(internal.Shape{3, 3}, internal.Float64)
		A.Set(2.0, 0, 0)
		A.Set(-1.0, 0, 1)
		A.Set(-1.0, 1, 0)
		A.Set(2.0, 1, 1)
		A.Set(-1.0, 1, 2)
		A.Set(-1.0, 2, 1)
		A.Set(2.0, 2, 2)

		// Perform ILU(0) factorization - only fill existing non-zero positions
		preconditioner, err := NewILUPreconditioner(A, 0)
		if err != nil {
			t.Fatalf("Failed to create ILU preconditioner: %v", err)
		}

		// Check that L and U factors exist
		L := preconditioner.GetL()
		U := preconditioner.GetU()

		if L == nil || U == nil {
			t.Fatal("L and U factors should not be nil")
		}

		// Verify L has ones on diagonal and lower triangular structure
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				lVal := L.At(i, j).(float64)
				if i == j {
					// Diagonal should be 1.0
					if math.Abs(lVal-1.0) > 1e-12 {
						t.Errorf("L[%d,%d] should be 1.0, got %f", i, j, lVal)
					}
				} else if i < j {
					// Upper triangle should be 0.0
					if math.Abs(lVal) > 1e-12 {
						t.Errorf("L[%d,%d] should be 0.0, got %f", i, j, lVal)
					}
				}
			}
		}

		// Test applying the preconditioner to a vector
		b, _ := array.FromSlice([]float64{1, 1, 1})
		y, err := preconditioner.Apply(b)
		if err != nil {
			t.Fatalf("Failed to apply ILU preconditioner: %v", err)
		}

		// Result should be reasonable (not testing exact values due to approximation)
		if y == nil || y.Size() != 3 {
			t.Error("Applied preconditioner should return vector of size 3")
		}

		t.Logf("ILU(0) applied to [1,1,1], result: [%.3f, %.3f, %.3f]",
			y.At(0).(float64), y.At(1).(float64), y.At(2).(float64))
	})

	t.Run("ILU factorization accuracy test", func(t *testing.T) {
		// Create a simple 2x2 SPD matrix for exact verification
		//  4  1
		//  1  2
		A := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(2.0, 1, 1)

		preconditioner, err := NewILUPreconditioner(A, 0)
		if err != nil {
			t.Fatalf("Failed to create ILU preconditioner: %v", err)
		}

		L := preconditioner.GetL()
		U := preconditioner.GetU()

		// For this simple matrix, we can verify exact ILU values
		// Expected L:  [1.0, 0.0]    Expected U:  [4.0, 1.0]
		//              [0.25, 1.0]                [0.0, 1.75]

		expectedL := [][]float64{{1.0, 0.0}, {0.25, 1.0}}
		expectedU := [][]float64{{4.0, 1.0}, {0.0, 1.75}}

		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				lActual := L.At(i, j).(float64)
				uActual := U.At(i, j).(float64)

				if math.Abs(lActual-expectedL[i][j]) > 1e-10 {
					t.Errorf("L[%d,%d]: expected %f, got %f", i, j, expectedL[i][j], lActual)
				}
				if math.Abs(uActual-expectedU[i][j]) > 1e-10 {
					t.Errorf("U[%d,%d]: expected %f, got %f", i, j, expectedU[i][j], uActual)
				}
			}
		}
	})

	t.Run("ILU with different fill levels", func(t *testing.T) {
		// Create a 4x4 matrix to test different fill-in levels
		A := array.Zeros(internal.Shape{4, 4}, internal.Float64)
		A.Set(4.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(0.0, 0, 2)
		A.Set(1.0, 0, 3)
		A.Set(1.0, 1, 0)
		A.Set(4.0, 1, 1)
		A.Set(1.0, 1, 2)
		A.Set(0.0, 1, 3)
		A.Set(0.0, 2, 0)
		A.Set(1.0, 2, 1)
		A.Set(4.0, 2, 2)
		A.Set(1.0, 2, 3)
		A.Set(1.0, 3, 0)
		A.Set(0.0, 3, 1)
		A.Set(1.0, 3, 2)
		A.Set(4.0, 3, 3)

		// Test ILU(0) - no additional fill-in
		ilu0, err := NewILUPreconditioner(A, 0)
		if err != nil {
			t.Fatalf("ILU(0) failed: %v", err)
		}

		// Test ILU(1) - allow 1 level of fill-in
		ilu1, err := NewILUPreconditioner(A, 1)
		if err != nil {
			t.Fatalf("ILU(1) failed: %v", err)
		}

		// Apply both preconditioners to same vector
		b, _ := array.FromSlice([]float64{1, 2, 3, 4})

		y0, err := ilu0.Apply(b)
		if err != nil {
			t.Fatalf("ILU(0) apply failed: %v", err)
		}

		y1, err := ilu1.Apply(b)
		if err != nil {
			t.Fatalf("ILU(1) apply failed: %v", err)
		}

		// Results should be different (ILU(1) should be more accurate)
		different := false
		for i := 0; i < 4; i++ {
			if math.Abs(y0.At(i).(float64)-y1.At(i).(float64)) > 1e-10 {
				different = true
				break
			}
		}

		if !different {
			t.Error("ILU(0) and ILU(1) should produce different results")
		}

		t.Logf("ILU(0) result: [%.3f, %.3f, %.3f, %.3f]",
			y0.At(0).(float64), y0.At(1).(float64), y0.At(2).(float64), y0.At(3).(float64))
		t.Logf("ILU(1) result: [%.3f, %.3f, %.3f, %.3f]",
			y1.At(0).(float64), y1.At(1).(float64), y1.At(2).(float64), y1.At(3).(float64))
	})

	t.Run("Parameter validation", func(t *testing.T) {
		// Test nil matrix
		_, err := NewILUPreconditioner(nil, 0)
		if err == nil {
			t.Error("Expected error for nil matrix")
		}

		// Test non-square matrix
		nonSquare := array.Zeros(internal.Shape{3, 4}, internal.Float64)
		_, err = NewILUPreconditioner(nonSquare, 0)
		if err == nil {
			t.Error("Expected error for non-square matrix")
		}

		// Test negative fill level
		validMatrix := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		validMatrix.Set(1.0, 0, 0)
		validMatrix.Set(1.0, 1, 1)
		_, err = NewILUPreconditioner(validMatrix, -1)
		if err == nil {
			t.Error("Expected error for negative fill level")
		}

		// Test zero diagonal (should handle gracefully or error)
		zeroDiag := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		zeroDiag.Set(0.0, 0, 0)
		zeroDiag.Set(1.0, 0, 1)
		zeroDiag.Set(1.0, 1, 0)
		zeroDiag.Set(1.0, 1, 1)
		_, err = NewILUPreconditioner(zeroDiag, 0)
		if err == nil {
			t.Error("Expected error for zero diagonal element")
		}

		// Test 1D array
		arr1D := array.Ones(internal.Shape{5}, internal.Float64)
		_, err = NewILUPreconditioner(arr1D, 0)
		if err == nil {
			t.Error("Expected error for 1D array")
		}
	})

	t.Run("Vector validation in Apply", func(t *testing.T) {
		// Create valid ILU preconditioner
		A := array.Zeros(internal.Shape{2, 2}, internal.Float64)
		A.Set(2.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(1.0, 1, 0)
		A.Set(2.0, 1, 1)
		preconditioner, _ := NewILUPreconditioner(A, 0)

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

func TestILUPreconditionedCG(t *testing.T) {
	t.Run("ILU preconditioned vs Jacobi preconditioned CG", func(t *testing.T) {
		// Create a test system with some condition number issues
		A := array.Zeros(internal.Shape{4, 4}, internal.Float64)
		A.Set(10.0, 0, 0)
		A.Set(1.0, 0, 1)
		A.Set(0.0, 0, 2)
		A.Set(1.0, 0, 3)
		A.Set(1.0, 1, 0)
		A.Set(8.0, 1, 1)
		A.Set(2.0, 1, 2)
		A.Set(0.0, 1, 3)
		A.Set(0.0, 2, 0)
		A.Set(2.0, 2, 1)
		A.Set(6.0, 2, 2)
		A.Set(1.0, 2, 3)
		A.Set(1.0, 3, 0)
		A.Set(0.0, 3, 1)
		A.Set(1.0, 3, 2)
		A.Set(4.0, 3, 3)

		b, _ := array.FromSlice([]float64{12, 11, 9, 6})

		options := &IterativeSolverOptions{
			MaxIterations: 20,
			Tolerance:     1e-8,
		}

		// Solve with Jacobi preconditioning
		jacobiPrec, _ := NewJacobiPreconditioner(A)
		jacobiResult := PreconditionedConjugateGradient(A, b, jacobiPrec, options)

		// Solve with ILU preconditioning
		iluPrec, err := NewILUPreconditioner(A, 0)
		if err != nil {
			t.Fatalf("Failed to create ILU preconditioner: %v", err)
		}
		iluResult := PreconditionedConjugateGradient(A, b, iluPrec, options)

		if jacobiResult.Error != nil {
			t.Fatalf("Jacobi PCG failed: %v", jacobiResult.Error)
		}
		if iluResult.Error != nil {
			t.Fatalf("ILU PCG failed: %v", iluResult.Error)
		}

		// Both should converge
		if !jacobiResult.Converged || !iluResult.Converged {
			t.Error("Both preconditioned solvers should converge")
		}

		// ILU should generally converge faster or to better accuracy
		improvement := false
		if iluResult.Iterations <= jacobiResult.Iterations {
			improvement = true
		}
		if iluResult.Residual <= jacobiResult.Residual {
			improvement = true
		}

		if !improvement {
			t.Logf("Note: ILU didn't clearly outperform Jacobi (may depend on matrix structure)")
		}

		t.Logf("Jacobi PCG: %d iterations, residual %.2e",
			jacobiResult.Iterations, jacobiResult.Residual)
		t.Logf("ILU PCG: %d iterations, residual %.2e",
			iluResult.Iterations, iluResult.Residual)

		// Solutions should be similar
		if jacobiResult.Converged && iluResult.Converged {
			for i := 0; i < 4; i++ {
				jacobiSol := jacobiResult.Solution.At(i).(float64)
				iluSol := iluResult.Solution.At(i).(float64)
				if math.Abs(jacobiSol-iluSol) > 1e-6 {
					t.Errorf("Solutions differ at [%d]: Jacobi=%f, ILU=%f",
						i, jacobiSol, iluSol)
				}
			}
		}
	})

	t.Run("ILU solver convergence properties", func(t *testing.T) {
		// Test ILU on a well-conditioned system
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

		b, _ := array.FromSlice([]float64{6, 6, 6}) // Solution should be [1, 1, 1]

		iluPrec, err := NewILUPreconditioner(A, 0)
		if err != nil {
			t.Fatalf("Failed to create ILU preconditioner: %v", err)
		}

		options := &IterativeSolverOptions{
			MaxIterations: 10,
			Tolerance:     1e-10,
		}

		result := PreconditionedConjugateGradient(A, b, iluPrec, options)
		if result.Error != nil {
			t.Fatalf("ILU PCG failed: %v", result.Error)
		}

		if !result.Converged {
			t.Error("ILU PCG should have converged")
		}

		// Check solution accuracy
		expectedSolution := []float64{1, 1, 1}
		for i, expected := range expectedSolution {
			actual := result.Solution.At(i).(float64)
			if math.Abs(actual-expected) > 1e-8 {
				t.Errorf("Solution[%d]: expected %f, got %f", i, expected, actual)
			}
		}

		t.Logf("ILU PCG converged in %d iterations with residual %e",
			result.Iterations, result.Residual)
	})
}
