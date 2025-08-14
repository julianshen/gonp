package math

import (
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// ILUPreconditioner implements Incomplete LU factorization preconditioning
type ILUPreconditioner struct {
	L         *array.Array // Lower triangular factor
	U         *array.Array // Upper triangular factor
	fillLevel int          // Level of fill-in allowed
	n         int          // Matrix size
}

// NewILUPreconditioner creates a new ILU preconditioner from matrix A
// The fillLevel parameter controls how much fill-in is allowed:
// - fillLevel = 0: ILU(0) - only original non-zero positions
// - fillLevel = 1: ILU(1) - allow one level of additional fill-in
func NewILUPreconditioner(A *array.Array, fillLevel int) (*ILUPreconditioner, error) {
	if A == nil {
		return nil, fmt.Errorf("matrix cannot be nil")
	}

	shape := A.Shape()
	if shape.Ndim() != 2 {
		return nil, fmt.Errorf("matrix must be 2-dimensional")
	}

	if shape[0] != shape[1] {
		return nil, fmt.Errorf("matrix must be square")
	}

	if fillLevel < 0 {
		return nil, fmt.Errorf("fill level must be non-negative")
	}

	n := shape[0]

	// Check for zero diagonal elements
	for i := 0; i < n; i++ {
		if convertToFloat64(A.At(i, i)) == 0 {
			return nil, fmt.Errorf("matrix has zero diagonal element at position %d", i)
		}
	}

	// Perform ILU factorization
	L, U, err := performILUFactorization(A, fillLevel)
	if err != nil {
		return nil, fmt.Errorf("ILU factorization failed: %v", err)
	}

	return &ILUPreconditioner{
		L:         L,
		U:         U,
		fillLevel: fillLevel,
		n:         n,
	}, nil
}

// performILUFactorization computes the incomplete LU factorization
func performILUFactorization(A *array.Array, fillLevel int) (*array.Array, *array.Array, error) {
	n := A.Shape()[0]

	// Initialize L and U matrices
	L := array.Zeros(internal.Shape{n, n}, internal.Float64)
	U := array.Zeros(internal.Shape{n, n}, internal.Float64)

	// Create a working copy of A
	work := A.Copy()

	// Determine sparsity pattern (which positions to keep)
	sparsityPattern := createSparsityPattern(A, fillLevel)

	// Perform incomplete LU factorization using Gaussian elimination
	for k := 0; k < n; k++ {
		// Set U[k,k] = work[k,k]
		ukk := convertToFloat64(work.At(k, k))
		if ukk == 0 {
			return nil, nil, fmt.Errorf("zero pivot encountered at position %d", k)
		}
		U.Set(ukk, k, k)

		// Set L[k,k] = 1
		L.Set(1.0, k, k)

		// Fill U row k
		for j := k + 1; j < n; j++ {
			if sparsityPattern[k][j] {
				val := convertToFloat64(work.At(k, j))
				U.Set(val, k, j)
			}
		}

		// Fill L column k
		for i := k + 1; i < n; i++ {
			if sparsityPattern[i][k] {
				val := convertToFloat64(work.At(i, k)) / ukk
				L.Set(val, i, k)
			}
		}

		// Update remaining submatrix
		for i := k + 1; i < n; i++ {
			for j := k + 1; j < n; j++ {
				if sparsityPattern[i][j] {
					aij := convertToFloat64(work.At(i, j))
					lik := convertToFloat64(L.At(i, k))
					ukj := convertToFloat64(U.At(k, j))
					newVal := aij - lik*ukj
					work.Set(newVal, i, j)
				}
			}
		}
	}

	return L, U, nil
}

// createSparsityPattern determines which matrix positions should be kept
func createSparsityPattern(A *array.Array, fillLevel int) [][]bool {
	n := A.Shape()[0]
	pattern := make([][]bool, n)
	for i := range pattern {
		pattern[i] = make([]bool, n)
	}

	// Start with original sparsity pattern
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if convertToFloat64(A.At(i, j)) != 0 {
				pattern[i][j] = true
			}
		}
	}

	// Add fill-in based on level
	if fillLevel > 0 {
		// For ILU(k), add positions that would be filled during factorization
		// This is a simplified version - production code would use more sophisticated algorithms
		for level := 0; level < fillLevel; level++ {
			newPattern := make([][]bool, n)
			for i := range newPattern {
				newPattern[i] = make([]bool, n)
				copy(newPattern[i], pattern[i])
			}

			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					if !pattern[i][j] {
						// Check if this position would be filled
						for k := 0; k < min(i, j); k++ {
							if pattern[i][k] && pattern[k][j] {
								newPattern[i][j] = true
								break
							}
						}
					}
				}
			}
			pattern = newPattern
		}
	}

	return pattern
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Apply applies the ILU preconditioner to vector x
// Solves (LU)^(-1) * x by forward substitution (L * y = x) then backward substitution (U * result = y)
func (ilu *ILUPreconditioner) Apply(x *array.Array) (*array.Array, error) {
	if x == nil {
		return nil, fmt.Errorf("input vector cannot be nil")
	}

	if x.Ndim() != 1 {
		return nil, fmt.Errorf("input must be a vector (1D array)")
	}

	if x.Size() != ilu.n {
		return nil, fmt.Errorf("vector size (%d) must match matrix size (%d)", x.Size(), ilu.n)
	}

	// Step 1: Forward substitution L * y = x
	y, err := ilu.forwardSubstitution(x)
	if err != nil {
		return nil, fmt.Errorf("forward substitution failed: %v", err)
	}

	// Step 2: Backward substitution U * result = y
	result, err := ilu.backwardSubstitution(y)
	if err != nil {
		return nil, fmt.Errorf("backward substitution failed: %v", err)
	}

	return result, nil
}

// forwardSubstitution solves L * y = x for y
func (ilu *ILUPreconditioner) forwardSubstitution(x *array.Array) (*array.Array, error) {
	n := ilu.n
	y := array.Empty(internal.Shape{n}, internal.Float64)

	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < i; j++ {
			lij := convertToFloat64(ilu.L.At(i, j))
			yj := convertToFloat64(y.At(j))
			sum += lij * yj
		}

		xi := convertToFloat64(x.At(i))
		// Since L[i,i] = 1, we have yi = (xi - sum)
		y.Set(xi-sum, i)
	}

	return y, nil
}

// backwardSubstitution solves U * result = y for result
func (ilu *ILUPreconditioner) backwardSubstitution(y *array.Array) (*array.Array, error) {
	n := ilu.n
	result := array.Empty(internal.Shape{n}, internal.Float64)

	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			uij := convertToFloat64(ilu.U.At(i, j))
			resultj := convertToFloat64(result.At(j))
			sum += uij * resultj
		}

		yi := convertToFloat64(y.At(i))
		uii := convertToFloat64(ilu.U.At(i, i))

		if uii == 0 {
			return nil, fmt.Errorf("zero diagonal element in U at position %d", i)
		}

		// resulti = (yi - sum) / U[i,i]
		result.Set((yi-sum)/uii, i)
	}

	return result, nil
}

// Solve solves M * y = x for y, which is equivalent to Apply for ILU preconditioners
func (ilu *ILUPreconditioner) Solve(x *array.Array) (*array.Array, error) {
	return ilu.Apply(x)
}

// GetDiagonal returns the diagonal of the U factor (not commonly used for ILU)
func (ilu *ILUPreconditioner) GetDiagonal() *array.Array {
	diagonal := array.Empty(internal.Shape{ilu.n}, internal.Float64)
	for i := 0; i < ilu.n; i++ {
		val := convertToFloat64(ilu.U.At(i, i))
		diagonal.Set(1.0/val, i) // Return reciprocal for consistency with Jacobi
	}
	return diagonal
}

// GetL returns the L factor
func (ilu *ILUPreconditioner) GetL() *array.Array {
	return ilu.L
}

// GetU returns the U factor
func (ilu *ILUPreconditioner) GetU() *array.Array {
	return ilu.U
}
