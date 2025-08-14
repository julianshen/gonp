package math

import (
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// SSORPreconditioner implements Symmetric Successive Over-Relaxation preconditioning
// SSOR is particularly effective for symmetric positive definite matrices
type SSORPreconditioner struct {
	D     *array.Array // Diagonal part of A
	L     *array.Array // Strictly lower triangular part of A
	U     *array.Array // Strictly upper triangular part of A
	omega float64      // Relaxation parameter (0 < ω < 2)
	n     int          // Matrix size
}

// NewSSORPreconditioner creates a new SSOR preconditioner from matrix A
// The omega parameter is the relaxation parameter:
// - omega = 1.0: Gauss-Seidel method
// - omega < 1.0: Under-relaxation (more stable)
// - omega > 1.0: Over-relaxation (faster convergence if stable)
// - Theoretical optimal range: 0 < omega < 2
func NewSSORPreconditioner(A *array.Array, omega float64) (*SSORPreconditioner, error) {
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

	if omega <= 0 || omega >= 2.0 {
		return nil, fmt.Errorf("relaxation parameter omega must be in range (0, 2), got %f", omega)
	}

	n := shape[0]

	// Check for zero diagonal elements
	for i := 0; i < n; i++ {
		if convertToFloat64(A.At(i, i)) == 0 {
			return nil, fmt.Errorf("matrix has zero diagonal element at position %d", i)
		}
	}

	// Extract D, L, and U components
	D, L, U := extractDLU(A)

	return &SSORPreconditioner{
		D:     D,
		L:     L,
		U:     U,
		omega: omega,
		n:     n,
	}, nil
}

// extractDLU extracts the diagonal (D), strictly lower triangular (L),
// and strictly upper triangular (U) parts from matrix A
func extractDLU(A *array.Array) (*array.Array, *array.Array, *array.Array) {
	n := A.Shape()[0]

	D := array.Zeros(internal.Shape{n, n}, internal.Float64)
	L := array.Zeros(internal.Shape{n, n}, internal.Float64)
	U := array.Zeros(internal.Shape{n, n}, internal.Float64)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aij := convertToFloat64(A.At(i, j))

			if i == j {
				// Diagonal element
				D.Set(aij, i, j)
			} else if i > j {
				// Lower triangular element
				L.Set(aij, i, j)
			} else {
				// Upper triangular element
				U.Set(aij, i, j)
			}
		}
	}

	return D, L, U
}

// Apply applies the SSOR preconditioner to vector x
// SSOR preconditioning involves two sweeps:
// 1. Forward sweep: (D + ωL) * y = ω * x
// 2. Backward sweep: (D + ωU) * result = D * y
// The complete SSOR preconditioner is M = (1/ω)(D + ωL)D^(-1)(D + ωU)
func (ssor *SSORPreconditioner) Apply(x *array.Array) (*array.Array, error) {
	if x == nil {
		return nil, fmt.Errorf("input vector cannot be nil")
	}

	if x.Ndim() != 1 {
		return nil, fmt.Errorf("input must be a vector (1D array)")
	}

	if x.Size() != ssor.n {
		return nil, fmt.Errorf("vector size (%d) must match matrix size (%d)", x.Size(), ssor.n)
	}

	// Step 1: Forward sweep - solve (D + ωL) * y = ω * x
	y, err := ssor.forwardSweep(x)
	if err != nil {
		return nil, fmt.Errorf("forward sweep failed: %v", err)
	}

	// Step 2: Backward sweep - solve (D + ωU) * result = D * y
	result, err := ssor.backwardSweep(y)
	if err != nil {
		return nil, fmt.Errorf("backward sweep failed: %v", err)
	}

	return result, nil
}

// forwardSweep solves (D + ωL) * y = ω * x for y
func (ssor *SSORPreconditioner) forwardSweep(x *array.Array) (*array.Array, error) {
	n := ssor.n
	y := array.Empty(internal.Shape{n}, internal.Float64)

	for i := 0; i < n; i++ {
		// Compute sum of L[i,j] * y[j] for j < i
		sum := 0.0
		for j := 0; j < i; j++ {
			lij := convertToFloat64(ssor.L.At(i, j))
			yj := convertToFloat64(y.At(j))
			sum += lij * yj
		}

		// Get diagonal element
		dii := convertToFloat64(ssor.D.At(i, i))
		if dii == 0 {
			return nil, fmt.Errorf("zero diagonal element at position %d", i)
		}

		// Solve: (dii + ω * 0) * yi = ω * xi - ω * sum
		// Simplified: dii * yi = ω * xi - ω * sum
		xi := convertToFloat64(x.At(i))
		yi := (ssor.omega*xi - ssor.omega*sum) / dii
		y.Set(yi, i)
	}

	return y, nil
}

// backwardSweep solves (D + ωU) * result = D * y for result
func (ssor *SSORPreconditioner) backwardSweep(y *array.Array) (*array.Array, error) {
	n := ssor.n
	result := array.Empty(internal.Shape{n}, internal.Float64)

	for i := n - 1; i >= 0; i-- {
		// Compute sum of U[i,j] * result[j] for j > i
		sum := 0.0
		for j := i + 1; j < n; j++ {
			uij := convertToFloat64(ssor.U.At(i, j))
			resultj := convertToFloat64(result.At(j))
			sum += uij * resultj
		}

		// Get diagonal element
		dii := convertToFloat64(ssor.D.At(i, i))
		if dii == 0 {
			return nil, fmt.Errorf("zero diagonal element at position %d", i)
		}

		// Solve: (dii + ω * 0) * resulti = dii * yi - ω * sum
		// Simplified: dii * resulti = dii * yi - ω * sum
		yi := convertToFloat64(y.At(i))
		resulti := (dii*yi - ssor.omega*sum) / dii
		result.Set(resulti, i)
	}

	return result, nil
}

// Solve solves M * z = x for z, which is equivalent to Apply for SSOR preconditioners
func (ssor *SSORPreconditioner) Solve(x *array.Array) (*array.Array, error) {
	return ssor.Apply(x)
}

// GetDiagonal returns the inverse of the diagonal elements (for consistency with other preconditioners)
func (ssor *SSORPreconditioner) GetDiagonal() *array.Array {
	diagonal := array.Empty(internal.Shape{ssor.n}, internal.Float64)
	for i := 0; i < ssor.n; i++ {
		dii := convertToFloat64(ssor.D.At(i, i))
		diagonal.Set(1.0/dii, i)
	}
	return diagonal
}

// GetOmega returns the relaxation parameter
func (ssor *SSORPreconditioner) GetOmega() float64 {
	return ssor.omega
}

// GetComponents returns the D, L, U components (for debugging/analysis)
func (ssor *SSORPreconditioner) GetComponents() (*array.Array, *array.Array, *array.Array) {
	return ssor.D, ssor.L, ssor.U
}
