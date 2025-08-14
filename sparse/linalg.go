// Linear algebra operations for sparse matrices
package sparse

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Norm computes various matrix norms
type NormType int

const (
	Frobenius NormType = iota // Frobenius norm (default)
	OneNorm                   // 1-norm (maximum absolute column sum)
	InfNorm                   // Infinity-norm (maximum absolute row sum)
)

// Norm computes the specified norm of the sparse matrix
func (sm *SparseMatrix) Norm(normType NormType) (float64, error) {
	switch normType {
	case Frobenius:
		return sm.frobeniusNorm()
	case OneNorm:
		return sm.oneNorm()
	case InfNorm:
		return sm.infNorm()
	default:
		return 0, fmt.Errorf("unsupported norm type: %d", normType)
	}
}

// frobeniusNorm computes the Frobenius norm (sqrt of sum of squares)
func (sm *SparseMatrix) frobeniusNorm() (float64, error) {
	var sum float64 = 0.0

	for i := 0; i < sm.nnz; i++ {
		val := sm.convertToFloat64(sm.data[i])
		sum += val * val
	}

	return math.Sqrt(sum), nil
}

// oneNorm computes the 1-norm (maximum absolute column sum)
func (sm *SparseMatrix) oneNorm() (float64, error) {
	colSums := make([]float64, sm.cols)

	err := sm.iterateNonZeros(func(row, col int, value interface{}) error {
		val := sm.convertToFloat64(value)
		colSums[col] += math.Abs(val)
		return nil
	})

	if err != nil {
		return 0, err
	}

	maxSum := colSums[0]
	for i := 1; i < len(colSums); i++ {
		if colSums[i] > maxSum {
			maxSum = colSums[i]
		}
	}

	return maxSum, nil
}

// infNorm computes the infinity-norm (maximum absolute row sum)
func (sm *SparseMatrix) infNorm() (float64, error) {
	rowSums := make([]float64, sm.rows)

	err := sm.iterateNonZeros(func(row, col int, value interface{}) error {
		val := sm.convertToFloat64(value)
		rowSums[row] += math.Abs(val)
		return nil
	})

	if err != nil {
		return 0, err
	}

	maxSum := rowSums[0]
	for i := 1; i < len(rowSums); i++ {
		if rowSums[i] > maxSum {
			maxSum = rowSums[i]
		}
	}

	return maxSum, nil
}

// Trace computes the trace (sum of diagonal elements)
func (sm *SparseMatrix) Trace() (interface{}, error) {
	if sm.rows != sm.cols {
		return nil, fmt.Errorf("trace is only defined for square matrices, got (%d, %d)",
			sm.rows, sm.cols)
	}

	trace := sm.getZeroValue()

	// Only iterate through diagonal elements
	for i := 0; i < sm.rows; i++ {
		diagValue, err := sm.Get(i, i)
		if err != nil {
			return nil, fmt.Errorf("failed to get diagonal element (%d, %d): %v", i, i, err)
		}

		if !sm.isZero(diagValue) {
			trace = sm.addValues(trace, diagValue)
		}
	}

	return trace, nil
}

// SpMV performs sparse matrix-vector multiplication
func (sm *SparseMatrix) SpMV(vector *array.Array) (*array.Array, error) {
	vShape := vector.Shape()

	// Check if vector is 1D with correct size
	if len(vShape) != 1 || vShape[0] != sm.cols {
		return nil, fmt.Errorf("vector shape mismatch: expected (%d,), got %v", sm.cols, vShape)
	}

	// Create result vector
	result := make([]interface{}, sm.rows)
	zeroValue := sm.getZeroValue()

	// Initialize result with zeros
	for i := range result {
		result[i] = zeroValue
	}

	// Perform multiplication based on format
	switch sm.format {
	case CSR:
		// CSR is optimal for matrix-vector multiplication
		for i := 0; i < sm.rows; i++ {
			rowSum := zeroValue
			for j := sm.indptr[i]; j < sm.indptr[i+1]; j++ {
				col := sm.indices[j]
				matVal := sm.data[j]
				vecVal := vector.At(col)

				product := sm.multiplyValues(matVal, vecVal)
				rowSum = sm.addValues(rowSum, product)
			}
			result[i] = rowSum
		}
	default:
		// For other formats, iterate through non-zeros
		err := sm.iterateNonZeros(func(row, col int, value interface{}) error {
			vecVal := vector.At(col)
			product := sm.multiplyValues(value, vecVal)
			result[row] = sm.addValues(result[row], product)
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("failed to perform SpMV: %v", err)
		}
	}

	return array.FromSlice(result)
}

// Solve attempts to solve the linear system Ax = b using iterative methods
// Currently implements Conjugate Gradient for symmetric positive definite matrices
func (sm *SparseMatrix) Solve(b *array.Array, tolerance float64, maxIter int) (*array.Array, error) {
	if sm.rows != sm.cols {
		return nil, fmt.Errorf("can only solve systems with square matrices, got (%d, %d)",
			sm.rows, sm.cols)
	}

	bShape := b.Shape()
	if len(bShape) != 1 || bShape[0] != sm.rows {
		return nil, fmt.Errorf("RHS vector shape mismatch: expected (%d,), got %v", sm.rows, bShape)
	}

	// Use Conjugate Gradient method (assumes symmetric positive definite matrix)
	return sm.conjugateGradient(b, tolerance, maxIter)
}

// conjugateGradient implements the Conjugate Gradient iterative solver
func (sm *SparseMatrix) conjugateGradient(b *array.Array, tolerance float64, maxIter int) (*array.Array, error) {
	n := sm.rows

	// Initialize solution vector x = 0
	x := make([]interface{}, n)
	zeroValue := sm.getZeroValue()
	for i := range x {
		x[i] = zeroValue
	}
	xArray, _ := array.FromSlice(x)

	// r = b - Ax (initial residual)
	Ax, err := sm.SpMV(xArray)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Ax: %v", err)
	}

	r := make([]interface{}, n)
	for i := 0; i < n; i++ {
		bVal := b.At(i)
		AxVal := Ax.At(i)
		r[i] = sm.subtractValues(bVal, AxVal)
	}
	rArray, _ := array.FromSlice(r)

	// p = r (initial search direction)
	p := make([]interface{}, n)
	copy(p, r)
	pArray, _ := array.FromSlice(p)

	// rsold = r^T * r
	rsold, err := sm.dotProduct(rArray, rArray)
	if err != nil {
		return nil, err
	}

	for iter := 0; iter < maxIter; iter++ {
		// Check convergence
		if math.Sqrt(sm.convertToFloat64(rsold)) < tolerance {
			break
		}

		// Ap = A * p
		Ap, err := sm.SpMV(pArray)
		if err != nil {
			return nil, fmt.Errorf("failed to compute Ap: %v", err)
		}

		// alpha = rsold / (p^T * Ap)
		pTAp, err := sm.dotProduct(pArray, Ap)
		if err != nil {
			return nil, err
		}

		if sm.convertToFloat64(pTAp) == 0.0 {
			return nil, fmt.Errorf("conjugate gradient failed: pTAp is zero")
		}

		alpha := sm.divideValues(rsold, pTAp)

		// x = x + alpha * p
		for i := 0; i < n; i++ {
			alphaP := sm.multiplyValues(alpha, pArray.At(i))
			x[i] = sm.addValues(xArray.At(i), alphaP)
		}
		xArray, _ = array.FromSlice(x)

		// r = r - alpha * Ap
		for i := 0; i < n; i++ {
			alphaAp := sm.multiplyValues(alpha, Ap.At(i))
			r[i] = sm.subtractValues(rArray.At(i), alphaAp)
		}
		rArray, _ = array.FromSlice(r)

		// rsnew = r^T * r
		rsnew, err := sm.dotProduct(rArray, rArray)
		if err != nil {
			return nil, err
		}

		// beta = rsnew / rsold
		beta := sm.divideValues(rsnew, rsold)

		// p = r + beta * p
		for i := 0; i < n; i++ {
			betaP := sm.multiplyValues(beta, pArray.At(i))
			p[i] = sm.addValues(rArray.At(i), betaP)
		}
		pArray, _ = array.FromSlice(p)

		rsold = rsnew
	}

	return array.FromSlice(x)
}

// Helper functions

func (sm *SparseMatrix) convertToFloat64(value interface{}) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int64:
		return float64(v)
	case int:
		return float64(v)
	default:
		return 0.0
	}
}

func (sm *SparseMatrix) dotProduct(a, b *array.Array) (interface{}, error) {
	if a.Size() != b.Size() {
		return nil, fmt.Errorf("vector size mismatch: %d vs %d", a.Size(), b.Size())
	}

	result := sm.getZeroValue()

	for i := 0; i < a.Size(); i++ {
		aVal := a.At(i)
		bVal := b.At(i)
		product := sm.multiplyValues(aVal, bVal)
		result = sm.addValues(result, product)
	}

	return result, nil
}

func (sm *SparseMatrix) divideValues(a, b interface{}) interface{} {
	switch sm.dtype {
	case internal.Float64:
		return a.(float64) / b.(float64)
	case internal.Float32:
		return a.(float32) / b.(float32)
	case internal.Int64:
		// For integer division, convert to float64 to avoid truncation
		return float64(a.(int64)) / float64(b.(int64))
	default:
		return a
	}
}

// PowerIteration performs power iteration to find the dominant eigenvalue
func (sm *SparseMatrix) PowerIteration(maxIter int, tolerance float64) (float64, *array.Array, error) {
	if sm.rows != sm.cols {
		return 0, nil, fmt.Errorf("power iteration requires square matrix, got (%d, %d)",
			sm.rows, sm.cols)
	}

	// Initialize random vector
	x := make([]interface{}, sm.rows)
	for i := range x {
		x[i] = 1.0 // Simple initialization
	}
	xArray, _ := array.FromSlice(x)

	var eigenvalue float64

	for iter := 0; iter < maxIter; iter++ {
		// y = A * x
		y, err := sm.SpMV(xArray)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to multiply matrix with vector: %v", err)
		}

		// Compute eigenvalue estimate: lambda = x^T * y
		xTy, err := sm.dotProduct(xArray, y)
		if err != nil {
			return 0, nil, err
		}
		newEigenvalue := sm.convertToFloat64(xTy)

		// Normalize y to get new x
		yNorm, err := sm.vectorNorm(y)
		if err != nil {
			return 0, nil, err
		}

		if yNorm == 0 {
			return 0, nil, fmt.Errorf("power iteration failed: zero norm vector")
		}

		// x = y / ||y||
		for i := 0; i < y.Size(); i++ {
			val := sm.convertToFloat64(y.At(i)) / yNorm
			x[i] = val
		}
		xArray, _ = array.FromSlice(x)

		// Check convergence
		if iter > 0 && math.Abs(newEigenvalue-eigenvalue) < tolerance {
			break
		}

		eigenvalue = newEigenvalue
	}

	return eigenvalue, xArray, nil
}

func (sm *SparseMatrix) vectorNorm(v *array.Array) (float64, error) {
	var sum float64 = 0.0

	for i := 0; i < v.Size(); i++ {
		val := sm.convertToFloat64(v.At(i))
		sum += val * val
	}

	return math.Sqrt(sum), nil
}
