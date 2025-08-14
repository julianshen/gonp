// Arithmetic operations for sparse matrices
package sparse

import (
	"fmt"

	"github.com/julianshen/gonp/internal"
)

// Add performs element-wise addition of two sparse matrices
func (sm *SparseMatrix) Add(other *SparseMatrix) (*SparseMatrix, error) {
	if sm.rows != other.rows || sm.cols != other.cols {
		return nil, fmt.Errorf("shape mismatch: (%d, %d) vs (%d, %d)",
			sm.rows, sm.cols, other.rows, other.cols)
	}

	if sm.dtype != other.dtype {
		return nil, fmt.Errorf("dtype mismatch: cannot add matrices with different dtypes")
	}

	// Result matrix uses COO format for efficient construction
	opts := &SparseOptions{
		Format:   COO,
		DType:    sm.dtype,
		Capacity: sm.nnz + other.nnz,
	}

	result := NewSparseMatrix(sm.rows, sm.cols, opts)

	// Add all elements from first matrix
	err := sm.iterateNonZeros(func(row, col int, value interface{}) error {
		return result.Set(row, col, value)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to copy first matrix: %v", err)
	}

	// Add all elements from second matrix (this will handle overlapping elements)
	err = other.iterateNonZeros(func(row, col int, value interface{}) error {
		// Get existing value from result
		existingValue, _ := result.Get(row, col)
		// Add the values
		newValue := sm.addValues(existingValue, value)
		return result.Set(row, col, newValue)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add second matrix: %v", err)
	}

	// Clean up duplicates and sort
	result.EliminateDuplicates()

	return result, nil
}

// Subtract performs element-wise subtraction of two sparse matrices
func (sm *SparseMatrix) Subtract(other *SparseMatrix) (*SparseMatrix, error) {
	if sm.rows != other.rows || sm.cols != other.cols {
		return nil, fmt.Errorf("shape mismatch: (%d, %d) vs (%d, %d)",
			sm.rows, sm.cols, other.rows, other.cols)
	}

	if sm.dtype != other.dtype {
		return nil, fmt.Errorf("dtype mismatch: cannot subtract matrices with different dtypes")
	}

	opts := &SparseOptions{
		Format:   COO,
		DType:    sm.dtype,
		Capacity: sm.nnz + other.nnz,
	}

	result := NewSparseMatrix(sm.rows, sm.cols, opts)

	// Add all elements from first matrix
	err := sm.iterateNonZeros(func(row, col int, value interface{}) error {
		return result.Set(row, col, value)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to copy first matrix: %v", err)
	}

	// Subtract all elements from second matrix
	err = other.iterateNonZeros(func(row, col int, value interface{}) error {
		existingValue, _ := result.Get(row, col)
		newValue := sm.subtractValues(existingValue, value)
		return result.Set(row, col, newValue)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to subtract second matrix: %v", err)
	}

	result.EliminateDuplicates()

	return result, nil
}

// ScalarMultiply multiplies all elements by a scalar
func (sm *SparseMatrix) ScalarMultiply(scalar interface{}) (*SparseMatrix, error) {
	// Convert scalar to the matrix's dtype
	convertedScalar, err := sm.convertValue(scalar)
	if err != nil {
		return nil, fmt.Errorf("failed to convert scalar: %v", err)
	}

	result := sm.Clone()

	// Multiply all non-zero values by the scalar
	for i := 0; i < result.nnz; i++ {
		result.data[i] = sm.multiplyValues(result.data[i], convertedScalar)
	}

	return result, nil
}

// ElementwiseMultiply performs element-wise multiplication (Hadamard product)
func (sm *SparseMatrix) ElementwiseMultiply(other *SparseMatrix) (*SparseMatrix, error) {
	if sm.rows != other.rows || sm.cols != other.cols {
		return nil, fmt.Errorf("shape mismatch: (%d, %d) vs (%d, %d)",
			sm.rows, sm.cols, other.rows, other.cols)
	}

	if sm.dtype != other.dtype {
		return nil, fmt.Errorf("dtype mismatch: cannot multiply matrices with different dtypes")
	}

	// Result will be at most min(sm.nnz, other.nnz) non-zeros
	maxNnz := sm.nnz
	if other.nnz < maxNnz {
		maxNnz = other.nnz
	}

	opts := &SparseOptions{
		Format:   COO,
		DType:    sm.dtype,
		Capacity: maxNnz,
	}

	result := NewSparseMatrix(sm.rows, sm.cols, opts)

	// Only iterate over non-zeros in the smaller matrix for efficiency
	var smaller, larger *SparseMatrix
	if sm.nnz <= other.nnz {
		smaller, larger = sm, other
	} else {
		smaller, larger = other, sm
	}

	err := smaller.iterateNonZeros(func(row, col int, value interface{}) error {
		otherValue, _ := larger.Get(row, col)
		if !larger.isZero(otherValue) {
			product := sm.multiplyValues(value, otherValue)
			if !sm.isZero(product) {
				return result.Set(row, col, product)
			}
		}
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to perform element-wise multiplication: %v", err)
	}

	return result, nil
}

// MatMul performs sparse matrix multiplication
func (sm *SparseMatrix) MatMul(other *SparseMatrix) (*SparseMatrix, error) {
	if sm.cols != other.rows {
		return nil, fmt.Errorf("incompatible shapes for matrix multiplication: (%d, %d) × (%d, %d)",
			sm.rows, sm.cols, other.rows, other.cols)
	}

	if sm.dtype != other.dtype {
		return nil, fmt.Errorf("dtype mismatch: cannot multiply matrices with different dtypes")
	}

	// Convert both matrices to CSR for efficient row access
	smCSR, err := sm.ToFormat(CSR)
	if err != nil {
		return nil, fmt.Errorf("failed to convert first matrix to CSR: %v", err)
	}

	otherCSC, err := other.ToFormat(CSC)
	if err != nil {
		return nil, fmt.Errorf("failed to convert second matrix to CSC: %v", err)
	}

	// Estimate result size (rough heuristic)
	estimatedNnz := (smCSR.nnz * otherCSC.nnz) / max(sm.cols, 1)
	if estimatedNnz > sm.rows*other.cols/4 {
		estimatedNnz = sm.rows * other.cols / 4
	}

	opts := &SparseOptions{
		Format:   COO,
		DType:    sm.dtype,
		Capacity: estimatedNnz,
	}

	result := NewSparseMatrix(sm.rows, other.cols, opts)

	// Perform matrix multiplication using CSR × CSC
	for i := 0; i < smCSR.rows; i++ {
		for j := 0; j < otherCSC.cols; j++ {
			dotProduct := sm.getZeroValue()

			// Compute dot product of row i from smCSR and column j from otherCSC
			rowStart := smCSR.indptr[i]
			rowEnd := smCSR.indptr[i+1]
			colStart := otherCSC.indptr[j]
			colEnd := otherCSC.indptr[j+1]

			// Two-pointer technique for sparse dot product
			rowPtr := rowStart
			colPtr := colStart

			for rowPtr < rowEnd && colPtr < colEnd {
				rowIdx := smCSR.indices[rowPtr]
				colIdx := otherCSC.indices[colPtr]

				if rowIdx == colIdx {
					// Found matching indices
					product := sm.multiplyValues(smCSR.data[rowPtr], otherCSC.data[colPtr])
					dotProduct = sm.addValues(dotProduct, product)
					rowPtr++
					colPtr++
				} else if rowIdx < colIdx {
					rowPtr++
				} else {
					colPtr++
				}
			}

			// Set result if non-zero
			if !sm.isZero(dotProduct) {
				err := result.Set(i, j, dotProduct)
				if err != nil {
					return nil, fmt.Errorf("failed to set result element: %v", err)
				}
			}
		}
	}

	return result, nil
}

// Transpose returns the transpose of the sparse matrix
func (sm *SparseMatrix) Transpose() *SparseMatrix {
	var newFormat SparseFormat
	switch sm.format {
	case CSR:
		newFormat = CSC // CSR transpose becomes CSC
	case CSC:
		newFormat = CSR // CSC transpose becomes CSR
	default:
		newFormat = COO // COO stays COO
	}

	opts := &SparseOptions{
		Format:   newFormat,
		DType:    sm.dtype,
		Capacity: sm.nnz,
	}

	result := NewSparseMatrix(sm.cols, sm.rows, opts)

	// Copy elements with swapped coordinates
	sm.iterateNonZeros(func(row, col int, value interface{}) error {
		return result.Set(col, row, value) // Note: swapped row and col
	})

	return result
}

// Helper functions for arithmetic operations

func (sm *SparseMatrix) subtractValues(a, b interface{}) interface{} {
	switch sm.dtype {
	case internal.Float64:
		return a.(float64) - b.(float64)
	case internal.Float32:
		return a.(float32) - b.(float32)
	case internal.Int64:
		return a.(int64) - b.(int64)
	default:
		return a
	}
}

func (sm *SparseMatrix) multiplyValues(a, b interface{}) interface{} {
	switch sm.dtype {
	case internal.Float64:
		return a.(float64) * b.(float64)
	case internal.Float32:
		return a.(float32) * b.(float32)
	case internal.Int64:
		return a.(int64) * b.(int64)
	default:
		return a
	}
}

// Utility function for max of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
