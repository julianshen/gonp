// Sparse matrix conversion utilities for different formats and dense arrays
package sparse

import (
	"fmt"
	"sort"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// FromArray creates a sparse matrix from a dense array
func FromArray(arr *array.Array, format SparseFormat) (*SparseMatrix, error) {
	shape := arr.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("can only create sparse matrix from 2D array, got %dD", len(shape))
	}

	rows, cols := shape[0], shape[1]

	// Determine dtype from array
	var dtype internal.DType
	switch arr.At(0, 0).(type) {
	case float64:
		dtype = internal.Float64
	case float32:
		dtype = internal.Float32
	case int64:
		dtype = internal.Int64
	case int:
		dtype = internal.Int64 // Convert int to int64
	default:
		return nil, fmt.Errorf("unsupported array dtype for sparse matrix")
	}

	opts := &SparseOptions{
		Format:   format,
		DType:    dtype,
		Capacity: rows * cols / 10, // Initial guess for sparsity
	}

	sm := NewSparseMatrix(rows, cols, opts)

	// Scan through array and add non-zero elements
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := arr.At(i, j)
			if !sm.isZero(value) {
				err := sm.Set(i, j, value)
				if err != nil {
					return nil, fmt.Errorf("failed to set sparse matrix element: %v", err)
				}
			}
		}
	}

	return sm, nil
}

// ToFormat converts a sparse matrix to a different format
func (sm *SparseMatrix) ToFormat(newFormat SparseFormat) (*SparseMatrix, error) {
	if sm.format == newFormat {
		return sm, nil // No conversion needed
	}

	opts := &SparseOptions{
		Format:   newFormat,
		DType:    sm.dtype,
		Capacity: sm.nnz,
	}

	newSM := NewSparseMatrix(sm.rows, sm.cols, opts)

	// Copy all non-zero elements
	err := sm.iterateNonZeros(func(row, col int, value interface{}) error {
		return newSM.Set(row, col, value)
	})

	if err != nil {
		return nil, fmt.Errorf("failed to convert sparse matrix format: %v", err)
	}

	return newSM, nil
}

// Clone creates a deep copy of the sparse matrix
func (sm *SparseMatrix) Clone() *SparseMatrix {
	newSM := &SparseMatrix{
		rows:   sm.rows,
		cols:   sm.cols,
		format: sm.format,
		dtype:  sm.dtype,
		nnz:    sm.nnz,
	}

	// Deep copy data arrays
	switch sm.format {
	case COO:
		newSM.data = make([]interface{}, len(sm.data))
		copy(newSM.data, sm.data)
		newSM.rowIdx = make([]int, len(sm.rowIdx))
		copy(newSM.rowIdx, sm.rowIdx)
		newSM.colIdx = make([]int, len(sm.colIdx))
		copy(newSM.colIdx, sm.colIdx)
	case CSR, CSC:
		newSM.data = make([]interface{}, len(sm.data))
		copy(newSM.data, sm.data)
		newSM.indices = make([]int, len(sm.indices))
		copy(newSM.indices, sm.indices)
		newSM.indptr = make([]int, len(sm.indptr))
		copy(newSM.indptr, sm.indptr)
	}

	return newSM
}

// SortCOO sorts the COO format by row then column (in-place)
func (sm *SparseMatrix) SortCOO() error {
	if sm.format != COO {
		return fmt.Errorf("SortCOO can only be called on COO format matrices")
	}

	// Create slice of indices for sorting
	indices := make([]int, sm.nnz)
	for i := range indices {
		indices[i] = i
	}

	// Sort by (row, col) tuples
	sort.Slice(indices, func(i, j int) bool {
		idxI, idxJ := indices[i], indices[j]
		if sm.rowIdx[idxI] != sm.rowIdx[idxJ] {
			return sm.rowIdx[idxI] < sm.rowIdx[idxJ]
		}
		return sm.colIdx[idxI] < sm.colIdx[idxJ]
	})

	// Reorder data arrays
	newData := make([]interface{}, sm.nnz)
	newRowIdx := make([]int, sm.nnz)
	newColIdx := make([]int, sm.nnz)

	for i, idx := range indices {
		newData[i] = sm.data[idx]
		newRowIdx[i] = sm.rowIdx[idx]
		newColIdx[i] = sm.colIdx[idx]
	}

	sm.data = newData
	sm.rowIdx = newRowIdx
	sm.colIdx = newColIdx

	return nil
}

// EliminateDuplicates removes duplicate entries in COO format by summing values
func (sm *SparseMatrix) EliminateDuplicates() error {
	if sm.format != COO {
		return fmt.Errorf("EliminateDuplicates can only be called on COO format matrices")
	}

	if sm.nnz <= 1 {
		return nil // No duplicates possible
	}

	// First sort the matrix
	err := sm.SortCOO()
	if err != nil {
		return err
	}

	// Now eliminate duplicates
	newData := make([]interface{}, 0, sm.nnz)
	newRowIdx := make([]int, 0, sm.nnz)
	newColIdx := make([]int, 0, sm.nnz)

	currentRow, currentCol := sm.rowIdx[0], sm.colIdx[0]
	currentSum := sm.data[0]

	for i := 1; i < sm.nnz; i++ {
		if sm.rowIdx[i] == currentRow && sm.colIdx[i] == currentCol {
			// Duplicate found - sum the values
			currentSum = sm.addValues(currentSum, sm.data[i])
		} else {
			// Different position - save current sum and start new
			if !sm.isZero(currentSum) {
				newData = append(newData, currentSum)
				newRowIdx = append(newRowIdx, currentRow)
				newColIdx = append(newColIdx, currentCol)
			}
			currentRow, currentCol = sm.rowIdx[i], sm.colIdx[i]
			currentSum = sm.data[i]
		}
	}

	// Don't forget the last element
	if !sm.isZero(currentSum) {
		newData = append(newData, currentSum)
		newRowIdx = append(newRowIdx, currentRow)
		newColIdx = append(newColIdx, currentCol)
	}

	// Update the matrix
	sm.data = newData
	sm.rowIdx = newRowIdx
	sm.colIdx = newColIdx
	sm.nnz = len(newData)

	return nil
}

// Helper function to iterate over all non-zero elements
func (sm *SparseMatrix) iterateNonZeros(fn func(row, col int, value interface{}) error) error {
	switch sm.format {
	case COO:
		for i := 0; i < sm.nnz; i++ {
			if err := fn(sm.rowIdx[i], sm.colIdx[i], sm.data[i]); err != nil {
				return err
			}
		}
	case CSR:
		for row := 0; row < sm.rows; row++ {
			for j := sm.indptr[row]; j < sm.indptr[row+1]; j++ {
				if err := fn(row, sm.indices[j], sm.data[j]); err != nil {
					return err
				}
			}
		}
	case CSC:
		for col := 0; col < sm.cols; col++ {
			for i := sm.indptr[col]; i < sm.indptr[col+1]; i++ {
				if err := fn(sm.indices[i], col, sm.data[i]); err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("unsupported sparse format for iteration")
	}

	return nil
}

// Helper function to add two values of the same type
func (sm *SparseMatrix) addValues(a, b interface{}) interface{} {
	switch sm.dtype {
	case internal.Float64:
		return a.(float64) + b.(float64)
	case internal.Float32:
		return a.(float32) + b.(float32)
	case internal.Int64:
		return a.(int64) + b.(int64)
	default:
		// Fallback - shouldn't happen with proper type checking
		return a
	}
}

// Utility functions for creating common sparse matrices

// Identity creates a sparse identity matrix
func Identity(n int, opts *SparseOptions) (*SparseMatrix, error) {
	if opts == nil {
		opts = DefaultSparseOptions()
	}

	sm := NewSparseMatrix(n, n, opts)

	// Set diagonal elements to 1
	one := sm.getOneValue()
	for i := 0; i < n; i++ {
		err := sm.Set(i, i, one)
		if err != nil {
			return nil, fmt.Errorf("failed to set diagonal element: %v", err)
		}
	}

	return sm, nil
}

// Diag creates a sparse diagonal matrix from a slice of values
func Diag(values []interface{}, opts *SparseOptions) (*SparseMatrix, error) {
	if opts == nil {
		opts = DefaultSparseOptions()
	}

	n := len(values)
	sm := NewSparseMatrix(n, n, opts)

	for i, value := range values {
		if !sm.isZero(value) {
			err := sm.Set(i, i, value)
			if err != nil {
				return nil, fmt.Errorf("failed to set diagonal element %d: %v", i, err)
			}
		}
	}

	return sm, nil
}

// Random creates a sparse matrix with random non-zero elements
func Random(rows, cols int, density float64, opts *SparseOptions) (*SparseMatrix, error) {
	if opts == nil {
		opts = DefaultSparseOptions()
	}

	if density < 0.0 || density > 1.0 {
		return nil, fmt.Errorf("density must be between 0.0 and 1.0, got %f", density)
	}

	sm := NewSparseMatrix(rows, cols, opts)

	// Calculate number of non-zero elements
	nnzTarget := int(float64(rows*cols) * density)

	// For simplicity, we'll use a basic approach to generate random sparse matrix
	// In a production implementation, you'd want to use proper random number generation
	positions := make(map[[2]int]bool)
	count := 0

	// Simple deterministic pattern for testing - in practice use proper RNG
	step := 1.0 / density
	for i := 0; i < rows && count < nnzTarget; i++ {
		for j := 0; j < cols && count < nnzTarget; j++ {
			if float64(i*cols+j)*step >= float64(count) {
				if _, exists := positions[[2]int{i, j}]; !exists {
					positions[[2]int{i, j}] = true

					// Generate a non-zero value
					var value interface{}
					switch opts.DType {
					case internal.Float64:
						value = float64(i + j + 1) // Simple pattern
					case internal.Float32:
						value = float32(i + j + 1)
					case internal.Int64:
						value = int64(i + j + 1)
					}

					err := sm.Set(i, j, value)
					if err != nil {
						return nil, fmt.Errorf("failed to set random element: %v", err)
					}
					count++
				}
			}
		}
	}

	return sm, nil
}

// Helper function to get the "one" value for the matrix's dtype
func (sm *SparseMatrix) getOneValue() interface{} {
	switch sm.dtype {
	case internal.Float64:
		return 1.0
	case internal.Float32:
		return float32(1.0)
	case internal.Int64:
		return int64(1)
	default:
		return 1.0
	}
}
