// Package sparse provides sparse matrix implementations for GoNP.
// Sparse matrices are memory-efficient representations of matrices where most elements are zero.
package sparse

import (
	"fmt"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// SparseFormat represents different sparse matrix storage formats
type SparseFormat int

const (
	// COO (Coordinate format) - stores (row, col, value) triplets
	COO SparseFormat = iota
	// CSR (Compressed Sparse Row) - efficient for row-wise operations
	CSR
	// CSC (Compressed Sparse Column) - efficient for column-wise operations
	CSC
)

// SparseMatrix represents a sparse matrix with configurable storage format
type SparseMatrix struct {
	rows   int            // Number of rows
	cols   int            // Number of columns
	format SparseFormat   // Storage format
	nnz    int            // Number of non-zero elements
	dtype  internal.DType // Data type

	// Storage arrays - usage depends on format
	data    []interface{} // Non-zero values
	indices []int         // Column indices (CSR) or row indices (CSC)
	indptr  []int         // Row pointers (CSR) or column pointers (CSC)
	rowIdx  []int         // Row indices (COO only)
	colIdx  []int         // Column indices (COO only)
}

// SparseOptions configures sparse matrix creation
type SparseOptions struct {
	Format   SparseFormat
	DType    internal.DType
	Capacity int // Initial capacity for non-zero elements
}

// DefaultSparseOptions returns default options for sparse matrix creation
func DefaultSparseOptions() *SparseOptions {
	return &SparseOptions{
		Format:   CSR, // Default to CSR format
		DType:    internal.Float64,
		Capacity: 100,
	}
}

// NewSparseMatrix creates a new sparse matrix with specified dimensions
func NewSparseMatrix(rows, cols int, opts *SparseOptions) *SparseMatrix {
	if opts == nil {
		opts = DefaultSparseOptions()
	}

	if rows <= 0 || cols <= 0 {
		panic("sparse matrix dimensions must be positive")
	}

	sm := &SparseMatrix{
		rows:   rows,
		cols:   cols,
		format: opts.Format,
		dtype:  opts.DType,
		nnz:    0,
	}

	// Initialize storage based on format
	switch opts.Format {
	case COO:
		sm.data = make([]interface{}, 0, opts.Capacity)
		sm.rowIdx = make([]int, 0, opts.Capacity)
		sm.colIdx = make([]int, 0, opts.Capacity)
	case CSR:
		sm.data = make([]interface{}, 0, opts.Capacity)
		sm.indices = make([]int, 0, opts.Capacity)
		sm.indptr = make([]int, rows+1)
	case CSC:
		sm.data = make([]interface{}, 0, opts.Capacity)
		sm.indices = make([]int, 0, opts.Capacity)
		sm.indptr = make([]int, cols+1)
	}

	return sm
}

// Shape returns the dimensions of the sparse matrix
func (sm *SparseMatrix) Shape() (int, int) {
	return sm.rows, sm.cols
}

// Nnz returns the number of non-zero elements
func (sm *SparseMatrix) Nnz() int {
	return sm.nnz
}

// Format returns the storage format
func (sm *SparseMatrix) Format() SparseFormat {
	return sm.format
}

// Density returns the density (fraction of non-zero elements)
func (sm *SparseMatrix) Density() float64 {
	if sm.rows == 0 || sm.cols == 0 {
		return 0.0
	}
	return float64(sm.nnz) / float64(sm.rows*sm.cols)
}

// Set sets a value at the specified position
func (sm *SparseMatrix) Set(row, col int, value interface{}) error {
	if row < 0 || row >= sm.rows || col < 0 || col >= sm.cols {
		return fmt.Errorf("index (%d, %d) out of bounds for matrix with shape (%d, %d)",
			row, col, sm.rows, sm.cols)
	}

	// Convert value to the matrix's dtype
	convertedValue, err := sm.convertValue(value)
	if err != nil {
		return fmt.Errorf("failed to convert value: %v", err)
	}

	// Check if value is effectively zero
	if sm.isZero(convertedValue) {
		return sm.removeElement(row, col)
	}

	return sm.setNonZero(row, col, convertedValue)
}

// Get retrieves the value at the specified position
func (sm *SparseMatrix) Get(row, col int) (interface{}, error) {
	if row < 0 || row >= sm.rows || col < 0 || col >= sm.cols {
		return nil, fmt.Errorf("index (%d, %d) out of bounds for matrix with shape (%d, %d)",
			row, col, sm.rows, sm.cols)
	}

	switch sm.format {
	case COO:
		return sm.getCOO(row, col), nil
	case CSR:
		return sm.getCSR(row, col), nil
	case CSC:
		return sm.getCSC(row, col), nil
	default:
		return nil, fmt.Errorf("unsupported sparse format: %d", sm.format)
	}
}

// ToArray converts the sparse matrix to a dense array
func (sm *SparseMatrix) ToArray() (*array.Array, error) {
	// Create dense array filled with zeros
	denseData := make([]interface{}, sm.rows*sm.cols)
	zeroValue := sm.getZeroValue()

	for i := range denseData {
		denseData[i] = zeroValue
	}

	// Fill in non-zero values
	switch sm.format {
	case COO:
		for i := 0; i < sm.nnz; i++ {
			row, col := sm.rowIdx[i], sm.colIdx[i]
			idx := row*sm.cols + col
			denseData[idx] = sm.data[i]
		}
	case CSR:
		for row := 0; row < sm.rows; row++ {
			for j := sm.indptr[row]; j < sm.indptr[row+1]; j++ {
				col := sm.indices[j]
				idx := row*sm.cols + col
				denseData[idx] = sm.data[j]
			}
		}
	case CSC:
		for col := 0; col < sm.cols; col++ {
			for i := sm.indptr[col]; i < sm.indptr[col+1]; i++ {
				row := sm.indices[i]
				idx := row*sm.cols + col
				denseData[idx] = sm.data[i]
			}
		}
	}

	// Create array with proper shape
	arr, err := array.FromSlice(denseData)
	if err != nil {
		return nil, fmt.Errorf("failed to create array: %v", err)
	}

	return arr.Reshape([]int{sm.rows, sm.cols}), nil
}

// String returns a string representation of the sparse matrix
func (sm *SparseMatrix) String() string {
	return fmt.Sprintf("SparseMatrix<%dx%d, nnz=%d, density=%.4f, format=%s>",
		sm.rows, sm.cols, sm.nnz, sm.Density(), sm.formatString())
}

// Helper methods

func (sm *SparseMatrix) formatString() string {
	switch sm.format {
	case COO:
		return "COO"
	case CSR:
		return "CSR"
	case CSC:
		return "CSC"
	default:
		return "Unknown"
	}
}

func (sm *SparseMatrix) convertValue(value interface{}) (interface{}, error) {
	switch sm.dtype {
	case internal.Float64:
		switch v := value.(type) {
		case float64:
			return v, nil
		case float32:
			return float64(v), nil
		case int:
			return float64(v), nil
		case int64:
			return float64(v), nil
		default:
			return nil, fmt.Errorf("cannot convert %T to float64", value)
		}
	case internal.Float32:
		switch v := value.(type) {
		case float32:
			return v, nil
		case float64:
			return float32(v), nil
		case int:
			return float32(v), nil
		case int64:
			return float32(v), nil
		default:
			return nil, fmt.Errorf("cannot convert %T to float32", value)
		}
	case internal.Int64:
		switch v := value.(type) {
		case int64:
			return v, nil
		case int:
			return int64(v), nil
		case float64:
			return int64(v), nil
		case float32:
			return int64(v), nil
		default:
			return nil, fmt.Errorf("cannot convert %T to int64", value)
		}
	default:
		return nil, fmt.Errorf("unsupported dtype: %d", sm.dtype)
	}
}

func (sm *SparseMatrix) isZero(value interface{}) bool {
	switch v := value.(type) {
	case float64:
		return v == 0.0
	case float32:
		return v == 0.0
	case int64:
		return v == 0
	case int:
		return v == 0
	default:
		return false
	}
}

func (sm *SparseMatrix) getZeroValue() interface{} {
	switch sm.dtype {
	case internal.Float64:
		return 0.0
	case internal.Float32:
		return float32(0.0)
	case internal.Int64:
		return int64(0)
	default:
		return 0.0
	}
}

// Format-specific implementations

func (sm *SparseMatrix) setNonZero(row, col int, value interface{}) error {
	switch sm.format {
	case COO:
		return sm.setCOO(row, col, value)
	case CSR:
		return sm.setCSR(row, col, value)
	case CSC:
		return sm.setCSC(row, col, value)
	default:
		return fmt.Errorf("unsupported sparse format: %d", sm.format)
	}
}

func (sm *SparseMatrix) removeElement(row, col int) error {
	switch sm.format {
	case COO:
		return sm.removeCOO(row, col)
	case CSR:
		return sm.removeCSR(row, col)
	case CSC:
		return sm.removeCSC(row, col)
	default:
		return fmt.Errorf("unsupported sparse format: %d", sm.format)
	}
}

// COO format methods
func (sm *SparseMatrix) setCOO(row, col int, value interface{}) error {
	// Check if element already exists
	for i := 0; i < sm.nnz; i++ {
		if sm.rowIdx[i] == row && sm.colIdx[i] == col {
			sm.data[i] = value
			return nil
		}
	}

	// Add new element
	sm.data = append(sm.data, value)
	sm.rowIdx = append(sm.rowIdx, row)
	sm.colIdx = append(sm.colIdx, col)
	sm.nnz++

	return nil
}

func (sm *SparseMatrix) getCOO(row, col int) interface{} {
	for i := 0; i < sm.nnz; i++ {
		if sm.rowIdx[i] == row && sm.colIdx[i] == col {
			return sm.data[i]
		}
	}
	return sm.getZeroValue()
}

func (sm *SparseMatrix) removeCOO(row, col int) error {
	for i := 0; i < sm.nnz; i++ {
		if sm.rowIdx[i] == row && sm.colIdx[i] == col {
			// Remove element by swapping with last and shrinking
			sm.data[i] = sm.data[sm.nnz-1]
			sm.rowIdx[i] = sm.rowIdx[sm.nnz-1]
			sm.colIdx[i] = sm.colIdx[sm.nnz-1]

			sm.data = sm.data[:sm.nnz-1]
			sm.rowIdx = sm.rowIdx[:sm.nnz-1]
			sm.colIdx = sm.colIdx[:sm.nnz-1]
			sm.nnz--

			return nil
		}
	}
	return nil // Element not found, nothing to remove
}

// CSR format methods
func (sm *SparseMatrix) setCSR(row, col int, value interface{}) error {
	// Find insertion point in the row
	start := sm.indptr[row]
	end := sm.indptr[row+1]

	// Binary search for column position
	insertPos := start
	for i := start; i < end; i++ {
		if sm.indices[i] == col {
			// Update existing element
			sm.data[i] = value
			return nil
		} else if sm.indices[i] > col {
			insertPos = i
			break
		} else {
			insertPos = i + 1
		}
	}

	// Insert new element
	sm.data = append(sm.data[:insertPos], append([]interface{}{value}, sm.data[insertPos:]...)...)
	sm.indices = append(sm.indices[:insertPos], append([]int{col}, sm.indices[insertPos:]...)...)

	// Update row pointers
	for i := row + 1; i <= sm.rows; i++ {
		sm.indptr[i]++
	}

	sm.nnz++
	return nil
}

func (sm *SparseMatrix) getCSR(row, col int) interface{} {
	start := sm.indptr[row]
	end := sm.indptr[row+1]

	for i := start; i < end; i++ {
		if sm.indices[i] == col {
			return sm.data[i]
		}
	}

	return sm.getZeroValue()
}

func (sm *SparseMatrix) removeCSR(row, col int) error {
	start := sm.indptr[row]
	end := sm.indptr[row+1]

	for i := start; i < end; i++ {
		if sm.indices[i] == col {
			// Remove element
			sm.data = append(sm.data[:i], sm.data[i+1:]...)
			sm.indices = append(sm.indices[:i], sm.indices[i+1:]...)

			// Update row pointers
			for j := row + 1; j <= sm.rows; j++ {
				sm.indptr[j]--
			}

			sm.nnz--
			return nil
		}
	}

	return nil // Element not found
}

// CSC format methods (similar to CSR but column-oriented)
func (sm *SparseMatrix) setCSC(row, col int, value interface{}) error {
	start := sm.indptr[col]
	end := sm.indptr[col+1]

	insertPos := start
	for i := start; i < end; i++ {
		if sm.indices[i] == row {
			sm.data[i] = value
			return nil
		} else if sm.indices[i] > row {
			insertPos = i
			break
		} else {
			insertPos = i + 1
		}
	}

	sm.data = append(sm.data[:insertPos], append([]interface{}{value}, sm.data[insertPos:]...)...)
	sm.indices = append(sm.indices[:insertPos], append([]int{row}, sm.indices[insertPos:]...)...)

	for i := col + 1; i <= sm.cols; i++ {
		sm.indptr[i]++
	}

	sm.nnz++
	return nil
}

func (sm *SparseMatrix) getCSC(row, col int) interface{} {
	start := sm.indptr[col]
	end := sm.indptr[col+1]

	for i := start; i < end; i++ {
		if sm.indices[i] == row {
			return sm.data[i]
		}
	}

	return sm.getZeroValue()
}

func (sm *SparseMatrix) removeCSC(row, col int) error {
	start := sm.indptr[col]
	end := sm.indptr[col+1]

	for i := start; i < end; i++ {
		if sm.indices[i] == row {
			sm.data = append(sm.data[:i], sm.data[i+1:]...)
			sm.indices = append(sm.indices[:i], sm.indices[i+1:]...)

			for j := col + 1; j <= sm.cols; j++ {
				sm.indptr[j]--
			}

			sm.nnz--
			return nil
		}
	}

	return nil
}
