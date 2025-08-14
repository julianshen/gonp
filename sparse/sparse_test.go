package sparse

import (
	"math"
	"testing"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

func TestSparseMatrixBasics(t *testing.T) {
	tests := []struct {
		name   string
		format SparseFormat
	}{
		{"COO Format", COO},
		{"CSR Format", CSR},
		{"CSC Format", CSC},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := &SparseOptions{
				Format: tt.format,
				DType:  internal.Float64,
			}

			sm := NewSparseMatrix(3, 3, opts)

			// Test basic properties
			rows, cols := sm.Shape()
			if rows != 3 || cols != 3 {
				t.Errorf("Expected shape (3, 3), got (%d, %d)", rows, cols)
			}

			if sm.Nnz() != 0 {
				t.Errorf("Expected 0 non-zeros, got %d", sm.Nnz())
			}

			if sm.Format() != tt.format {
				t.Errorf("Expected format %d, got %d", tt.format, sm.Format())
			}

			// Test setting and getting values
			err := sm.Set(1, 1, 5.0)
			if err != nil {
				t.Errorf("Failed to set value: %v", err)
			}

			val, err := sm.Get(1, 1)
			if err != nil {
				t.Errorf("Failed to get value: %v", err)
			}
			if val.(float64) != 5.0 {
				t.Errorf("Expected 5.0, got %v", val)
			}

			if sm.Nnz() != 1 {
				t.Errorf("Expected 1 non-zero, got %d", sm.Nnz())
			}
		})
	}
}

func TestSparseMatrixSetGet(t *testing.T) {
	sm := NewSparseMatrix(4, 4, DefaultSparseOptions())

	// Set some values
	testCases := []struct {
		row, col int
		value    float64
	}{
		{0, 0, 1.0},
		{1, 2, 2.5},
		{2, 1, -3.0},
		{3, 3, 4.0},
	}

	for _, tc := range testCases {
		err := sm.Set(tc.row, tc.col, tc.value)
		if err != nil {
			t.Errorf("Failed to set (%d, %d) = %f: %v", tc.row, tc.col, tc.value, err)
		}
	}

	// Check values
	for _, tc := range testCases {
		val, err := sm.Get(tc.row, tc.col)
		if err != nil {
			t.Errorf("Failed to get (%d, %d): %v", tc.row, tc.col, err)
		}
		if val.(float64) != tc.value {
			t.Errorf("Expected (%d, %d) = %f, got %f", tc.row, tc.col, tc.value, val.(float64))
		}
	}

	// Check zero values
	val, err := sm.Get(0, 1)
	if err != nil {
		t.Errorf("Failed to get zero element: %v", err)
	}
	if val.(float64) != 0.0 {
		t.Errorf("Expected zero, got %f", val.(float64))
	}
}

func TestSparseMatrixDensity(t *testing.T) {
	sm := NewSparseMatrix(10, 10, DefaultSparseOptions())

	// Initially empty
	if sm.Density() != 0.0 {
		t.Errorf("Expected density 0.0, got %f", sm.Density())
	}

	// Add 5 elements
	for i := 0; i < 5; i++ {
		sm.Set(i, i, float64(i+1))
	}

	expectedDensity := 5.0 / 100.0 // 5 non-zeros out of 100 total
	if math.Abs(sm.Density()-expectedDensity) > 1e-10 {
		t.Errorf("Expected density %f, got %f", expectedDensity, sm.Density())
	}
}

func TestSparseMatrixConversion(t *testing.T) {
	// Create test array
	data := []float64{
		1, 0, 3,
		0, 5, 0,
		7, 0, 9,
	}
	arr, err := array.FromSlice(data)
	if err != nil {
		t.Fatalf("Failed to create array: %v", err)
	}
	arr = arr.Reshape([]int{3, 3})

	// Convert to sparse
	sm, err := FromArray(arr, CSR)
	if err != nil {
		t.Fatalf("Failed to convert to sparse: %v", err)
	}

	// Check non-zero count
	if sm.Nnz() != 5 {
		t.Errorf("Expected 5 non-zeros, got %d", sm.Nnz())
	}

	// Convert back to dense
	dense, err := sm.ToArray()
	if err != nil {
		t.Fatalf("Failed to convert to dense: %v", err)
	}

	// Check values
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected := arr.At(i, j).(float64)
			actual := dense.At(i, j).(float64)
			if expected != actual {
				t.Errorf("Mismatch at (%d, %d): expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestSparseMatrixArithmetic(t *testing.T) {
	// Create two sparse matrices
	sm1 := NewSparseMatrix(2, 2, DefaultSparseOptions())
	sm1.Set(0, 0, 1.0)
	sm1.Set(0, 1, 2.0)
	sm1.Set(1, 1, 3.0)

	sm2 := NewSparseMatrix(2, 2, DefaultSparseOptions())
	sm2.Set(0, 0, 4.0)
	sm2.Set(1, 0, 5.0)
	sm2.Set(1, 1, 6.0)

	// Test addition
	result, err := sm1.Add(sm2)
	if err != nil {
		t.Fatalf("Failed to add matrices: %v", err)
	}

	// Check results
	val, _ := result.Get(0, 0)
	if val.(float64) != 5.0 { // 1 + 4
		t.Errorf("Expected (0,0) = 5.0, got %f", val.(float64))
	}

	val, _ = result.Get(0, 1)
	if val.(float64) != 2.0 { // 2 + 0
		t.Errorf("Expected (0,1) = 2.0, got %f", val.(float64))
	}

	val, _ = result.Get(1, 0)
	if val.(float64) != 5.0 { // 0 + 5
		t.Errorf("Expected (1,0) = 5.0, got %f", val.(float64))
	}

	val, _ = result.Get(1, 1)
	if val.(float64) != 9.0 { // 3 + 6
		t.Errorf("Expected (1,1) = 9.0, got %f", val.(float64))
	}

	// Test scalar multiplication
	scaled, err := sm1.ScalarMultiply(2.0)
	if err != nil {
		t.Fatalf("Failed to multiply by scalar: %v", err)
	}

	val, _ = scaled.Get(0, 0)
	if val.(float64) != 2.0 { // 1 * 2
		t.Errorf("Expected scaled (0,0) = 2.0, got %f", val.(float64))
	}

	val, _ = scaled.Get(0, 1)
	if val.(float64) != 4.0 { // 2 * 2
		t.Errorf("Expected scaled (0,1) = 4.0, got %f", val.(float64))
	}
}

func TestSparseMatrixMultiplication(t *testing.T) {
	// Create A = [[1, 2], [3, 0]]
	A := NewSparseMatrix(2, 2, DefaultSparseOptions())
	A.Set(0, 0, 1.0)
	A.Set(0, 1, 2.0)
	A.Set(1, 0, 3.0)

	// Create B = [[4, 0], [5, 6]]
	B := NewSparseMatrix(2, 2, DefaultSparseOptions())
	B.Set(0, 0, 4.0)
	B.Set(1, 0, 5.0)
	B.Set(1, 1, 6.0)

	// C = A * B should be [[14, 12], [12, 0]]
	C, err := A.MatMul(B)
	if err != nil {
		t.Fatalf("Failed to multiply matrices: %v", err)
	}

	expected := [][]float64{
		{14.0, 12.0},
		{12.0, 0.0},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			val, _ := C.Get(i, j)
			if val.(float64) != expected[i][j] {
				t.Errorf("Expected C[%d][%d] = %f, got %f", i, j, expected[i][j], val.(float64))
			}
		}
	}
}

func TestSparseMatrixTranspose(t *testing.T) {
	// Create 2x3 matrix
	sm := NewSparseMatrix(2, 3, DefaultSparseOptions())
	sm.Set(0, 1, 1.0)
	sm.Set(0, 2, 2.0)
	sm.Set(1, 0, 3.0)
	sm.Set(1, 2, 4.0)

	// Transpose should be 3x2
	transposed := sm.Transpose()

	rows, cols := transposed.Shape()
	if rows != 3 || cols != 2 {
		t.Errorf("Expected transposed shape (3, 2), got (%d, %d)", rows, cols)
	}

	// Check values
	val, _ := transposed.Get(1, 0)
	if val.(float64) != 1.0 {
		t.Errorf("Expected transposed[1][0] = 1.0, got %f", val.(float64))
	}

	val, _ = transposed.Get(0, 1)
	if val.(float64) != 3.0 {
		t.Errorf("Expected transposed[0][1] = 3.0, got %f", val.(float64))
	}
}

func TestSparseMatrixNorms(t *testing.T) {
	sm := NewSparseMatrix(3, 3, DefaultSparseOptions())
	sm.Set(0, 0, 3.0)
	sm.Set(0, 1, 4.0)
	sm.Set(1, 1, 5.0)
	sm.Set(2, 2, 12.0)

	// Test Frobenius norm: sqrt(3^2 + 4^2 + 5^2 + 12^2) = sqrt(9 + 16 + 25 + 144) = sqrt(194)
	frobNorm, err := sm.Norm(Frobenius)
	if err != nil {
		t.Errorf("Failed to compute Frobenius norm: %v", err)
	}
	expected := math.Sqrt(194.0)
	if math.Abs(frobNorm-expected) > 1e-10 {
		t.Errorf("Expected Frobenius norm %f, got %f", expected, frobNorm)
	}

	// Test 1-norm (max column sum)
	oneNorm, err := sm.Norm(OneNorm)
	if err != nil {
		t.Errorf("Failed to compute 1-norm: %v", err)
	}
	// Column sums: [3, 9, 12] -> max = 12
	if oneNorm != 12.0 {
		t.Errorf("Expected 1-norm 12.0, got %f", oneNorm)
	}

	// Test infinity-norm (max row sum)
	infNorm, err := sm.Norm(InfNorm)
	if err != nil {
		t.Errorf("Failed to compute infinity norm: %v", err)
	}
	// Row sums: [7, 5, 12] -> max = 12
	if infNorm != 12.0 {
		t.Errorf("Expected infinity norm 12.0, got %f", infNorm)
	}
}

func TestSparseMatrixTrace(t *testing.T) {
	sm := NewSparseMatrix(3, 3, DefaultSparseOptions())
	sm.Set(0, 0, 1.0)
	sm.Set(1, 1, 2.0)
	sm.Set(2, 2, 3.0)
	sm.Set(0, 1, 4.0) // off-diagonal element

	trace, err := sm.Trace()
	if err != nil {
		t.Errorf("Failed to compute trace: %v", err)
	}

	if trace.(float64) != 6.0 { // 1 + 2 + 3
		t.Errorf("Expected trace 6.0, got %f", trace.(float64))
	}
}

func TestSparseMatrixSpMV(t *testing.T) {
	// Create 3x3 matrix
	sm := NewSparseMatrix(3, 3, DefaultSparseOptions())
	sm.Set(0, 0, 1.0)
	sm.Set(0, 2, 3.0)
	sm.Set(1, 1, 2.0)
	sm.Set(2, 0, 4.0)
	sm.Set(2, 2, 5.0)

	// Create vector [1, 2, 3]
	vec, err := array.FromSlice([]float64{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("Failed to create vector: %v", err)
	}

	// Multiply: result should be [10, 4, 19]
	result, err := sm.SpMV(vec)
	if err != nil {
		t.Fatalf("Failed to perform SpMV: %v", err)
	}

	expected := []float64{10.0, 4.0, 19.0}
	for i, exp := range expected {
		if result.At(i).(float64) != exp {
			t.Errorf("Expected result[%d] = %f, got %f", i, exp, result.At(i).(float64))
		}
	}
}

func TestSparseMatrixFormatConversion(t *testing.T) {
	// Create COO matrix
	coo := NewSparseMatrix(3, 3, &SparseOptions{Format: COO, DType: internal.Float64})
	coo.Set(0, 0, 1.0)
	coo.Set(1, 1, 2.0)
	coo.Set(2, 0, 3.0)
	coo.Set(0, 2, 4.0)

	// Convert to CSR
	csr, err := coo.ToFormat(CSR)
	if err != nil {
		t.Fatalf("Failed to convert COO to CSR: %v", err)
	}

	// Convert to CSC
	csc, err := coo.ToFormat(CSC)
	if err != nil {
		t.Fatalf("Failed to convert COO to CSC: %v", err)
	}

	// All should have same values
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			cooVal, _ := coo.Get(i, j)
			csrVal, _ := csr.Get(i, j)
			cscVal, _ := csc.Get(i, j)

			if cooVal != csrVal || cooVal != cscVal {
				t.Errorf("Format conversion mismatch at (%d, %d): COO=%v, CSR=%v, CSC=%v",
					i, j, cooVal, csrVal, cscVal)
			}
		}
	}
}

func TestSparseMatrixUtilities(t *testing.T) {
	// Test Identity matrix
	identity, err := Identity(3, DefaultSparseOptions())
	if err != nil {
		t.Fatalf("Failed to create identity matrix: %v", err)
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			val, _ := identity.Get(i, j)
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if val.(float64) != expected {
				t.Errorf("Identity matrix error at (%d, %d): expected %f, got %f",
					i, j, expected, val.(float64))
			}
		}
	}

	// Test diagonal matrix
	diagValues := []interface{}{1.0, 2.0, 3.0}
	diag, err := Diag(diagValues, DefaultSparseOptions())
	if err != nil {
		t.Fatalf("Failed to create diagonal matrix: %v", err)
	}

	for i := 0; i < 3; i++ {
		val, _ := diag.Get(i, i)
		if val.(float64) != diagValues[i].(float64) {
			t.Errorf("Diagonal matrix error at (%d, %d): expected %f, got %f",
				i, i, diagValues[i].(float64), val.(float64))
		}
	}
}

func TestSparseMatrixEdgeCases(t *testing.T) {
	sm := NewSparseMatrix(2, 2, DefaultSparseOptions())

	// Test out of bounds
	err := sm.Set(-1, 0, 1.0)
	if err == nil {
		t.Error("Expected error for negative row index")
	}

	err = sm.Set(0, 5, 1.0)
	if err == nil {
		t.Error("Expected error for out of bounds column index")
	}

	_, err = sm.Get(2, 0)
	if err == nil {
		t.Error("Expected error for out of bounds row index")
	}

	// Test setting zero (should remove element)
	sm.Set(0, 0, 5.0)
	if sm.Nnz() != 1 {
		t.Errorf("Expected 1 non-zero, got %d", sm.Nnz())
	}

	sm.Set(0, 0, 0.0)
	if sm.Nnz() != 0 {
		t.Errorf("Expected 0 non-zeros after setting to zero, got %d", sm.Nnz())
	}
}

// Benchmark tests
func BenchmarkSparseMatrixSetCSR(b *testing.B) {
	sm := NewSparseMatrix(1000, 1000, &SparseOptions{Format: CSR, DType: internal.Float64})
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		row := i % 1000
		col := (i * 7) % 1000 // Some pseudo-random pattern
		sm.Set(row, col, float64(i))
	}
}

func BenchmarkSparseMatrixGetCSR(b *testing.B) {
	sm := NewSparseMatrix(1000, 1000, &SparseOptions{Format: CSR, DType: internal.Float64})

	// Pre-populate with some values
	for i := 0; i < 1000; i++ {
		sm.Set(i, i, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row := i % 1000
		col := i % 1000
		sm.Get(row, col)
	}
}

func BenchmarkSparseMatrixMultiplication(b *testing.B) {
	// Create two sparse matrices
	A := NewSparseMatrix(100, 100, DefaultSparseOptions())
	B := NewSparseMatrix(100, 100, DefaultSparseOptions())

	// Fill with some values (about 10% density)
	for i := 0; i < 100; i++ {
		for j := 0; j < 10; j++ {
			A.Set(i, j*10, float64(i*j+1))
			B.Set(j*10, i, float64(i+j+1))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		A.MatMul(B)
	}
}
