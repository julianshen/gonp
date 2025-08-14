package array

import (
	"testing"

	"github.com/julianshen/gonp/internal"
)

func TestArrayCreationFromFloat64Slice(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	arr, err := NewArray(data)

	if err != nil {
		t.Fatalf("NewArray() error = %v, want nil", err)
	}

	if arr == nil {
		t.Fatal("NewArray() returned nil array")
	}

	expectedShape := internal.Shape{4}
	if !arr.Shape().Equal(expectedShape) {
		t.Errorf("Array shape = %v, want %v", arr.Shape(), expectedShape)
	}

	if arr.DType() != internal.Float64 {
		t.Errorf("Array dtype = %v, want %v", arr.DType(), internal.Float64)
	}

	if arr.Size() != 4 {
		t.Errorf("Array size = %d, want 4", arr.Size())
	}
}

func TestArrayCreationFromInt64Slice(t *testing.T) {
	data := []int64{1, 2, 3}
	arr, err := NewArray(data)

	if err != nil {
		t.Fatalf("NewArray() error = %v, want nil", err)
	}

	expectedShape := internal.Shape{3}
	if !arr.Shape().Equal(expectedShape) {
		t.Errorf("Array shape = %v, want %v", arr.Shape(), expectedShape)
	}

	if arr.DType() != internal.Int64 {
		t.Errorf("Array dtype = %v, want %v", arr.DType(), internal.Int64)
	}
}

func TestArrayCreationWithInvalidData(t *testing.T) {
	_, err := NewArray(nil)
	if err == nil {
		t.Error("NewArray(nil) should return error")
	}

	_, err = NewArray("invalid")
	if err == nil {
		t.Error("NewArray(string) should return error")
	}
}

func TestArrayWithSpecificShape(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}

	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v, want nil", err)
	}

	if !arr.Shape().Equal(shape) {
		t.Errorf("Array shape = %v, want %v", arr.Shape(), shape)
	}

	if arr.Size() != 6 {
		t.Errorf("Array size = %d, want 6", arr.Size())
	}
}

func TestArrayWithIncompatibleShape(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	shape := internal.Shape{2, 3} // Needs 6 elements, but only 5 provided

	_, err := NewArrayWithShape(data, shape)
	if err == nil {
		t.Error("NewArrayWithShape() should return error for incompatible shape")
	}
}

func TestArrayElementAccess1D(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	val := arr.At(0)
	if val != 1.0 {
		t.Errorf("At(0) = %v, want 1.0", val)
	}

	val = arr.At(3)
	if val != 4.0 {
		t.Errorf("At(3) = %v, want 4.0", val)
	}
}

func TestArrayElementAccess2D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	val := arr.At(0, 0)
	if val != 1.0 {
		t.Errorf("At(0, 0) = %v, want 1.0", val)
	}

	val = arr.At(1, 2)
	if val != 6.0 {
		t.Errorf("At(1, 2) = %v, want 6.0", val)
	}
}

func TestArrayElementAccessOutOfBounds(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test panics for out of bounds access
	defer func() {
		if r := recover(); r == nil {
			t.Error("At(3) should panic for out of bounds access")
		}
	}()
	arr.At(3)
}

func TestArrayElementAccessNegativeIndex(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test panics for negative index
	defer func() {
		if r := recover(); r == nil {
			t.Error("At(-1) should panic for negative index")
		}
	}()
	arr.At(-1)
}

func TestArrayElementAccessWrongDimensions(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	// Test panics for wrong number of indices
	defer func() {
		if r := recover(); r == nil {
			t.Error("At(1) should panic for wrong number of indices")
		}
	}()
	arr.At(1)
}

func TestArrayElementAccessTooManyDimensions(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	// Test panics for too many indices
	defer func() {
		if r := recover(); r == nil {
			t.Error("At(0, 1, 2) should panic for wrong number of indices")
		}
	}()
	arr.At(0, 1, 2)
}

func TestArrayElementSet1D(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	err = arr.Set(5.0, 1)
	if err != nil {
		t.Fatalf("Set(5.0, 1) error = %v", err)
	}

	val := arr.At(1)
	if val != 5.0 {
		t.Errorf("After Set, At(1) = %v, want 5.0", val)
	}
}

func TestArrayElementSet2D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	err = arr.Set(99.0, 1, 1)
	if err != nil {
		t.Fatalf("Set(99.0, 1, 1) error = %v", err)
	}

	val := arr.At(1, 1)
	if val != 99.0 {
		t.Errorf("After Set, At(1, 1) = %v, want 99.0", val)
	}
}

func TestArraySlice1D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	slice, err := arr.Slice(internal.NewRange(1, 4))
	if err != nil {
		t.Fatalf("Slice(1:4) error = %v", err)
	}

	expectedShape := internal.Shape{3}
	if !slice.Shape().Equal(expectedShape) {
		t.Errorf("Slice shape = %v, want %v", slice.Shape(), expectedShape)
	}

	val := slice.At(0)
	if val != 2.0 {
		t.Errorf("slice.At(0) = %v, want 2.0", val)
	}

	val = slice.At(2)
	if val != 4.0 {
		t.Errorf("slice.At(2) = %v, want 4.0", val)
	}
}

func TestArraySlice2D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	shape := internal.Shape{3, 4}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	// Slice rows 1:3, all columns
	slice, err := arr.Slice(internal.NewRange(1, 3), internal.NewRange(0, 4))
	if err != nil {
		t.Fatalf("Slice(1:3, 0:4) error = %v", err)
	}

	expectedShape := internal.Shape{2, 4}
	if !slice.Shape().Equal(expectedShape) {
		t.Errorf("Slice shape = %v, want %v", slice.Shape(), expectedShape)
	}

	val := slice.At(0, 0)
	if val != 5.0 { // Original arr[1,0] = 5
		t.Errorf("slice.At(0, 0) = %v, want 5.0", val)
	}
}

func TestArraySliceWithStep(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	slice, err := arr.Slice(internal.Range{Start: 0, Stop: 8, Step: 2})
	if err != nil {
		t.Fatalf("Slice(0:8:2) error = %v", err)
	}

	expectedShape := internal.Shape{4}
	if !slice.Shape().Equal(expectedShape) {
		t.Errorf("Slice shape = %v, want %v", slice.Shape(), expectedShape)
	}

	val := slice.At(0)
	if val != 1.0 {
		t.Errorf("slice.At(0) = %v, want 1.0", val)
	}

	val = slice.At(1)
	if val != 3.0 {
		t.Errorf("slice.At(1) = %v, want 3.0", val)
	}
}

func TestArraySliceInvalidRange(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	_, err = arr.Slice(internal.NewRange(-1, 2))
	if err == nil {
		t.Error("Slice with negative start should return error")
	}

	_, err = arr.Slice(internal.NewRange(0, 5))
	if err == nil {
		t.Error("Slice with stop > length should return error")
	}
}

func TestArrayReshape1D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	reshaped := arr.Reshape(internal.Shape{2, 3})

	expectedShape := internal.Shape{2, 3}
	if !reshaped.Shape().Equal(expectedShape) {
		t.Errorf("Reshaped shape = %v, want %v", reshaped.Shape(), expectedShape)
	}

	val := reshaped.At(0, 0)
	if val != 1.0 {
		t.Errorf("reshaped.At(0, 0) = %v, want 1.0", val)
	}

	val = reshaped.At(1, 2)
	if val != 6.0 {
		t.Errorf("reshaped.At(1, 2) = %v, want 6.0", val)
	}
}

func TestArrayReshape2D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	shape := internal.Shape{2, 4}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	reshaped := arr.Reshape(internal.Shape{4, 2})

	expectedShape := internal.Shape{4, 2}
	if !reshaped.Shape().Equal(expectedShape) {
		t.Errorf("Reshaped shape = %v, want %v", reshaped.Shape(), expectedShape)
	}

	val := reshaped.At(3, 1)
	if val != 8.0 {
		t.Errorf("reshaped.At(3, 1) = %v, want 8.0", val)
	}
}

func TestArrayReshapeIncompatibleSize(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Test panics for incompatible reshape
	defer func() {
		if r := recover(); r == nil {
			t.Error("Reshape with incompatible size should panic")
		}
	}()
	arr.Reshape(internal.Shape{2, 4}) // Needs 8 elements but only 6 available
}

func TestArrayTranspose2D(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	transposed, err := arr.Transpose()
	if err != nil {
		t.Fatalf("Transpose() error = %v", err)
	}

	expectedShape := internal.Shape{3, 2}
	if !transposed.Shape().Equal(expectedShape) {
		t.Errorf("Transposed shape = %v, want %v", transposed.Shape(), expectedShape)
	}

	val := transposed.At(0, 0)
	if val != 1.0 {
		t.Errorf("transposed.At(0, 0) = %v, want 1.0", val)
	}

	val = transposed.At(0, 1)
	if val != 4.0 { // Original arr[1,0] = 4
		t.Errorf("transposed.At(0, 1) = %v, want 4.0", val)
	}

	val = transposed.At(2, 1)
	if val != 6.0 { // Original arr[1,2] = 6
		t.Errorf("transposed.At(2, 1) = %v, want 6.0", val)
	}
}

func TestArrayTranspose1D(t *testing.T) {
	data := []float64{1, 2, 3}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	_, err = arr.Transpose()
	if err == nil {
		t.Error("Transpose of 1D array should return error")
	}
}

func TestArrayNdim(t *testing.T) {
	// Test 1D array
	data1d := []float64{1, 2, 3}
	arr1d, err := NewArray(data1d)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}
	if arr1d.Ndim() != 1 {
		t.Errorf("1D array Ndim() = %d, want 1", arr1d.Ndim())
	}

	// Test 2D array
	data2d := []float64{1, 2, 3, 4, 5, 6}
	shape2d := internal.Shape{2, 3}
	arr2d, err := NewArrayWithShape(data2d, shape2d)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}
	if arr2d.Ndim() != 2 {
		t.Errorf("2D array Ndim() = %d, want 2", arr2d.Ndim())
	}
}

func TestArrayFlatten(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := internal.Shape{2, 3}
	arr, err := NewArrayWithShape(data, shape)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	flattened := arr.Flatten()

	expectedShape := internal.Shape{6}
	if !flattened.Shape().Equal(expectedShape) {
		t.Errorf("Flattened shape = %v, want %v", flattened.Shape(), expectedShape)
	}

	// Check values are correct
	for i := 0; i < 6; i++ {
		val := flattened.At(i)
		expected := float64(i + 1)
		if val != expected {
			t.Errorf("Flattened.At(%d) = %v, want %v", i, val, expected)
		}
	}
}

func TestArrayCopy(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	copied := arr.Copy()

	// Check shapes match
	if !copied.Shape().Equal(arr.Shape()) {
		t.Errorf("Copied shape = %v, want %v", copied.Shape(), arr.Shape())
	}

	// Check values match
	for i := 0; i < arr.Size(); i++ {
		if copied.At(i) != arr.At(i) {
			t.Errorf("Copied.At(%d) = %v, want %v", i, copied.At(i), arr.At(i))
		}
	}

	// Modify original and check copy is unaffected
	arr.Set(99.0, 0)
	if copied.At(0) == 99.0 {
		t.Error("Copy should be independent of original array")
	}
}

func TestArrayToSlice(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	slice := arr.ToSlice()
	float64Slice, ok := slice.([]float64)
	if !ok {
		t.Fatalf("ToSlice() returned %T, want []float64", slice)
	}

	if len(float64Slice) != 4 {
		t.Errorf("ToSlice() length = %d, want 4", len(float64Slice))
	}

	for i, val := range float64Slice {
		expected := float64(i + 1)
		if val != expected {
			t.Errorf("ToSlice()[%d] = %v, want %v", i, val, expected)
		}
	}
}

func TestArrayString(t *testing.T) {
	// Test 1D array
	data1d := []float64{1, 2, 3}
	arr1d, err := NewArray(data1d)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	str1d := arr1d.String()
	if str1d == "" {
		t.Error("String() should not return empty string")
	}

	// Test 2D array (should show summary format)
	data2d := []float64{1, 2, 3, 4}
	shape2d := internal.Shape{2, 2}
	arr2d, err := NewArrayWithShape(data2d, shape2d)
	if err != nil {
		t.Fatalf("NewArrayWithShape() error = %v", err)
	}

	str2d := arr2d.String()
	if str2d == "" {
		t.Error("String() should not return empty string")
	}
}

func TestArrayAsType(t *testing.T) {
	data := []float64{1.5, 2.7, 3.9}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	// Convert to int64
	intArr := arr.AsType(internal.Int64)
	if intArr.DType() != internal.Int64 {
		t.Errorf("AsType(Int64) dtype = %v, want %v", intArr.DType(), internal.Int64)
	}

	// Check converted values
	expectedValues := []int64{1, 2, 3}
	for i, expected := range expectedValues {
		val := intArr.At(i)
		if val != expected {
			t.Errorf("AsType(Int64).At(%d) = %v, want %v", i, val, expected)
		}
	}
}

func TestArrayFill(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	arr, err := NewArray(data)
	if err != nil {
		t.Fatalf("NewArray() error = %v", err)
	}

	arr.Fill(9.0)

	// Check all values are 9.0
	for i := 0; i < arr.Size(); i++ {
		val := arr.At(i)
		if val != 9.0 {
			t.Errorf("After Fill(9.0), At(%d) = %v, want 9.0", i, val)
		}
	}
}

func TestZerosWithDType(t *testing.T) {
	shape := internal.Shape{2, 3}
	arr := Zeros(shape, internal.Int32)

	if arr.DType() != internal.Int32 {
		t.Errorf("Zeros dtype = %v, want %v", arr.DType(), internal.Int32)
	}

	if !arr.Shape().Equal(shape) {
		t.Errorf("Zeros shape = %v, want %v", arr.Shape(), shape)
	}

	// Check all values are zero
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			val := arr.At(i, j)
			if val != int32(0) {
				t.Errorf("Zeros.At(%d, %d) = %v, want 0", i, j, val)
			}
		}
	}
}

func TestOnesWithDType(t *testing.T) {
	shape := internal.Shape{2, 2}
	arr := Ones(shape, internal.Float32)

	if arr.DType() != internal.Float32 {
		t.Errorf("Ones dtype = %v, want %v", arr.DType(), internal.Float32)
	}

	// Check all values are one
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			val := arr.At(i, j)
			if val != float32(1.0) {
				t.Errorf("Ones.At(%d, %d) = %v, want 1.0", i, j, val)
			}
		}
	}
}

func TestFromSlice(t *testing.T) {
	data := []int64{10, 20, 30}
	arr, err := FromSlice(data)
	if err != nil {
		t.Fatalf("FromSlice() error = %v", err)
	}

	if arr.DType() != internal.Int64 {
		t.Errorf("FromSlice dtype = %v, want %v", arr.DType(), internal.Int64)
	}

	expectedShape := internal.Shape{3}
	if !arr.Shape().Equal(expectedShape) {
		t.Errorf("FromSlice shape = %v, want %v", arr.Shape(), expectedShape)
	}

	for i, expected := range data {
		val := arr.At(i)
		if val != expected {
			t.Errorf("FromSlice.At(%d) = %v, want %v", i, val, expected)
		}
	}
}

func TestFull(t *testing.T) {
	shape := internal.Shape{2, 2}
	value := 42.5
	arr := Full(shape, value, internal.Float64)

	if arr.DType() != internal.Float64 {
		t.Errorf("Full dtype = %v, want %v", arr.DType(), internal.Float64)
	}

	// Check all values are the fill value
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			val := arr.At(i, j)
			if val != value {
				t.Errorf("Full.At(%d, %d) = %v, want %v", i, j, val, value)
			}
		}
	}
}

func TestEmpty(t *testing.T) {
	shape := internal.Shape{3, 2}
	arr := Empty(shape, internal.Bool)

	if arr.DType() != internal.Bool {
		t.Errorf("Empty dtype = %v, want %v", arr.DType(), internal.Bool)
	}

	if !arr.Shape().Equal(shape) {
		t.Errorf("Empty shape = %v, want %v", arr.Shape(), shape)
	}

	if arr.Size() != 6 {
		t.Errorf("Empty size = %d, want 6", arr.Size())
	}
}

func TestLinspace(t *testing.T) {
	arr := Linspace(0, 10, 5)

	expectedShape := internal.Shape{5}
	if !arr.Shape().Equal(expectedShape) {
		t.Errorf("Linspace shape = %v, want %v", arr.Shape(), expectedShape)
	}

	expectedValues := []float64{0, 2.5, 5, 7.5, 10}
	for i, expected := range expectedValues {
		val := arr.At(i)
		if val != expected {
			t.Errorf("Linspace.At(%d) = %v, want %v", i, val, expected)
		}
	}
}
