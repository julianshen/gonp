package array

import (
	"fmt"

	"github.com/julianshen/gonp/internal"
)

// Concatenate joins arrays along an existing axis
// arrays: slice of arrays to concatenate
// axis: axis along which to concatenate
func Concatenate(arrays []*Array, axis int) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("cannot concatenate empty list of arrays")
	}

	first := arrays[0]
	if first == nil {
		return nil, fmt.Errorf("first array cannot be nil")
	}

	// Validate axis
	if axis < 0 || axis >= first.Ndim() {
		return nil, fmt.Errorf("axis %d out of bounds for array with %d dimensions", axis, first.Ndim())
	}

	if len(arrays) == 1 {
		return arrays[0].Copy(), nil
	}

	// Check all arrays have same dtype and compatible shapes
	dtype := first.DType()
	firstShape := first.Shape()
	totalSize := firstShape[axis]

	for i, arr := range arrays[1:] {
		if arr == nil {
			return nil, fmt.Errorf("array at index %d is nil", i+1)
		}

		if arr.DType() != dtype {
			return nil, fmt.Errorf("all arrays must have same dtype: expected %v, got %v at index %d", dtype, arr.DType(), i+1)
		}

		arrShape := arr.Shape()
		if len(arrShape) != len(firstShape) {
			return nil, fmt.Errorf("all arrays must have same number of dimensions: expected %d, got %d at index %d", len(firstShape), len(arrShape), i+1)
		}

		// Check all dimensions except concat axis are the same
		for j := range firstShape {
			if j != axis && arrShape[j] != firstShape[j] {
				return nil, fmt.Errorf("arrays must have same shape except on concatenation axis: expected %v, got %v at index %d", firstShape, arrShape, i+1)
			}
		}

		totalSize += arrShape[axis]
	}

	// Create result shape
	resultShape := make(internal.Shape, len(firstShape))
	copy(resultShape, firstShape)
	resultShape[axis] = totalSize

	// Create result array
	resultSize := resultShape.Size()
	data := allocateSliceForDType(resultSize, dtype)
	storage := internal.NewTypedStorage(data, dtype)
	stride := calculateStride(resultShape)

	result := &Array{
		storage: storage,
		shape:   resultShape,
		stride:  stride,
		dtype:   dtype,
		offset:  0,
	}

	// Copy data from all arrays
	currentPos := 0
	for _, arr := range arrays {
		arrSize := arr.Shape()[axis]

		// Copy elements from this array to result
		if err := copyArrayData(result, arr, currentPos, axis); err != nil {
			return nil, fmt.Errorf("failed to copy array data: %v", err)
		}

		currentPos += arrSize
	}

	return result, nil
}

// VStack stacks arrays vertically (row-wise)
// Equivalent to concatenation along axis 0 after 1-D arrays are reshaped to (1,N)
func VStack(arrays []*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("cannot stack empty list of arrays")
	}

	// For 1D arrays, reshape to (1, N)
	reshapedArrays := make([]*Array, len(arrays))
	for i, arr := range arrays {
		if arr == nil {
			return nil, fmt.Errorf("array at index %d is nil", i)
		}

		if arr.Ndim() == 1 {
			// Reshape 1D array to 2D (1, N)
			newShape := internal.Shape{1, arr.Size()}
			reshaped := arr.Reshape(newShape)
			reshapedArrays[i] = reshaped
		} else {
			reshapedArrays[i] = arr
		}
	}

	return Concatenate(reshapedArrays, 0)
}

// HStack stacks arrays horizontally (column-wise)
// Equivalent to concatenation along axis 1, except for 1-D arrays where it concatenates along axis 0
func HStack(arrays []*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("cannot stack empty list of arrays")
	}

	first := arrays[0]
	if first == nil {
		return nil, fmt.Errorf("first array cannot be nil")
	}

	// For 1D arrays, just concatenate along axis 0
	if first.Ndim() == 1 {
		return Concatenate(arrays, 0)
	}

	// For 2D+ arrays, concatenate along axis 1
	return Concatenate(arrays, 1)
}

// DStack stacks arrays depth-wise (along third axis)
// Takes a sequence of arrays and stacks them along the third axis
func DStack(arrays []*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("cannot stack empty list of arrays")
	}

	// Ensure all arrays are at least 2D
	reshapedArrays := make([]*Array, len(arrays))
	for i, arr := range arrays {
		if arr == nil {
			return nil, fmt.Errorf("array at index %d is nil", i)
		}

		switch arr.Ndim() {
		case 1:
			// Reshape 1D (N,) to 2D (1, N)
			newShape := internal.Shape{1, arr.Size()}
			reshaped := arr.Reshape(newShape)
			reshapedArrays[i] = reshaped
		case 2:
			// Add third dimension (M, N) -> (M, N, 1)
			shape := arr.Shape()
			newShape := internal.Shape{shape[0], shape[1], 1}
			reshaped := arr.Reshape(newShape)
			reshapedArrays[i] = reshaped
		default:
			reshapedArrays[i] = arr
		}
	}

	return Concatenate(reshapedArrays, 2)
}

// Split divides an array into multiple sub-arrays along the given axis
func Split(array *Array, numSections int, axis int) ([]*Array, error) {
	if array == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}

	if numSections <= 0 {
		return nil, fmt.Errorf("number of sections must be positive, got %d", numSections)
	}

	if axis < 0 || axis >= array.Ndim() {
		return nil, fmt.Errorf("axis %d out of bounds for array with %d dimensions", axis, array.Ndim())
	}

	axisSize := array.Shape()[axis]
	if axisSize%numSections != 0 {
		return nil, fmt.Errorf("array cannot be split into %d equal sections along axis %d (size %d)", numSections, axis, axisSize)
	}

	sectionSize := axisSize / numSections
	result := make([]*Array, numSections)

	for i := 0; i < numSections; i++ {
		start := i * sectionSize
		end := start + sectionSize

		// Create slice ranges for all dimensions
		ranges := make([]internal.Range, array.Ndim())
		for j := range ranges {
			if j == axis {
				ranges[j] = internal.Range{Start: start, Stop: end, Step: 1}
			} else {
				ranges[j] = internal.Range{Start: 0, Stop: array.Shape()[j], Step: 1}
			}
		}

		subArray, err := array.Slice(ranges...)
		if err != nil {
			return nil, fmt.Errorf("failed to slice array for section %d: %v", i, err)
		}

		result[i] = subArray
	}

	return result, nil
}

// HSplit splits an array horizontally (column-wise)
// For 1D arrays, equivalent to Split along axis 0
// For 2D+ arrays, equivalent to Split along axis 1
func HSplit(array *Array, numSections int) ([]*Array, error) {
	if array == nil {
		return nil, fmt.Errorf("array cannot be nil")
	}

	axis := 0
	if array.Ndim() > 1 {
		axis = 1
	}

	return Split(array, numSections, axis)
}

// VSplit splits an array vertically (row-wise)
// Equivalent to Split along axis 0
func VSplit(array *Array, numSections int) ([]*Array, error) {
	return Split(array, numSections, 0)
}

// DSplit splits an array along the third axis (depth-wise)
// Equivalent to Split along axis 2
func DSplit(array *Array, numSections int) ([]*Array, error) {
	return Split(array, numSections, 2)
}

// Helper function to copy array data during concatenation
func copyArrayData(dest *Array, src *Array, offset int, axis int) error {
	// For simple case of 1D arrays
	if src.Ndim() == 1 && dest.Ndim() == 1 {
		for i := 0; i < src.Size(); i++ {
			val := src.At(i)
			err := dest.Set(val, offset+i)
			if err != nil {
				return err
			}
		}
		return nil
	}

	// For multi-dimensional arrays, we need to copy slice by slice
	return copyMultiDimData(dest, src, offset, axis)
}

// Helper function to copy multi-dimensional array data
func copyMultiDimData(dest *Array, src *Array, offset int, axis int) error {
	srcShape := src.Shape()

	// Generate all index combinations for the source array
	return iterateIndices(srcShape, func(indices []int) error {
		// Calculate destination indices
		destIndices := make([]int, len(indices))
		copy(destIndices, indices)
		destIndices[axis] += offset

		// Copy the value
		val := src.At(indices...)
		return dest.Set(val, destIndices...)
	})
}

// Helper function to iterate over all indices in a multi-dimensional array
func iterateIndices(shape internal.Shape, fn func([]int) error) error {
	indices := make([]int, len(shape))
	return iterateIndicesRecursive(shape, indices, 0, fn)
}

// Recursive helper for iterateIndices
func iterateIndicesRecursive(shape internal.Shape, indices []int, dim int, fn func([]int) error) error {
	if dim == len(shape) {
		// Base case: process this index combination
		indexCopy := make([]int, len(indices))
		copy(indexCopy, indices)
		return fn(indexCopy)
	}

	// Recursive case: iterate over this dimension
	for i := 0; i < shape[dim]; i++ {
		indices[dim] = i
		if err := iterateIndicesRecursive(shape, indices, dim+1, fn); err != nil {
			return err
		}
	}

	return nil
}
