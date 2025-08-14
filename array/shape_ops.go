package array

import (
	"fmt"

	"github.com/julianshen/gonp/internal"
)

// Squeeze removes single-dimensional entries from the shape of an array
func (a *Array) Squeeze(axis ...int) (*Array, error) {
	shape := a.Shape()

	// If no axis specified, remove all dimensions of size 1
	if len(axis) == 0 {
		newShape := make(internal.Shape, 0)
		for _, dim := range shape {
			if dim != 1 {
				newShape = append(newShape, dim)
			}
		}

		// Handle case where all dimensions are 1
		if len(newShape) == 0 {
			newShape = internal.Shape{1}
		}

		return a.Reshape(newShape), nil
	}

	// Validate specified axes
	for _, ax := range axis {
		if ax < 0 || ax >= len(shape) {
			return nil, fmt.Errorf("axis %d is out of bounds for array of dimension %d", ax, len(shape))
		}
		if shape[ax] != 1 {
			return nil, fmt.Errorf("cannot squeeze axis %d which has size %d", ax, shape[ax])
		}
	}

	// Create new shape excluding specified axes
	newShape := make(internal.Shape, 0)
	for i, dim := range shape {
		shouldSqueeze := false
		for _, ax := range axis {
			if i == ax {
				shouldSqueeze = true
				break
			}
		}
		if !shouldSqueeze {
			newShape = append(newShape, dim)
		}
	}

	// Handle case where all dimensions are squeezed
	if len(newShape) == 0 {
		newShape = internal.Shape{1}
	}

	return a.Reshape(newShape), nil
}

// ExpandDims expands the shape of an array by inserting new axes of length 1
func (a *Array) ExpandDims(axis ...int) (*Array, error) {
	shape := a.Shape()

	// Calculate final dimensions count after expansion
	finalNdim := len(shape) + len(axis)

	// Normalize and validate all axes
	normalizedAxes := make([]int, len(axis))
	for i, ax := range axis {
		if ax < 0 {
			ax = finalNdim + ax
		}
		if ax < 0 || ax > finalNdim {
			return nil, fmt.Errorf("axis %d is out of bounds for array after expansion", axis[i])
		}
		normalizedAxes[i] = ax
	}

	// Check for duplicate axes
	for i := 0; i < len(normalizedAxes); i++ {
		for j := i + 1; j < len(normalizedAxes); j++ {
			if normalizedAxes[i] == normalizedAxes[j] {
				return nil, fmt.Errorf("repeated axis %d", normalizedAxes[i])
			}
		}
	}

	// Create the new shape by inserting 1s at specified positions
	newShape := make(internal.Shape, finalNdim)

	// Mark positions where new axes should be inserted
	isNewAxis := make([]bool, finalNdim)
	for _, ax := range normalizedAxes {
		isNewAxis[ax] = true
	}

	// Fill the new shape
	oldIdx := 0
	for i := 0; i < finalNdim; i++ {
		if isNewAxis[i] {
			newShape[i] = 1 // New axis has size 1
		} else {
			newShape[i] = shape[oldIdx] // Copy from original shape
			oldIdx++
		}
	}

	return a.Reshape(newShape), nil
}

// Swapaxes interchanges two axes of an array
func (a *Array) Swapaxes(axis1, axis2 int) (*Array, error) {
	shape := a.Shape()
	ndim := len(shape)

	// Handle negative indices
	if axis1 < 0 {
		axis1 = ndim + axis1
	}
	if axis2 < 0 {
		axis2 = ndim + axis2
	}

	// Validate axes
	if axis1 < 0 || axis1 >= ndim {
		return nil, fmt.Errorf("axis1 %d is out of bounds for array of dimension %d", axis1, ndim)
	}
	if axis2 < 0 || axis2 >= ndim {
		return nil, fmt.Errorf("axis2 %d is out of bounds for array of dimension %d", axis2, ndim)
	}

	// If axes are the same, return copy
	if axis1 == axis2 {
		return a.Copy(), nil
	}

	// Create permutation for transpose
	axes := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		axes[i] = i
	}
	axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

	return a.TransposeWithAxes(axes...)
}

// TransposeWithAxes transposes the array according to the specified axes permutation
func (a *Array) TransposeWithAxes(axes ...int) (*Array, error) {
	shape := a.Shape()
	ndim := len(shape)

	// If no axes specified, reverse all axes (default transpose)
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := 0; i < ndim; i++ {
			axes[i] = ndim - 1 - i
		}
	}

	// Validate axes
	if len(axes) != ndim {
		return nil, fmt.Errorf("axes must have same length as array dimension: got %d, expected %d", len(axes), ndim)
	}

	// Check that axes is a valid permutation
	used := make([]bool, ndim)
	for _, ax := range axes {
		if ax < 0 {
			ax = ndim + ax
		}
		if ax < 0 || ax >= ndim {
			return nil, fmt.Errorf("axis %d is out of bounds for array of dimension %d", ax, ndim)
		}
		if used[ax] {
			return nil, fmt.Errorf("axis %d is repeated in axes", ax)
		}
		used[ax] = true
	}

	// Create new shape according to axes permutation
	newShape := make(internal.Shape, ndim)
	for i, ax := range axes {
		if ax < 0 {
			ax = ndim + ax
		}
		newShape[i] = shape[ax]
	}

	// Create new stride according to axes permutation
	newStride := make(internal.Stride, ndim)
	oldStride := a.stride
	for i, ax := range axes {
		if ax < 0 {
			ax = ndim + ax
		}
		newStride[i] = oldStride[ax]
	}

	// Return new array with transposed shape and stride
	return &Array{
		storage: a.storage,
		shape:   newShape,
		stride:  newStride,
		dtype:   a.dtype,
		offset:  a.offset,
	}, nil
}

// Moveaxis moves axes of an array to new positions
func (a *Array) Moveaxis(source, destination []int) (*Array, error) {
	ndim := len(a.Shape())

	if len(source) != len(destination) {
		return nil, fmt.Errorf("source and destination must have same length")
	}

	// Normalize negative indices
	for i := range source {
		if source[i] < 0 {
			source[i] = ndim + source[i]
		}
		if destination[i] < 0 {
			destination[i] = ndim + destination[i]
		}
	}

	// Validate indices
	for i := range source {
		if source[i] < 0 || source[i] >= ndim {
			return nil, fmt.Errorf("source axis %d is out of bounds for array of dimension %d", source[i], ndim)
		}
		if destination[i] < 0 || destination[i] >= ndim {
			return nil, fmt.Errorf("destination axis %d is out of bounds for array of dimension %d", destination[i], ndim)
		}
	}

	// Create axes permutation
	axes := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		axes[i] = i
	}

	// Remove source axes
	temp := make([]int, 0, ndim-len(source))
	for i := 0; i < ndim; i++ {
		isSource := false
		for _, src := range source {
			if i == src {
				isSource = true
				break
			}
		}
		if !isSource {
			temp = append(temp, i)
		}
	}

	// Insert source axes at destination positions
	result := make([]int, ndim)
	tempIdx := 0

	for i := 0; i < ndim; i++ {
		isDest := false
		var srcAxis int
		for j, dest := range destination {
			if i == dest {
				isDest = true
				srcAxis = source[j]
				break
			}
		}

		if isDest {
			result[i] = srcAxis
		} else {
			result[i] = temp[tempIdx]
			tempIdx++
		}
	}

	return a.TransposeWithAxes(result...)
}

// Rollaxis rolls the specified axis backwards until it lies in a given position
func (a *Array) Rollaxis(axis, start int) (*Array, error) {
	ndim := len(a.Shape())

	// Normalize negative indices
	if axis < 0 {
		axis = ndim + axis
	}
	if start < 0 {
		start = ndim + start
	}

	// Validate indices
	if axis < 0 || axis >= ndim {
		return nil, fmt.Errorf("axis %d is out of bounds for array of dimension %d", axis, ndim)
	}
	if start < 0 || start > ndim {
		return nil, fmt.Errorf("start %d is out of bounds for rollaxis", start)
	}

	// If axis is already at the desired position, return copy
	if axis == start {
		return a.Copy(), nil
	}

	// Create new axes order
	axes := make([]int, ndim)

	if axis < start {
		// Roll forward
		for i := 0; i < axis; i++ {
			axes[i] = i
		}
		for i := axis; i < start-1; i++ {
			axes[i] = i + 1
		}
		axes[start-1] = axis
		for i := start; i < ndim; i++ {
			axes[i] = i
		}
	} else {
		// Roll backward
		for i := 0; i < start; i++ {
			axes[i] = i
		}
		axes[start] = axis
		for i := start + 1; i <= axis; i++ {
			axes[i] = i - 1
		}
		for i := axis + 1; i < ndim; i++ {
			axes[i] = i
		}
	}

	return a.TransposeWithAxes(axes...)
}
