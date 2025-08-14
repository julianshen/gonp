package array

import (
	"fmt"
)

// AddInPlace performs element-wise addition in-place with broadcasting
func (a *Array) AddInPlace(b *Array) error {
	return a.inplaceOperation(b, addOp)
}

// SubInPlace performs element-wise subtraction in-place with broadcasting
func (a *Array) SubInPlace(b *Array) error {
	return a.inplaceOperation(b, subOp)
}

// MulInPlace performs element-wise multiplication in-place with broadcasting
func (a *Array) MulInPlace(b *Array) error {
	return a.inplaceOperation(b, mulOp)
}

// DivInPlace performs element-wise division in-place with broadcasting
func (a *Array) DivInPlace(b *Array) error {
	return a.inplaceOperation(b, divOp)
}

// PowInPlace performs element-wise power operation in-place with broadcasting
func (a *Array) PowInPlace(b *Array) error {
	return a.inplaceOperation(b, powOp)
}

// ModInPlace performs element-wise modulo operation in-place with broadcasting
func (a *Array) ModInPlace(b *Array) error {
	return a.inplaceOperation(b, modOp)
}

// AddScalarInPlace adds a scalar value to all elements in-place
func (a *Array) AddScalarInPlace(scalar interface{}) error {
	return a.inplaceScalarOperation(scalar, addOp)
}

// SubScalarInPlace subtracts a scalar value from all elements in-place
func (a *Array) SubScalarInPlace(scalar interface{}) error {
	return a.inplaceScalarOperation(scalar, subOp)
}

// MulScalarInPlace multiplies all elements by a scalar value in-place
func (a *Array) MulScalarInPlace(scalar interface{}) error {
	return a.inplaceScalarOperation(scalar, mulOp)
}

// DivScalarInPlace divides all elements by a scalar value in-place
func (a *Array) DivScalarInPlace(scalar interface{}) error {
	return a.inplaceScalarOperation(scalar, divOp)
}

// PowScalarInPlace raises all elements to a scalar power in-place
func (a *Array) PowScalarInPlace(scalar interface{}) error {
	return a.inplaceScalarOperation(scalar, powOp)
}

// inplaceOperation performs a general element-wise operation in-place with broadcasting
func (a *Array) inplaceOperation(b *Array, op ElementwiseOp) error {
	// Check if broadcasting is possible
	_, err := BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return fmt.Errorf("broadcasting failed: %v", err)
	}

	// For in-place operations, the first array must be broadcastable to itself
	// and the second array must be broadcastable to the first array's shape
	if !CanBroadcast(a.Shape(), b.Shape()) {
		return fmt.Errorf("cannot broadcast array with shape %v to shape %v", b.Shape(), a.Shape())
	}

	// Apply operation element-wise directly to the first array
	size := a.Size()
	for i := 0; i < size; i++ {
		indices := a.unflattenIndex(i)

		valA := a.At(indices...)

		// Get corresponding value from b, handling broadcasting
		valB := a.getBroadcastedValue(b, indices)

		resultVal := op(valA, valB)

		err := a.Set(resultVal, indices...)
		if err != nil {
			return fmt.Errorf("failed to set result value: %v", err)
		}
	}

	return nil
}

// inplaceScalarOperation performs an operation between array and scalar in-place
func (a *Array) inplaceScalarOperation(scalar interface{}, op ElementwiseOp) error {
	// Apply operation element-wise directly to the array
	size := a.Size()
	for i := 0; i < size; i++ {
		indices := a.unflattenIndex(i)

		val := a.At(indices...)
		resultVal := op(val, scalar)

		err := a.Set(resultVal, indices...)
		if err != nil {
			return fmt.Errorf("failed to set result value: %v", err)
		}
	}

	return nil
}

// getBroadcastedValue gets the corresponding value from array b for given indices in array a
// This handles broadcasting by mapping indices according to broadcasting rules
func (a *Array) getBroadcastedValue(b *Array, indices []int) interface{} {
	aShape := a.Shape()
	bShape := b.Shape()

	// If shapes are identical, use indices directly
	if aShape.Equal(bShape) {
		return b.At(indices...)
	}

	// Map indices from a's shape to b's shape according to broadcasting rules
	// For example: a(2,3) + b(3,) -> b index should map from a's (i,j) to b's (j)

	// Calculate how many trailing dimensions match
	aLen := len(aShape)
	bLen := len(bShape)

	// Build indices for b by taking the appropriate trailing indices from a
	bIndices := make([]int, bLen)

	// Map from right to left (trailing dimensions)
	for i := 0; i < bLen; i++ {
		aIdx := aLen - bLen + i
		if aIdx >= 0 && aIdx < len(indices) {
			// Check if this dimension should broadcast (size 1 in b)
			if bShape[i] == 1 {
				bIndices[i] = 0 // Broadcast dimension
			} else {
				bIndices[i] = indices[aIdx]
			}
		} else {
			bIndices[i] = 0
		}
	}

	return b.At(bIndices...)
}
