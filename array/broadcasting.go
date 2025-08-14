package array

import (
	"fmt"

	"github.com/julianshen/gonp/internal"
)

// BroadcastArrays broadcasts two arrays to compatible shapes for element-wise operations
// Follows NumPy broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html
func BroadcastArrays(a, b *Array) (*Array, *Array, error) {
	// Get broadcasted shape
	resultShape, err := BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, nil, err
	}

	// If arrays already have the correct shape, return them as-is
	if a.Shape().Equal(resultShape) && b.Shape().Equal(resultShape) {
		return a, b, nil
	}

	// Broadcast array a to result shape
	broadcastA, err := broadcastToShape(a, resultShape)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to broadcast first array: %v", err)
	}

	// Broadcast array b to result shape
	broadcastB, err := broadcastToShape(b, resultShape)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to broadcast second array: %v", err)
	}

	return broadcastA, broadcastB, nil
}

// BroadcastShapes determines the broadcasted shape of two shapes following NumPy rules
func BroadcastShapes(shape1, shape2 internal.Shape) (internal.Shape, error) {
	// Make copies to avoid modifying original shapes
	s1 := make(internal.Shape, len(shape1))
	s2 := make(internal.Shape, len(shape2))
	copy(s1, shape1)
	copy(s2, shape2)

	// Pad the shorter shape with 1s on the left
	maxLen := len(s1)
	if len(s2) > maxLen {
		maxLen = len(s2)
	}

	// Pad s1 if shorter
	for len(s1) < maxLen {
		s1 = append([]int{1}, s1...)
	}

	// Pad s2 if shorter
	for len(s2) < maxLen {
		s2 = append([]int{1}, s2...)
	}

	// Check compatibility and determine result shape
	result := make(internal.Shape, maxLen)
	for i := 0; i < maxLen; i++ {
		if s1[i] == s2[i] {
			result[i] = s1[i]
		} else if s1[i] == 1 {
			result[i] = s2[i]
		} else if s2[i] == 1 {
			result[i] = s1[i]
		} else {
			return nil, internal.NewShapeErrorWithMsg("BroadcastShapes",
				fmt.Sprintf("cannot broadcast shapes %v and %v: incompatible dimensions at position %d (%d vs %d)",
					shape1, shape2, i, s1[i], s2[i]))
		}
	}

	return result, nil
}

// broadcastToShape broadcasts an array to a target shape
func broadcastToShape(arr *Array, targetShape internal.Shape) (*Array, error) {
	if arr.Shape().Equal(targetShape) {
		return arr, nil
	}

	// Validate that broadcasting is possible
	arrShape := arr.Shape()
	_, err := BroadcastShapes(arrShape, targetShape)
	if err != nil {
		return nil, err
	}

	// Create new array with target shape
	// For now, we'll create a copy with expanded data
	// In a full implementation, this would use views and stride manipulation
	return expandArrayToShape(arr, targetShape)
}

// expandArrayToShape creates a new array by expanding the source array to the target shape
func expandArrayToShape(arr *Array, targetShape internal.Shape) (*Array, error) {
	// Create new array with target shape and same dtype
	newArr := Empty(targetShape, arr.DType())

	// Get the source and target shapes, padded to same length
	srcShape := arr.Shape()

	// Pad source shape with 1s on the left to match target length
	paddedSrcShape := make(internal.Shape, len(targetShape))
	offset := len(targetShape) - len(srcShape)
	for i := 0; i < offset; i++ {
		paddedSrcShape[i] = 1
	}
	for i := 0; i < len(srcShape); i++ {
		paddedSrcShape[offset+i] = srcShape[i]
	}

	// Fill the new array by replicating values according to broadcasting rules
	return fillBroadcastedArray(arr, newArr, paddedSrcShape, targetShape)
}

// fillBroadcastedArray fills the target array with broadcasted values from source
func fillBroadcastedArray(src, dst *Array, srcShape, dstShape internal.Shape) (*Array, error) {
	// Calculate total number of elements in destination
	dstSize := dst.Size()

	// Fill each position in the destination array
	for flatIdx := 0; flatIdx < dstSize; flatIdx++ {
		// Convert flat index to multi-dimensional indices for destination
		dstIndices := dst.unflattenIndex(flatIdx)

		// Map destination indices to source indices according to broadcasting rules
		srcIndices := make([]int, len(srcShape))
		for i := 0; i < len(dstShape); i++ {
			if srcShape[i] == 1 {
				srcIndices[i] = 0 // Broadcast dimension
			} else {
				srcIndices[i] = dstIndices[i]
			}
		}

		// Get value from source (handling dimension padding)
		var val interface{}
		if len(src.Shape()) == len(srcIndices) {
			val = src.At(srcIndices...)
		} else {
			// Remove leading padding indices
			actualIndices := srcIndices[len(srcIndices)-len(src.Shape()):]
			val = src.At(actualIndices...)
		}

		// Set value in destination
		err := dst.Set(val, dstIndices...)
		if err != nil {
			return nil, fmt.Errorf("failed to set value during broadcasting: %v", err)
		}
	}

	return dst, nil
}

// CanBroadcast checks if two shapes can be broadcasted together
func CanBroadcast(shape1, shape2 internal.Shape) bool {
	_, err := BroadcastShapes(shape1, shape2)
	return err == nil
}

// GetBroadcastShape returns the result shape of broadcasting two shapes, or error if incompatible
func GetBroadcastShape(shape1, shape2 internal.Shape) (internal.Shape, error) {
	return BroadcastShapes(shape1, shape2)
}
