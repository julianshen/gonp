package series

import (
	"fmt"
)

// ILocSingle returns the value at a single integer position
func (s *Series) ILocSingle(position int) interface{} {
	if position < 0 || position >= s.Len() {
		panic(fmt.Sprintf("position %d out of bounds for Series of length %d", position, s.Len()))
	}
	return s.data.At(position)
}

// ILocSlice returns a new Series with elements from start to end (exclusive)
func (s *Series) ILocSlice(start, end int) (*Series, error) {
	if start < 0 {
		start = 0
	}
	if end > s.Len() {
		end = s.Len()
	}
	if start >= end {
		return Empty(s.DType(), s.name), nil
	}

	positions := make([]int, end-start)
	for i := range positions {
		positions[i] = start + i
	}

	return s.ILoc(positions)
}

// Note: Copy method already exists in series.go
