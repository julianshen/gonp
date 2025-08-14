package array

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/internal"
)

// ElementwiseOp represents an element-wise operation function
type ElementwiseOp func(a, b interface{}) interface{}

// Add performs element-wise addition with broadcasting
func (a *Array) Add(b *Array) (*Array, error) {
	ctx := internal.StartProfiler("Array.Add")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()
	internal.IncrementOperations()

	if err := internal.QuickValidateNotNil(a, "Add", "first array"); err != nil {
		return nil, err
	}
	if err := internal.QuickValidateNotNil(b, "Add", "second array"); err != nil {
		return nil, err
	}
	return a.elementwiseOperation(b, addOp)
}

// Sub performs element-wise subtraction with broadcasting
func (a *Array) Sub(b *Array) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "Sub", "first array"); err != nil {
		return nil, err
	}
	if err := internal.QuickValidateNotNil(b, "Sub", "second array"); err != nil {
		return nil, err
	}
	return a.elementwiseOperation(b, subOp)
}

// Mul performs element-wise multiplication with broadcasting
func (a *Array) Mul(b *Array) (*Array, error) {
	ctx := internal.StartProfiler("Array.Mul")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()
	internal.IncrementOperations()

	if err := internal.QuickValidateNotNil(a, "Mul", "first array"); err != nil {
		return nil, err
	}
	if err := internal.QuickValidateNotNil(b, "Mul", "second array"); err != nil {
		return nil, err
	}
	return a.elementwiseOperation(b, mulOp)
}

// Div performs element-wise division with broadcasting
func (a *Array) Div(b *Array) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "Div", "first array"); err != nil {
		return nil, err
	}
	if err := internal.QuickValidateNotNil(b, "Div", "second array"); err != nil {
		return nil, err
	}
	return a.elementwiseOperation(b, divOp)
}

// Pow performs element-wise power operation with broadcasting
func (a *Array) Pow(b *Array) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "Pow", "first array"); err != nil {
		return nil, err
	}
	if err := internal.QuickValidateNotNil(b, "Pow", "second array"); err != nil {
		return nil, err
	}
	return a.elementwiseOperation(b, powOp)
}

// Mod performs element-wise modulo operation with broadcasting
func (a *Array) Mod(b *Array) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "Mod", "first array"); err != nil {
		return nil, err
	}
	if err := internal.QuickValidateNotNil(b, "Mod", "second array"); err != nil {
		return nil, err
	}
	return a.elementwiseOperation(b, modOp)
}

// AddScalar adds a scalar value to all elements
func (a *Array) AddScalar(scalar interface{}) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "AddScalar", "array"); err != nil {
		return nil, err
	}
	return a.scalarOperation(scalar, addOp)
}

// SubScalar subtracts a scalar value from all elements
func (a *Array) SubScalar(scalar interface{}) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "SubScalar", "array"); err != nil {
		return nil, err
	}
	return a.scalarOperation(scalar, subOp)
}

// MulScalar multiplies all elements by a scalar value
func (a *Array) MulScalar(scalar interface{}) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "MulScalar", "array"); err != nil {
		return nil, err
	}
	return a.scalarOperation(scalar, mulOp)
}

// DivScalar divides all elements by a scalar value
func (a *Array) DivScalar(scalar interface{}) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "DivScalar", "array"); err != nil {
		return nil, err
	}
	return a.scalarOperation(scalar, divOp)
}

// PowScalar raises all elements to a scalar power
func (a *Array) PowScalar(scalar interface{}) (*Array, error) {
	if err := internal.QuickValidateNotNil(a, "PowScalar", "array"); err != nil {
		return nil, err
	}
	return a.scalarOperation(scalar, powOp)
}

// elementwiseOperation performs a general element-wise operation with broadcasting
func (a *Array) elementwiseOperation(b *Array, op ElementwiseOp) (*Array, error) {
	// Broadcast arrays to compatible shapes
	broadcastA, broadcastB, err := BroadcastArrays(a, b)
	if err != nil {
		return nil, fmt.Errorf("broadcasting failed: %v", err)
	}

	// Create result array with the broadcasted shape
	resultShape := broadcastA.Shape()
	result := Empty(resultShape, a.DType()) // Use first array's dtype as default

	// Apply operation element-wise
	size := result.Size()
	for i := 0; i < size; i++ {
		indices := result.unflattenIndex(i)

		valA := broadcastA.At(indices...)
		valB := broadcastB.At(indices...)

		resultVal := op(valA, valB)

		err := result.Set(resultVal, indices...)
		if err != nil {
			return nil, fmt.Errorf("failed to set result value: %v", err)
		}
	}

	return result, nil
}

// scalarOperation performs an operation between array and scalar
func (a *Array) scalarOperation(scalar interface{}, op ElementwiseOp) (*Array, error) {
	// Create result array with same shape and dtype
	result := Empty(a.Shape(), a.DType())

	// Apply operation element-wise
	size := result.Size()
	for i := 0; i < size; i++ {
		indices := result.unflattenIndex(i)

		val := a.At(indices...)
		resultVal := op(val, scalar)

		err := result.Set(resultVal, indices...)
		if err != nil {
			return nil, fmt.Errorf("failed to set result value: %v", err)
		}
	}

	return result, nil
}

// Operation functions for different arithmetic operations
func addOp(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va + convertToFloat64(b)
	case float32:
		return va + float32(convertToFloat64(b))
	case int64:
		return va + convertToInt64(b)
	case int32:
		return va + int32(convertToInt64(b))
	case int16:
		return va + int16(convertToInt64(b))
	case int8:
		return va + int8(convertToInt64(b))
	case uint64:
		return va + uint64(convertToInt64(b))
	case uint32:
		return va + uint32(convertToInt64(b))
	case uint16:
		return va + uint16(convertToInt64(b))
	case uint8:
		return va + uint8(convertToInt64(b))
	case complex64:
		return va + convertToComplex64(b)
	case complex128:
		return va + convertToComplex128(b)
	default:
		panic(fmt.Sprintf("unsupported type for addition: %T", a))
	}
}

func subOp(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va - convertToFloat64(b)
	case float32:
		return va - float32(convertToFloat64(b))
	case int64:
		return va - convertToInt64(b)
	case int32:
		return va - int32(convertToInt64(b))
	case int16:
		return va - int16(convertToInt64(b))
	case int8:
		return va - int8(convertToInt64(b))
	case uint64:
		return va - uint64(convertToInt64(b))
	case uint32:
		return va - uint32(convertToInt64(b))
	case uint16:
		return va - uint16(convertToInt64(b))
	case uint8:
		return va - uint8(convertToInt64(b))
	case complex64:
		return va - convertToComplex64(b)
	case complex128:
		return va - convertToComplex128(b)
	default:
		panic(fmt.Sprintf("unsupported type for subtraction: %T", a))
	}
}

func mulOp(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return va * convertToFloat64(b)
	case float32:
		return va * float32(convertToFloat64(b))
	case int64:
		return va * convertToInt64(b)
	case int32:
		return va * int32(convertToInt64(b))
	case int16:
		return va * int16(convertToInt64(b))
	case int8:
		return va * int8(convertToInt64(b))
	case uint64:
		return va * uint64(convertToInt64(b))
	case uint32:
		return va * uint32(convertToInt64(b))
	case uint16:
		return va * uint16(convertToInt64(b))
	case uint8:
		return va * uint8(convertToInt64(b))
	case complex64:
		return va * convertToComplex64(b)
	case complex128:
		return va * convertToComplex128(b)
	default:
		panic(fmt.Sprintf("unsupported type for multiplication: %T", a))
	}
}

func divOp(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		vb := convertToFloat64(b)
		if vb == 0 {
			return math.Inf(1) // Handle division by zero
		}
		return va / vb
	case float32:
		vb := float32(convertToFloat64(b))
		if vb == 0 {
			return float32(math.Inf(1))
		}
		return va / vb
	case int64:
		vb := convertToInt64(b)
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case int32:
		vb := int32(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case int16:
		vb := int16(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case int8:
		vb := int8(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case uint64:
		vb := uint64(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case uint32:
		vb := uint32(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case uint16:
		vb := uint16(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case uint8:
		vb := uint8(convertToInt64(b))
		if vb == 0 {
			panic("integer division by zero")
		}
		return va / vb
	case complex64:
		vb := convertToComplex64(b)
		if vb == 0 {
			return complex64(complex(float32(math.Inf(1)), float32(math.Inf(1))))
		}
		return va / vb
	case complex128:
		vb := convertToComplex128(b)
		if vb == 0 {
			return complex(math.Inf(1), math.Inf(1))
		}
		return va / vb
	default:
		panic(fmt.Sprintf("unsupported type for division: %T", a))
	}
}

func powOp(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return math.Pow(va, convertToFloat64(b))
	case float32:
		return float32(math.Pow(float64(va), convertToFloat64(b)))
	case int64:
		return int64(math.Pow(float64(va), convertToFloat64(b)))
	case int32:
		return int32(math.Pow(float64(va), convertToFloat64(b)))
	case int16:
		return int16(math.Pow(float64(va), convertToFloat64(b)))
	case int8:
		return int8(math.Pow(float64(va), convertToFloat64(b)))
	case uint64:
		return uint64(math.Pow(float64(va), convertToFloat64(b)))
	case uint32:
		return uint32(math.Pow(float64(va), convertToFloat64(b)))
	case uint16:
		return uint16(math.Pow(float64(va), convertToFloat64(b)))
	case uint8:
		return uint8(math.Pow(float64(va), convertToFloat64(b)))
	case complex64:
		return complex64(complexPow(complex128(va), complex128(convertToComplex64(b))))
	case complex128:
		return complexPow(va, convertToComplex128(b))
	default:
		panic(fmt.Sprintf("unsupported type for power: %T", a))
	}
}

func modOp(a, b interface{}) interface{} {
	switch va := a.(type) {
	case float64:
		return math.Mod(va, convertToFloat64(b))
	case float32:
		return float32(math.Mod(float64(va), convertToFloat64(b)))
	case int64:
		vb := convertToInt64(b)
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case int32:
		vb := int32(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case int16:
		vb := int16(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case int8:
		vb := int8(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case uint64:
		vb := uint64(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case uint32:
		vb := uint32(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case uint16:
		vb := uint16(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	case uint8:
		vb := uint8(convertToInt64(b))
		if vb == 0 {
			panic("modulo by zero")
		}
		return va % vb
	default:
		panic(fmt.Sprintf("unsupported type for modulo: %T", a))
	}
}

// complexPow computes complex power using exp(b * log(a))
func complexPow(a, b complex128) complex128 {
	if a == 0 {
		if real(b) == 0 && imag(b) == 0 {
			return 1
		}
		return 0
	}
	// Use math.Pow on complex numbers - simplified implementation
	// For full implementation, would use complex logarithm and exponential
	r := math.Sqrt(real(a)*real(a) + imag(a)*imag(a))
	theta := math.Atan2(imag(a), real(a))

	newR := math.Pow(r, real(b)) * math.Exp(-imag(b)*theta)
	newTheta := real(b)*theta + imag(b)*math.Log(r)

	return complex(newR*math.Cos(newTheta), newR*math.Sin(newTheta))
}
