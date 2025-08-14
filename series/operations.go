package series

import (
	"fmt"
	"math"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// Arithmetic operations with data alignment

// Add performs element-wise addition with another Series, aligning on index
func (s *Series) Add(other *Series) (*Series, error) {
	return s.binaryOperation(other, addOperation, "add")
}

// Sub performs element-wise subtraction with another Series, aligning on index
func (s *Series) Sub(other *Series) (*Series, error) {
	return s.binaryOperation(other, subOperation, "subtract")
}

// Mul performs element-wise multiplication with another Series, aligning on index
func (s *Series) Mul(other *Series) (*Series, error) {
	return s.binaryOperation(other, mulOperation, "multiply")
}

// Div performs element-wise division with another Series, aligning on index
func (s *Series) Div(other *Series) (*Series, error) {
	return s.binaryOperation(other, divOperation, "divide")
}

// Pow performs element-wise exponentiation with another Series, aligning on index
func (s *Series) Pow(other *Series) (*Series, error) {
	return s.binaryOperation(other, powOperation, "power")
}

// Mod performs element-wise modulo with another Series, aligning on index
func (s *Series) Mod(other *Series) (*Series, error) {
	return s.binaryOperation(other, modOperation, "modulo")
}

// Scalar operations

// AddScalar adds a scalar value to all elements
func (s *Series) AddScalar(scalar interface{}) (*Series, error) {
	return s.scalarOperation(scalar, addOperation, "add")
}

// SubScalar subtracts a scalar value from all elements
func (s *Series) SubScalar(scalar interface{}) (*Series, error) {
	return s.scalarOperation(scalar, subOperation, "subtract")
}

// MulScalar multiplies all elements by a scalar value
func (s *Series) MulScalar(scalar interface{}) (*Series, error) {
	return s.scalarOperation(scalar, mulOperation, "multiply")
}

// DivScalar divides all elements by a scalar value
func (s *Series) DivScalar(scalar interface{}) (*Series, error) {
	return s.scalarOperation(scalar, divOperation, "divide")
}

// PowScalar raises all elements to the power of a scalar value
func (s *Series) PowScalar(scalar interface{}) (*Series, error) {
	return s.scalarOperation(scalar, powOperation, "power")
}

// Comparison operations

// Eq performs element-wise equality comparison
func (s *Series) Eq(other *Series) (*Series, error) {
	return s.comparisonOperation(other, eqOperation, "equal")
}

// Ne performs element-wise not-equal comparison
func (s *Series) Ne(other *Series) (*Series, error) {
	return s.comparisonOperation(other, neOperation, "not_equal")
}

// Lt performs element-wise less-than comparison
func (s *Series) Lt(other *Series) (*Series, error) {
	return s.comparisonOperation(other, ltOperation, "less_than")
}

// Le performs element-wise less-than-or-equal comparison
func (s *Series) Le(other *Series) (*Series, error) {
	return s.comparisonOperation(other, leOperation, "less_equal")
}

// Gt performs element-wise greater-than comparison
func (s *Series) Gt(other *Series) (*Series, error) {
	return s.comparisonOperation(other, gtOperation, "greater_than")
}

// Ge performs element-wise greater-than-or-equal comparison
func (s *Series) Ge(other *Series) (*Series, error) {
	return s.comparisonOperation(other, geOperation, "greater_equal")
}

// Scalar comparison operations

// EqScalar performs element-wise equality comparison with a scalar
func (s *Series) EqScalar(scalar interface{}) (*Series, error) {
	return s.scalarComparisonOperation(scalar, eqOperation, "equal")
}

// NeScalar performs element-wise not-equal comparison with a scalar
func (s *Series) NeScalar(scalar interface{}) (*Series, error) {
	return s.scalarComparisonOperation(scalar, neOperation, "not_equal")
}

// LtScalar performs element-wise less-than comparison with a scalar
func (s *Series) LtScalar(scalar interface{}) (*Series, error) {
	return s.scalarComparisonOperation(scalar, ltOperation, "less_than")
}

// LeScalar performs element-wise less-than-or-equal comparison with a scalar
func (s *Series) LeScalar(scalar interface{}) (*Series, error) {
	return s.scalarComparisonOperation(scalar, leOperation, "less_equal")
}

// GtScalar performs element-wise greater-than comparison with a scalar
func (s *Series) GtScalar(scalar interface{}) (*Series, error) {
	return s.scalarComparisonOperation(scalar, gtOperation, "greater_than")
}

// GeScalar performs element-wise greater-than-or-equal comparison with a scalar
func (s *Series) GeScalar(scalar interface{}) (*Series, error) {
	return s.scalarComparisonOperation(scalar, geOperation, "greater_equal")
}

// Unary operations

// Neg returns a new Series with negated values
func (s *Series) Neg() (*Series, error) {
	result := array.Empty(s.data.Shape(), s.data.DType())

	for i := 0; i < s.Len(); i++ {
		val := s.data.At(i)
		negVal, err := negateValue(val)
		if err != nil {
			return nil, fmt.Errorf("failed to negate value at index %d: %v", i, err)
		}

		err = result.Set(negVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set negated value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// Abs returns a new Series with absolute values
func (s *Series) Abs() (*Series, error) {
	result := array.Empty(s.data.Shape(), s.data.DType())

	for i := 0; i < s.Len(); i++ {
		val := s.data.At(i)
		absVal, err := absoluteValue(val)
		if err != nil {
			return nil, fmt.Errorf("failed to get absolute value at index %d: %v", i, err)
		}

		err = result.Set(absVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set absolute value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// Internal operation types
type operationType int

const (
	addOperation operationType = iota
	subOperation
	mulOperation
	divOperation
	powOperation
	modOperation
	eqOperation
	neOperation
	ltOperation
	leOperation
	gtOperation
	geOperation
)

// binaryOperation performs a binary operation between two Series with index alignment
func (s *Series) binaryOperation(other *Series, op operationType, opName string) (*Series, error) {
	if other == nil {
		return nil, fmt.Errorf("cannot %s with nil Series", opName)
	}

	// Find the union of indices
	leftIndices := s.index.Values()
	rightIndices := other.index.Values()

	// Create a map for quick lookup
	leftMap := make(map[interface{}]int)
	for i, idx := range leftIndices {
		leftMap[idx] = i
	}

	rightMap := make(map[interface{}]int)
	for i, idx := range rightIndices {
		rightMap[idx] = i
	}

	// Find all unique indices (union)
	unionIndices := make([]interface{}, 0)
	indexSet := make(map[interface{}]bool)

	for _, idx := range leftIndices {
		if !indexSet[idx] {
			unionIndices = append(unionIndices, idx)
			indexSet[idx] = true
		}
	}

	for _, idx := range rightIndices {
		if !indexSet[idx] {
			unionIndices = append(unionIndices, idx)
			indexSet[idx] = true
		}
	}

	// Determine result dtype (promote to higher precision if needed)
	resultDType := promoteDataTypes(s.DType(), other.DType())

	// Create result arrays
	resultData := array.Empty(internal.Shape{len(unionIndices)}, resultDType)
	resultIndex := NewIndex(unionIndices)

	// Perform the operation
	for i, idx := range unionIndices {
		var leftVal, rightVal interface{}
		var leftExists, rightExists bool

		if leftPos, exists := leftMap[idx]; exists {
			leftVal = s.data.At(leftPos)
			leftExists = true
		} else {
			leftVal = getNaNValue(resultDType)
			leftExists = false
		}

		if rightPos, exists := rightMap[idx]; exists {
			rightVal = other.data.At(rightPos)
			rightExists = true
		} else {
			rightVal = getNaNValue(resultDType)
			rightExists = false
		}

		// If both values exist, perform operation; otherwise result is NaN
		var result interface{}
		if leftExists && rightExists {
			var err error
			result, err = performOperation(leftVal, rightVal, op)
			if err != nil {
				return nil, fmt.Errorf("failed to %s values at index %v: %v", opName, idx, err)
			}
		} else {
			result = getNaNValue(resultDType)
		}

		err := resultData.Set(result, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set result at index %d: %v", i, err)
		}
	}

	return NewSeries(resultData, resultIndex, s.name)
}

// scalarOperation performs a scalar operation on all elements
func (s *Series) scalarOperation(scalar interface{}, op operationType, opName string) (*Series, error) {
	result := array.Empty(s.data.Shape(), s.data.DType())

	for i := 0; i < s.Len(); i++ {
		val := s.data.At(i)
		resultVal, err := performOperation(val, scalar, op)
		if err != nil {
			return nil, fmt.Errorf("failed to %s value at index %d with scalar: %v", opName, i, err)
		}

		err = result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// comparisonOperation performs a comparison operation between two Series
func (s *Series) comparisonOperation(other *Series, op operationType, opName string) (*Series, error) {
	if other == nil {
		return nil, fmt.Errorf("cannot compare with nil Series")
	}

	// Use similar alignment logic as binary operations but return boolean Series
	leftIndices := s.index.Values()
	rightIndices := other.index.Values()

	leftMap := make(map[interface{}]int)
	for i, idx := range leftIndices {
		leftMap[idx] = i
	}

	rightMap := make(map[interface{}]int)
	for i, idx := range rightIndices {
		rightMap[idx] = i
	}

	// Find union of indices
	unionIndices := make([]interface{}, 0)
	indexSet := make(map[interface{}]bool)

	for _, idx := range leftIndices {
		if !indexSet[idx] {
			unionIndices = append(unionIndices, idx)
			indexSet[idx] = true
		}
	}

	for _, idx := range rightIndices {
		if !indexSet[idx] {
			unionIndices = append(unionIndices, idx)
			indexSet[idx] = true
		}
	}

	// Create boolean result
	resultData := array.Empty(internal.Shape{len(unionIndices)}, internal.Bool)
	resultIndex := NewIndex(unionIndices)

	for i, idx := range unionIndices {
		var leftVal, rightVal interface{}
		var leftExists, rightExists bool

		if leftPos, exists := leftMap[idx]; exists {
			leftVal = s.data.At(leftPos)
			leftExists = true
		}

		if rightPos, exists := rightMap[idx]; exists {
			rightVal = other.data.At(rightPos)
			rightExists = true
		}

		// Perform comparison
		var result bool
		if leftExists && rightExists {
			var err error
			compResult, err := performOperation(leftVal, rightVal, op)
			if err != nil {
				return nil, fmt.Errorf("failed to compare values at index %v: %v", idx, err)
			}
			result = compResult.(bool)
		} else {
			result = false // Missing values compare as false
		}

		err := resultData.Set(result, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set comparison result at index %d: %v", i, err)
		}
	}

	return NewSeries(resultData, resultIndex, s.name)
}

// scalarComparisonOperation performs a scalar comparison operation
func (s *Series) scalarComparisonOperation(scalar interface{}, op operationType, opName string) (*Series, error) {
	result := array.Empty(internal.Shape{s.Len()}, internal.Bool)

	for i := 0; i < s.Len(); i++ {
		val := s.data.At(i)
		compResult, err := performOperation(val, scalar, op)
		if err != nil {
			return nil, fmt.Errorf("failed to compare value at index %d with scalar: %v", i, err)
		}

		err = result.Set(compResult, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set comparison result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, s.index.Copy(), s.name)
}

// performOperation executes the specified operation on two values
func performOperation(left, right interface{}, op operationType) (interface{}, error) {
	// Convert values to float64 for arithmetic operations (except comparisons)
	var leftFloat, rightFloat float64
	var err error

	if op <= modOperation {
		// Arithmetic operations
		leftFloat, err = convertToFloat64(left)
		if err != nil {
			return nil, fmt.Errorf("cannot convert left operand to float64: %v", err)
		}

		rightFloat, err = convertToFloat64(right)
		if err != nil {
			return nil, fmt.Errorf("cannot convert right operand to float64: %v", err)
		}

		switch op {
		case addOperation:
			return leftFloat + rightFloat, nil
		case subOperation:
			return leftFloat - rightFloat, nil
		case mulOperation:
			return leftFloat * rightFloat, nil
		case divOperation:
			if rightFloat == 0 {
				return math.Inf(1), nil // Return +Inf for division by zero
			}
			return leftFloat / rightFloat, nil
		case powOperation:
			return math.Pow(leftFloat, rightFloat), nil
		case modOperation:
			if rightFloat == 0 {
				return math.NaN(), nil
			}
			return math.Mod(leftFloat, rightFloat), nil
		}
	} else {
		// Comparison operations - can work with any type
		switch op {
		case eqOperation:
			return compareValues(left, right) == 0, nil
		case neOperation:
			return compareValues(left, right) != 0, nil
		case ltOperation:
			return compareValues(left, right) < 0, nil
		case leOperation:
			return compareValues(left, right) <= 0, nil
		case gtOperation:
			return compareValues(left, right) > 0, nil
		case geOperation:
			return compareValues(left, right) >= 0, nil
		}
	}

	return nil, fmt.Errorf("unknown operation: %v", op)
}

// compareValues compares two values, returning -1, 0, or 1
func compareValues(left, right interface{}) int {
	// Try to convert both to float64 for numeric comparison
	leftFloat, leftErr := convertToFloat64(left)
	rightFloat, rightErr := convertToFloat64(right)

	if leftErr == nil && rightErr == nil {
		if leftFloat < rightFloat {
			return -1
		} else if leftFloat > rightFloat {
			return 1
		}
		return 0
	}

	// Fallback to string comparison
	leftStr := fmt.Sprintf("%v", left)
	rightStr := fmt.Sprintf("%v", right)

	if leftStr < rightStr {
		return -1
	} else if leftStr > rightStr {
		return 1
	}
	return 0
}

// negateValue returns the negation of a value
func negateValue(value interface{}) (interface{}, error) {
	switch v := value.(type) {
	case float64:
		return -v, nil
	case float32:
		return -v, nil
	case int64:
		return -v, nil
	case int32:
		return -v, nil
	case int16:
		return -v, nil
	case int8:
		return -v, nil
	case complex64:
		return -v, nil
	case complex128:
		return -v, nil
	default:
		// Try converting to float64 and negating
		if f, err := convertToFloat64(value); err == nil {
			return -f, nil
		}
		return nil, fmt.Errorf("cannot negate value of type %T", value)
	}
}

// absoluteValue returns the absolute value of a value
func absoluteValue(value interface{}) (interface{}, error) {
	switch v := value.(type) {
	case float64:
		return math.Abs(v), nil
	case float32:
		return float32(math.Abs(float64(v))), nil
	case int64:
		if v < 0 {
			return -v, nil
		}
		return v, nil
	case int32:
		if v < 0 {
			return -v, nil
		}
		return v, nil
	case int16:
		if v < 0 {
			return -v, nil
		}
		return v, nil
	case int8:
		if v < 0 {
			return -v, nil
		}
		return v, nil
	case complex64:
		real := float32(real(v))
		imag := float32(imag(v))
		return float32(math.Sqrt(float64(real*real + imag*imag))), nil
	case complex128:
		real := real(v)
		imag := imag(v)
		return math.Sqrt(real*real + imag*imag), nil
	default:
		// Try converting to float64 and getting absolute value
		if f, err := convertToFloat64(value); err == nil {
			return math.Abs(f), nil
		}
		return nil, fmt.Errorf("cannot get absolute value of type %T", value)
	}
}

// promoteDataTypes returns the promoted data type for operations between two types
func promoteDataTypes(dtype1, dtype2 internal.DType) internal.DType {
	// Promotion hierarchy: Complex128 > Complex64 > Float64 > Float32 > Int64 > Int32 > Bool
	typeRank := map[internal.DType]int{
		internal.Bool:       0,
		internal.Int8:       1,
		internal.Int16:      2,
		internal.Int32:      3,
		internal.Int64:      4,
		internal.Uint8:      1,
		internal.Uint16:     2,
		internal.Uint32:     3,
		internal.Uint64:     4,
		internal.Float32:    5,
		internal.Float64:    6,
		internal.Complex64:  7,
		internal.Complex128: 8,
	}

	rank1 := typeRank[dtype1]
	rank2 := typeRank[dtype2]

	if rank1 >= rank2 {
		return dtype1
	}
	return dtype2
}

// getNaNValue returns an appropriate NaN/missing value for the given data type
func getNaNValue(dtype internal.DType) interface{} {
	switch dtype {
	case internal.Float64:
		return math.NaN()
	case internal.Float32:
		return float32(math.NaN())
	case internal.Complex64:
		return complex64(complex(float32(math.NaN()), 0))
	case internal.Complex128:
		return complex(math.NaN(), 0)
	case internal.Int64:
		return int64(0) // Could define a special NaN value for integers
	case internal.Int32:
		return int32(0)
	case internal.Bool:
		return false
	default:
		return nil
	}
}
