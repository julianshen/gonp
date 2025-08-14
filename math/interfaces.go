package math

import "github.com/julianshen/gonp/array"

// ArithmeticOps defines element-wise arithmetic operations interface
type ArithmeticOps interface {
	// Add performs element-wise addition with broadcasting
	Add(other *array.Array) (*array.Array, error)

	// Sub performs element-wise subtraction with broadcasting
	Sub(other *array.Array) (*array.Array, error)

	// Mul performs element-wise multiplication with broadcasting
	Mul(other *array.Array) (*array.Array, error)

	// Div performs element-wise division with broadcasting
	Div(other *array.Array) (*array.Array, error)

	// Mod performs element-wise modulo with broadcasting
	Mod(other *array.Array) (*array.Array, error)

	// Pow performs element-wise power with broadcasting
	Pow(other *array.Array) (*array.Array, error)
}

// ComparisonOps defines element-wise comparison operations interface
type ComparisonOps interface {
	// Equal performs element-wise equality comparison
	Equal(other *array.Array) (*array.Array, error)

	// NotEqual performs element-wise inequality comparison
	NotEqual(other *array.Array) (*array.Array, error)

	// Less performs element-wise less than comparison
	Less(other *array.Array) (*array.Array, error)

	// LessEqual performs element-wise less than or equal comparison
	LessEqual(other *array.Array) (*array.Array, error)

	// Greater performs element-wise greater than comparison
	Greater(other *array.Array) (*array.Array, error)

	// GreaterEqual performs element-wise greater than or equal comparison
	GreaterEqual(other *array.Array) (*array.Array, error)
}

// UniversalFunc defines universal mathematical functions interface
type UniversalFunc interface {
	// Trigonometric functions
	Sin() *array.Array
	Cos() *array.Array
	Tan() *array.Array
	Asin() *array.Array
	Acos() *array.Array
	Atan() *array.Array

	// Hyperbolic functions
	Sinh() *array.Array
	Cosh() *array.Array
	Tanh() *array.Array

	// Exponential and logarithmic functions
	Exp() *array.Array
	Exp2() *array.Array
	Log() *array.Array
	Log2() *array.Array
	Log10() *array.Array

	// Power and root functions
	Sqrt() *array.Array
	Cbrt() *array.Array
	Square() *array.Array

	// Rounding functions
	Ceil() *array.Array
	Floor() *array.Array
	Round() *array.Array
	Trunc() *array.Array

	// Sign and absolute functions
	Abs() *array.Array
	Sign() *array.Array

	// Special functions
	IsNaN() *array.Array
	IsInf() *array.Array
	IsFinite() *array.Array
}

// ReductionOps defines reduction operations interface
type ReductionOps interface {
	// Sum computes sum along specified axes
	Sum(axis ...int) (*array.Array, error)

	// Mean computes arithmetic mean along specified axes
	Mean(axis ...int) (*array.Array, error)

	// Max finds maximum values along specified axes
	Max(axis ...int) (*array.Array, error)

	// Min finds minimum values along specified axes
	Min(axis ...int) (*array.Array, error)

	// Std computes standard deviation along specified axes
	Std(axis ...int) (*array.Array, error)

	// Var computes variance along specified axes
	Var(axis ...int) (*array.Array, error)

	// Prod computes product along specified axes
	Prod(axis ...int) (*array.Array, error)

	// ArgMax finds indices of maximum values along specified axes
	ArgMax(axis ...int) (*array.Array, error)

	// ArgMin finds indices of minimum values along specified axes
	ArgMin(axis ...int) (*array.Array, error)

	// All tests whether all elements evaluate to True along specified axes
	All(axis ...int) (*array.Array, error)

	// Any tests whether any element evaluates to True along specified axes
	Any(axis ...int) (*array.Array, error)
}

// BroadcastingOps defines broadcasting operations interface
type BroadcastingOps interface {
	// BroadcastTo broadcasts array to specified shape
	BroadcastTo(shape []int) (*array.Array, error)

	// CanBroadcast checks if two shapes can be broadcast together
	CanBroadcast(shape1, shape2 []int) bool

	// BroadcastShapes determines the result shape of broadcasting two shapes
	BroadcastShapes(shape1, shape2 []int) ([]int, error)
}

// LinearAlgebraOps defines linear algebra operations interface
type LinearAlgebraOps interface {
	// Dot performs matrix multiplication or dot product
	Dot(other *array.Array) (*array.Array, error)

	// MatMul performs matrix multiplication (same as Dot but stricter about dimensions)
	MatMul(other *array.Array) (*array.Array, error)

	// Cross computes cross product for 3D vectors
	Cross(other *array.Array) (*array.Array, error)

	// Norm computes vector or matrix norm
	Norm(ord interface{}, axis ...int) (*array.Array, error)
}

// ScalarOps defines operations with scalar values interface
type ScalarOps interface {
	// AddScalar adds a scalar value to all elements
	AddScalar(scalar interface{}) (*array.Array, error)

	// SubScalar subtracts a scalar value from all elements
	SubScalar(scalar interface{}) (*array.Array, error)

	// MulScalar multiplies all elements by a scalar value
	MulScalar(scalar interface{}) (*array.Array, error)

	// DivScalar divides all elements by a scalar value
	DivScalar(scalar interface{}) (*array.Array, error)

	// PowScalar raises all elements to a scalar power
	PowScalar(scalar interface{}) (*array.Array, error)
}

// StatisticalOps defines statistical operations interface
type StatisticalOps interface {
	// Median computes median along specified axes
	Median(axis ...int) (*array.Array, error)

	// Percentile computes percentile along specified axes
	Percentile(q float64, axis ...int) (*array.Array, error)

	// Quantile computes quantile along specified axes
	Quantile(q float64, axis ...int) (*array.Array, error)

	// Histogram computes histogram of array values
	Histogram(bins int) (values *array.Array, binEdges *array.Array, err error)

	// Corrcoef computes correlation coefficient matrix
	Corrcoef() (*array.Array, error)

	// Cov computes covariance matrix
	Cov() (*array.Array, error)
}

// ArrayManipulationOps defines array manipulation operations interface
type ArrayManipulationOps interface {
	// Concatenate joins arrays along existing axis
	Concatenate(other *array.Array, axis int) (*array.Array, error)

	// Stack joins arrays along new axis
	Stack(other *array.Array, axis int) (*array.Array, error)

	// Split divides array into sub-arrays
	Split(indices []int, axis int) ([]*array.Array, error)

	// Squeeze removes single-dimensional entries
	Squeeze(axis ...int) (*array.Array, error)

	// ExpandDims expands array dimensions
	ExpandDims(axis int) (*array.Array, error)

	// Flatten returns a flattened copy of the array
	Flatten() *array.Array

	// Ravel returns a flattened view of the array if possible
	Ravel() (*array.Array, error)
}

// MaskingOps defines masking and conditional operations interface
type MaskingOps interface {
	// Where returns elements chosen from x or y depending on condition
	Where(condition, x, y *array.Array) (*array.Array, error)

	// MaskedWhere applies mask and returns elements where condition is true
	MaskedWhere(condition *array.Array) (*array.Array, error)

	// Select returns elements chosen from choices based on conditions
	Select(condlist []*array.Array, choicelist []*array.Array, default_ interface{}) (*array.Array, error)

	// Clip limits values to specified range
	Clip(min, max interface{}) (*array.Array, error)
}

// SortingOps defines sorting and searching operations interface
type SortingOps interface {
	// Sort returns sorted copy of array
	Sort(axis int) (*array.Array, error)

	// ArgSort returns indices that would sort the array
	ArgSort(axis int) (*array.Array, error)

	// SearchSorted finds indices where elements should be inserted to maintain order
	SearchSorted(values *array.Array, side string) (*array.Array, error)

	// Unique finds unique elements of array
	Unique() (*array.Array, *array.Array, *array.Array, error) // values, indices, inverse, counts

	// Partition partitions array elements
	Partition(kth int, axis int) (*array.Array, error)
}

// NaNHandlingOps defines NaN-aware operations interface
type NaNHandlingOps interface {
	// NaNSum computes sum, ignoring NaN values
	NaNSum(axis ...int) (*array.Array, error)

	// NaNMean computes mean, ignoring NaN values
	NaNMean(axis ...int) (*array.Array, error)

	// NaNMax finds maximum, ignoring NaN values
	NaNMax(axis ...int) (*array.Array, error)

	// NaNMin finds minimum, ignoring NaN values
	NaNMin(axis ...int) (*array.Array, error)

	// NaNStd computes standard deviation, ignoring NaN values
	NaNStd(axis ...int) (*array.Array, error)

	// NaNVar computes variance, ignoring NaN values
	NaNVar(axis ...int) (*array.Array, error)
}
