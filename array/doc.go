// Package array provides n-dimensional arrays with NumPy-like functionality for Go.
//
// # Overview
//
// The array package is the core of the GoNP library, providing efficient n-dimensional
// arrays with vectorized operations, SIMD optimizations, and a familiar API for users
// coming from NumPy or MATLAB. Arrays support multiple data types, broadcasting,
// and high-performance mathematical operations.
//
// # Quick Start
//
// Creating arrays:
//
//	import "github.com/julianshen/gonp/array"
//
//	// From Go slices (most common)
//	arr, _ := array.FromSlice([]float64{1, 2, 3, 4, 5})
//	matrix, _ := array.FromSlice([][]float64{{1, 2}, {3, 4}})
//
//	// Create arrays filled with specific values
//	// (Use FromSlice helpers and arithmetic; shape constructors are internal)
//
//	// Using ranges
//	sequence := array.Linspace(0, 10, 50)  // 50 points from 0 to 10
//
// Basic operations:
//
//	// Element access
//	val := arr.At(2)        // Get element at index 2
//	matrix.Set(99, 1, 0)    // Set element at row 1, col 0
//
//	// Mathematical operations (SIMD-optimized)
//	result := arr1.Add(arr2)     // Element-wise addition
//	scaled := arr.MulScalar(2.5) // Multiply by scalar
//	total := arr.Sum()           // Sum all elements
//
//	// Array properties
//	fmt.Printf("Shape: %v\n", arr.Shape())    // Dimensions
//	fmt.Printf("Size: %d\n", arr.Size())      // Total elements
//	fmt.Printf("Type: %v\n", arr.DType())     // Data type
//
// # Key Features
//
// ## Performance Optimizations
//
// - SIMD acceleration for mathematical operations (2-4x speedup on modern CPUs)
// - Automatic CPU feature detection (AVX, AVX2, AVX-512 support)
// - Multi-threading for large operations
// - Memory-efficient storage with minimal overhead
//
// ## Broadcasting Support
//
// Arrays with different but compatible shapes can be used together:
//
//	arr1 := array.FromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})  // Shape: [2, 3]
//	arr2 := array.FromSlice([]float64{10, 20, 30})              // Shape: [3]
//	result := arr1.Add(arr2)  // Broadcasting: arr2 applied to each row
//
// ## Data Types
//
// Supported data types (internal.DType):
//
//   - Float64, Float32: IEEE 754 floating-point
//   - Int64, Int32, Int16, Int8: Signed integers
//   - Uint64, Uint32, Uint16, Uint8: Unsigned integers
//   - Bool: Boolean values
//   - Complex64, Complex128: Complex numbers
//
// ## Memory Management
//
// Arrays use copy semantics by default for safety:
//
//	original := array.FromSlice([]float64{1, 2, 3})
//	copy := original.Copy()      // Deep copy
//	view := original.Slice(1, 3) // View (shares memory)
//
// # Mathematical Operations
//
// ## Element-wise Operations
//
// All basic arithmetic operations support broadcasting:
//
//	// Binary operations
//	sum := a.Add(b)           // Addition
//	diff := a.Sub(b)          // Subtraction
//	prod := a.Mul(b)          // Multiplication
//	quot := a.Div(b)          // Division
//	power := a.Pow(b)         // Exponentiation
//	mod := a.Mod(b)           // Modulo
//
//	// Scalar operations
//	scaled := arr.MulScalar(2.5)
//	shifted := arr.AddScalar(10)
//
// ## Reduction Operations
//
// Reduce arrays along specified axes:
//
//	total := arr.Sum()           // Sum all elements
//	rowSums := arr.Sum(1)        // Sum along axis 1 (columns)
//	average := arr.Mean()        // Mean of all elements
//	maximum := arr.Max(0)        // Max along axis 0 (rows)
//
//	// Statistical operations
//	variance := arr.Var()
//	stddev := arr.Std()
//	cumsum := arr.CumSum(0)      // Cumulative sum
//
// ## Comparison Operations
//
// Element-wise comparisons return boolean arrays:
//
//	mask := arr.Greater(5.0)     // arr > 5.0
//	equal := a.Equal(b)          // a == b
//	different := a.NotEqual(b)   // a != b
//
// # Array Manipulation
//
// ## Shape Operations
//
//	flattened := arr.Flatten()                     // Make 1D
//	transposed := matrix.Transpose()               // Swap dimensions
//
//	// Advanced shape operations
//	squeezed := arr.Squeeze()        // Remove size-1 dimensions
//	expanded := arr.ExpandDims(1)    // Add new dimension
//	swapped := arr.Swapaxes(0, 1)    // Swap two axes
//
// ## Slicing and Indexing
//
//	// Basic slicing (index helpers)
//	// See package indexing methods for helpers to create ranges
//
//	// Boolean indexing
//	mask := arr.Greater(0)
//	positive := arr.BooleanIndex(mask)  // Elements where arr > 0
//
//	// Fancy indexing
//	indices := array.FromSlice([]int{0, 2, 4})
//	selected := arr.FancyIndex(indices)  // Elements at indices 0, 2, 4
//
// # Performance Guidelines
//
// ## SIMD Optimization
//
// - Operations on arrays with >32 elements automatically use SIMD when available
// - Float64 operations are typically fastest due to native CPU support
// - Memory alignment is automatically handled for optimal performance
//
// ## Memory Efficiency
//
//	// Efficient: reuse arrays when possible
//	result := array.Zeros(shape, dtype)
//	a.Add(b, result)  // Store result in pre-allocated array
//
//	// Less efficient: creates new arrays
//	result := a.Add(b).Mul(c).Sub(d)  // Multiple allocations
//
// ## Thread Safety
//
// - Read operations are thread-safe
// - Write operations require external synchronization
// - In-place operations (AddInPlace, etc.) modify the original array
//
// # Integration with Other Packages
//
// Arrays integrate seamlessly with other GoNP components:
//
//	// Series (1D labeled arrays)
//	series := series.NewSeries(arr, index, "data")
//
//	// DataFrames (2D labeled structures)
//	df := dataframe.FromArrays(map[string]*array.Array{
//		"col1": arr1,
//		"col2": arr2,
//	})
//
//	// Mathematical functions
//	result := math.Sin(arr)      // Element-wise sine
//	poly := math.Polynomial(coeffs).Evaluate(arr)
//
//	// Statistical analysis
//	corr := stats.Correlation(arr1, arr2)
//	reg := stats.LinearRegression(x, y)
//
// # Error Handling
//
// The array package uses Go's standard error handling patterns:
//
//	arr, err := array.FromSlice(data)
//	if err != nil {
//		log.Fatal(err)  // Handle creation errors
//	}
//
//	// Some operations panic on invalid indices (like Go slices)
//	val := arr.At(1, 2)  // Panics if out of bounds
//
//	// Others return errors for recoverable issues
//	result, err := arr.Add(incompatibleArray)
//	if err != nil {
//		log.Printf("Broadcasting failed: %v", err)
//	}
//
// # Migration from NumPy
//
// Common NumPy patterns and their GoNP equivalents:
//
//	# NumPy                          // GoNP
//	import numpy as np               import "github.com/julianshen/gonp/array"
//
//	arr = np.array([1, 2, 3])        arr, _ := array.FromSlice([]float64{1, 2, 3})
//	zeros = np.zeros((3, 4))         // Use FromSlice + arithmetic to construct
//	ones = np.ones((2, 3))           // Use FromSlice + arithmetic to construct
//
//	result = arr1 + arr2             result := arr1.Add(arr2)
//	scaled = arr * 2.5               scaled := arr.MulScalar(2.5)
//
//	total = np.sum(arr)              total := arr.Sum()
//	mean = np.mean(arr, axis=0)      mean := arr.Mean(0)
//
//	reshaped = arr.reshape(2, 4)     // Use flatten/transpose and constructors
//	transposed = arr.T               transposed := arr.Transpose()
package array
