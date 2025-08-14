//go:build arm64 || arm

package internal

import (
	"math"
	"unsafe"
)

// NEONProvider implements SIMD operations using ARM NEON instructions
type NEONProvider struct {
	capability SIMDCapability
}

// NewNEONProvider creates a new NEON SIMD provider
func NewNEONProvider() *NEONProvider {
	capability := SIMDNEON
	features := GetCPUFeatures()
	if features.HasASIMD || (features.HasNEON && features.HasNEONFP) {
		capability = SIMDNEONV8
	}

	return &NEONProvider{
		capability: capability,
	}
}

func (p *NEONProvider) Capability() SIMDCapability {
	return p.capability
}

func (p *NEONProvider) VectorWidth() int {
	return 16 // NEON operates on 128-bit vectors
}

// AddFloat64 performs vectorized addition of float64 slices using NEON
func (p *NEONProvider) AddFloat64(a, b, result []float64, n int) {
	// Process 2 float64 values at a time (128-bit NEON vector)
	vectorSize := 2
	i := 0

	// Process vectorized elements
	for ; i <= n-vectorSize; i += vectorSize {
		// Use NEON intrinsics for vectorized addition
		addFloat64NEON(&a[i], &b[i], &result[i])
	}

	// Handle remaining elements with scalar operations
	for ; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

// SubFloat64 performs vectorized subtraction using NEON
func (p *NEONProvider) SubFloat64(a, b, result []float64, n int) {
	vectorSize := 2
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		subFloat64NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] - b[i]
	}
}

// MulFloat64 performs vectorized multiplication using NEON
func (p *NEONProvider) MulFloat64(a, b, result []float64, n int) {
	vectorSize := 2
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		mulFloat64NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

// DivFloat64 performs vectorized division using NEON
func (p *NEONProvider) DivFloat64(a, b, result []float64, n int) {
	vectorSize := 2
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		divFloat64NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] / b[i]
	}
}

// AddFloat32 performs vectorized addition of float32 slices using NEON
func (p *NEONProvider) AddFloat32(a, b, result []float32, n int) {
	// Process 4 float32 values at a time (128-bit NEON vector)
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		addFloat32NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

// SubFloat32 performs vectorized subtraction of float32 using NEON
func (p *NEONProvider) SubFloat32(a, b, result []float32, n int) {
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		subFloat32NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] - b[i]
	}
}

// MulFloat32 performs vectorized multiplication of float32 using NEON
func (p *NEONProvider) MulFloat32(a, b, result []float32, n int) {
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		mulFloat32NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

// DivFloat32 performs vectorized division of float32 using NEON
func (p *NEONProvider) DivFloat32(a, b, result []float32, n int) {
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		divFloat32NEON(&a[i], &b[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] / b[i]
	}
}

// Scalar operations
func (p *NEONProvider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	vectorSize := 2
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		addScalarFloat64NEON(&a[i], scalar, &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] + scalar
	}
}

func (p *NEONProvider) MulScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	vectorSize := 2
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		mulScalarFloat64NEON(&a[i], scalar, &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] * scalar
	}
}

func (p *NEONProvider) AddScalarFloat32(a []float32, scalar float32, result []float32, n int) {
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		addScalarFloat32NEON(&a[i], scalar, &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] + scalar
	}
}

func (p *NEONProvider) MulScalarFloat32(a []float32, scalar float32, result []float32, n int) {
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		mulScalarFloat32NEON(&a[i], scalar, &result[i])
	}

	for ; i < n; i++ {
		result[i] = a[i] * scalar
	}
}

// Statistical operations
func (p *NEONProvider) SumFloat64(a []float64, n int) float64 {
	return sumFloat64NEON(a, n)
}

func (p *NEONProvider) SumFloat32(a []float32, n int) float32 {
	return sumFloat32NEON(a, n)
}

func (p *NEONProvider) MeanFloat64(a []float64, n int) float64 {
	if n == 0 {
		return 0
	}
	return p.SumFloat64(a, n) / float64(n)
}

func (p *NEONProvider) VarianceFloat64(a []float64, n int) float64 {
	if n <= 1 {
		return 0
	}

	mean := p.MeanFloat64(a, n)
	return varianceFloat64NEON(a, mean, n)
}

func (p *NEONProvider) DotProductFloat64(a, b []float64, n int) float64 {
	return dotProductFloat64NEON(a, b, n)
}

// Advanced mathematical functions
func (p *NEONProvider) ExpFloat64(a, result []float64, n int) {
	// NEON doesn't have native exp, so we use scalar fallback for now
	// Future optimization could use polynomial approximations with NEON
	for i := 0; i < n; i++ {
		result[i] = math.Exp(a[i])
	}
}

func (p *NEONProvider) LogFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Log(a[i])
	}
}

func (p *NEONProvider) SqrtFloat64(a, result []float64, n int) {
	vectorSize := 2
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		sqrtFloat64NEON(&a[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = math.Sqrt(a[i])
	}
}

func (p *NEONProvider) PowFloat64(a, b, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Pow(a[i], b[i])
	}
}

func (p *NEONProvider) SinFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Sin(a[i])
	}
}

func (p *NEONProvider) CosFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Cos(a[i])
	}
}

func (p *NEONProvider) ExpFloat32(a, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = float32(math.Exp(float64(a[i])))
	}
}

func (p *NEONProvider) SqrtFloat32(a, result []float32, n int) {
	vectorSize := 4
	i := 0

	for ; i <= n-vectorSize; i += vectorSize {
		sqrtFloat32NEON(&a[i], &result[i])
	}

	for ; i < n; i++ {
		result[i] = float32(math.Sqrt(float64(a[i])))
	}
}

// Utility functions
func (p *NEONProvider) IsAligned(ptr unsafe.Pointer) bool {
	return uintptr(ptr)%16 == 0 // NEON requires 16-byte alignment
}

func (p *NEONProvider) AlignmentRequirement() int {
	return 16 // NEON requires 16-byte alignment
}
