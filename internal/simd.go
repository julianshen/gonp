package internal

import (
	"math"
	"unsafe"
)

// SIMDProvider defines the interface for SIMD implementations
type SIMDProvider interface {
	// Capability returns the SIMD capability level
	Capability() SIMDCapability

	// VectorWidth returns the vector width in bytes
	VectorWidth() int

	// Mathematical operations
	AddFloat64(a, b, result []float64, n int)
	SubFloat64(a, b, result []float64, n int)
	MulFloat64(a, b, result []float64, n int)
	DivFloat64(a, b, result []float64, n int)

	AddFloat32(a, b, result []float32, n int)
	SubFloat32(a, b, result []float32, n int)
	MulFloat32(a, b, result []float32, n int)
	DivFloat32(a, b, result []float32, n int)

	// Scalar operations
	AddScalarFloat64(a []float64, scalar float64, result []float64, n int)
	MulScalarFloat64(a []float64, scalar float64, result []float64, n int)
	AddScalarFloat32(a []float32, scalar float32, result []float32, n int)
	MulScalarFloat32(a []float32, scalar float32, result []float32, n int)

	// Statistical operations
	SumFloat64(a []float64, n int) float64
	SumFloat32(a []float32, n int) float32
	MeanFloat64(a []float64, n int) float64
	VarianceFloat64(a []float64, n int) float64
	DotProductFloat64(a, b []float64, n int) float64

	// Advanced mathematical functions
	ExpFloat64(a, result []float64, n int)
	LogFloat64(a, result []float64, n int)
	SqrtFloat64(a, result []float64, n int)
	PowFloat64(a, b, result []float64, n int)
	SinFloat64(a, result []float64, n int)
	CosFloat64(a, result []float64, n int)

	ExpFloat32(a, result []float32, n int)
	SqrtFloat32(a, result []float32, n int)

	// Utility functions
	IsAligned(ptr unsafe.Pointer) bool
	AlignmentRequirement() int
}

// Global SIMD provider - selected at runtime based on CPU capabilities
var globalSIMDProvider SIMDProvider

// Initialize SIMD provider at startup
func init() {
	globalSIMDProvider = selectBestSIMDProvider()
}

// GetSIMDProvider returns the current SIMD provider
func GetSIMDProvider() SIMDProvider {
	return globalSIMDProvider
}

// selectBestSIMDProvider selects the best available SIMD implementation
func selectBestSIMDProvider() SIMDProvider {
	capability := GetBestSIMDCapability()

	switch capability {
	case SIMDAVX512:
		return NewAVX512Provider()
	case SIMDAVX2:
		return NewAVX2Provider()
	case SIMDAVX:
		return NewAVXProvider()
	case SIMDSSE42, SIMDSSE41, SIMDSSE3, SIMDSSE2, SIMDSSE:
		return NewSSEProvider()
	case SIMDNEONV8, SIMDNEON:
		return NewNEONProvider()
	default:
		return NewScalarProvider()
	}
}

// ScalarProvider implements scalar (non-SIMD) operations as fallback
type ScalarProvider struct{}

// NewScalarProvider creates a new scalar provider
func NewScalarProvider() *ScalarProvider {
	return &ScalarProvider{}
}

func (p *ScalarProvider) Capability() SIMDCapability {
	return SIMDNone
}

func (p *ScalarProvider) VectorWidth() int {
	return 8 // Process one element at a time
}

func (p *ScalarProvider) AddFloat64(a, b, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

func (p *ScalarProvider) SubFloat64(a, b, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] - b[i]
	}
}

func (p *ScalarProvider) MulFloat64(a, b, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

func (p *ScalarProvider) DivFloat64(a, b, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] / b[i]
	}
}

func (p *ScalarProvider) AddFloat32(a, b, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] + b[i]
	}
}

func (p *ScalarProvider) SubFloat32(a, b, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] - b[i]
	}
}

func (p *ScalarProvider) MulFloat32(a, b, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] * b[i]
	}
}

func (p *ScalarProvider) DivFloat32(a, b, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] / b[i]
	}
}

func (p *ScalarProvider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] + scalar
	}
}

func (p *ScalarProvider) MulScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] * scalar
	}
}

func (p *ScalarProvider) AddScalarFloat32(a []float32, scalar float32, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] + scalar
	}
}

func (p *ScalarProvider) MulScalarFloat32(a []float32, scalar float32, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = a[i] * scalar
	}
}

func (p *ScalarProvider) SumFloat64(a []float64, n int) float64 {
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += a[i]
	}
	return sum
}

func (p *ScalarProvider) SumFloat32(a []float32, n int) float32 {
	sum := float32(0.0)
	for i := 0; i < n; i++ {
		sum += a[i]
	}
	return sum
}

// Statistical operations
func (p *ScalarProvider) MeanFloat64(a []float64, n int) float64 {
	sum := p.SumFloat64(a, n)
	return sum / float64(n)
}

func (p *ScalarProvider) VarianceFloat64(a []float64, n int) float64 {
	if n <= 1 {
		return 0.0
	}

	mean := p.MeanFloat64(a, n)
	sumSq := 0.0
	for i := 0; i < n; i++ {
		diff := a[i] - mean
		sumSq += diff * diff
	}
	return sumSq / float64(n-1) // Sample variance
}

func (p *ScalarProvider) DotProductFloat64(a, b []float64, n int) float64 {
	dot := 0.0
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
	}
	return dot
}

// Advanced mathematical functions
func (p *ScalarProvider) ExpFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Exp(a[i])
	}
}

func (p *ScalarProvider) LogFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Log(a[i])
	}
}

func (p *ScalarProvider) SqrtFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Sqrt(a[i])
	}
}

func (p *ScalarProvider) PowFloat64(a, b, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Pow(a[i], b[i])
	}
}

func (p *ScalarProvider) SinFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Sin(a[i])
	}
}

func (p *ScalarProvider) CosFloat64(a, result []float64, n int) {
	for i := 0; i < n; i++ {
		result[i] = math.Cos(a[i])
	}
}

func (p *ScalarProvider) ExpFloat32(a, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = float32(math.Exp(float64(a[i])))
	}
}

func (p *ScalarProvider) SqrtFloat32(a, result []float32, n int) {
	for i := 0; i < n; i++ {
		result[i] = float32(math.Sqrt(float64(a[i])))
	}
}

func (p *ScalarProvider) IsAligned(ptr unsafe.Pointer) bool {
	return IsAligned(ptr, 8) // 8-byte alignment for scalar operations
}

func (p *ScalarProvider) AlignmentRequirement() int {
	return 8
}

// SSEProvider implements SSE-optimized operations
type SSEProvider struct {
	*ScalarProvider // Embed scalar provider for fallback
}

// NewSSEProvider creates a new SSE provider
func NewSSEProvider() *SSEProvider {
	return &SSEProvider{
		ScalarProvider: NewScalarProvider(),
	}
}

func (p *SSEProvider) Capability() SIMDCapability {
	return SIMDSSE2 // Use SSE2 as baseline
}

func (p *SSEProvider) VectorWidth() int {
	return 16 // 128 bits = 16 bytes
}

func (p *SSEProvider) AlignmentRequirement() int {
	return 16
}

func (p *SSEProvider) IsAligned(ptr unsafe.Pointer) bool {
	return IsAligned(ptr, 16)
}

// AVXProvider implements AVX-optimized operations
type AVXProvider struct {
	*ScalarProvider // Embed scalar provider for fallback
}

// NewAVXProvider creates a new AVX provider
func NewAVXProvider() *AVXProvider {
	return &AVXProvider{
		ScalarProvider: NewScalarProvider(),
	}
}

func (p *AVXProvider) Capability() SIMDCapability {
	return SIMDAVX
}

func (p *AVXProvider) VectorWidth() int {
	return 32 // 256 bits = 32 bytes
}

func (p *AVXProvider) AlignmentRequirement() int {
	return 32
}

func (p *AVXProvider) IsAligned(ptr unsafe.Pointer) bool {
	return IsAligned(ptr, 32)
}

// AVX2Provider implements AVX2-optimized operations
type AVX2Provider struct {
	*AVXProvider // Embed AVX provider
}

// NewAVX2Provider creates a new AVX2 provider
func NewAVX2Provider() *AVX2Provider {
	return &AVX2Provider{
		AVXProvider: NewAVXProvider(),
	}
}

func (p *AVX2Provider) Capability() SIMDCapability {
	return SIMDAVX2
}

// AVX512Provider implements AVX-512 operations
type AVX512Provider struct {
	*AVX2Provider // Embed AVX2 provider
}

// NewAVX512Provider creates a new AVX-512 provider
func NewAVX512Provider() *AVX512Provider {
	return &AVX512Provider{
		AVX2Provider: NewAVX2Provider(),
	}
}

func (p *AVX512Provider) Capability() SIMDCapability {
	return SIMDAVX512
}

func (p *AVX512Provider) VectorWidth() int {
	return 64 // 512 bits = 64 bytes
}

func (p *AVX512Provider) AlignmentRequirement() int {
	return 64
}

func (p *AVX512Provider) IsAligned(ptr unsafe.Pointer) bool {
	return IsAligned(ptr, 64)
}

// Convenience functions that use the global SIMD provider

// SIMDAddFloat64 performs SIMD-optimized addition of float64 slices
func SIMDAddFloat64(a, b, result []float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if len(result) < n {
		n = len(result)
	}
	globalSIMDProvider.AddFloat64(a, b, result, n)
}

// SIMDMulFloat64 performs SIMD-optimized multiplication of float64 slices
func SIMDMulFloat64(a, b, result []float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if len(result) < n {
		n = len(result)
	}
	globalSIMDProvider.MulFloat64(a, b, result, n)
}

// SIMDSumFloat64 performs SIMD-optimized sum of float64 slice
func SIMDSumFloat64(a []float64) float64 {
	return globalSIMDProvider.SumFloat64(a, len(a))
}

// SIMDAddScalarFloat64 performs SIMD-optimized scalar addition
func SIMDAddScalarFloat64(a []float64, scalar float64, result []float64) {
	n := len(a)
	if len(result) < n {
		n = len(result)
	}
	globalSIMDProvider.AddScalarFloat64(a, scalar, result, n)
}

// SIMDMulScalarFloat64 performs SIMD-optimized scalar multiplication
func SIMDMulScalarFloat64(a []float64, scalar float64, result []float64) {
	n := len(a)
	if len(result) < n {
		n = len(result)
	}
	globalSIMDProvider.MulScalarFloat64(a, scalar, result, n)
}

// Advanced mathematical function convenience wrappers

// SIMDExpFloat64 performs SIMD-optimized exponential function
func SIMDExpFloat64(a, result []float64, n int) {
	globalSIMDProvider.ExpFloat64(a, result, n)
}

// SIMDLogFloat64 performs SIMD-optimized natural logarithm
func SIMDLogFloat64(a, result []float64, n int) {
	globalSIMDProvider.LogFloat64(a, result, n)
}

// SIMDSqrtFloat64 performs SIMD-optimized square root
func SIMDSqrtFloat64(a, result []float64, n int) {
	globalSIMDProvider.SqrtFloat64(a, result, n)
}

// SIMDPowFloat64 performs SIMD-optimized power function
func SIMDPowFloat64(a, b, result []float64, n int) {
	globalSIMDProvider.PowFloat64(a, b, result, n)
}

// SIMDSinFloat64 performs SIMD-optimized sine function
func SIMDSinFloat64(a, result []float64, n int) {
	globalSIMDProvider.SinFloat64(a, result, n)
}

// SIMDCosFloat64 performs SIMD-optimized cosine function
func SIMDCosFloat64(a, result []float64, n int) {
	globalSIMDProvider.CosFloat64(a, result, n)
}

// SIMDExpFloat32 performs SIMD-optimized exponential function for float32
func SIMDExpFloat32(a, result []float32, n int) {
	globalSIMDProvider.ExpFloat32(a, result, n)
}

// SIMDSqrtFloat32 performs SIMD-optimized square root for float32
func SIMDSqrtFloat32(a, result []float32, n int) {
	globalSIMDProvider.SqrtFloat32(a, result, n)
}

// Statistical function convenience wrappers

// SIMDMeanFloat64 performs SIMD-optimized mean calculation
func SIMDMeanFloat64(a []float64, n int) float64 {
	return globalSIMDProvider.MeanFloat64(a, n)
}

// SIMDVarianceFloat64 performs SIMD-optimized variance calculation
func SIMDVarianceFloat64(a []float64, n int) float64 {
	return globalSIMDProvider.VarianceFloat64(a, n)
}

// SIMDDotProductFloat64 performs SIMD-optimized dot product
func SIMDDotProductFloat64(a, b []float64, n int) float64 {
	return globalSIMDProvider.DotProductFloat64(a, b, n)
}

// Performance utilities

// SIMDThreshold defines the minimum array size where SIMD becomes beneficial
const SIMDThreshold = 32

// ShouldUseSIMD determines if SIMD should be used based on array size and alignment
func ShouldUseSIMD(size int, ptrs ...unsafe.Pointer) bool {
	// Don't use SIMD for small arrays
	if size < SIMDThreshold {
		return false
	}

	// Check if all pointers are properly aligned (if any provided)
	provider := GetSIMDProvider()
	for _, ptr := range ptrs {
		if ptr != nil && !provider.IsAligned(ptr) {
			DebugVerbose("SIMD disabled: unaligned memory (ptr=%v, alignment=%d)",
				ptr, provider.AlignmentRequirement())
			return false
		}
	}

	return provider.Capability() != SIMDNone
}

// GetSIMDStatistics returns information about SIMD usage
func GetSIMDStatistics() map[string]interface{} {
	provider := GetSIMDProvider()
	return map[string]interface{}{
		"capability":     provider.Capability().String(),
		"vector_width":   provider.VectorWidth(),
		"alignment_req":  provider.AlignmentRequirement(),
		"simd_threshold": SIMDThreshold,
	}
}
