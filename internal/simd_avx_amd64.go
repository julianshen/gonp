//go:build amd64
// +build amd64

package internal

import "unsafe"

// Assembly function declarations
func avxAddFloat64(a, b, result *float64, n int)
func avxMulFloat64(a, b, result *float64, n int)
func avxSumFloat64(a *float64, n int) float64
func avxAddScalarFloat64(a *float64, scalar float64, result *float64, n int)
func avxSqrtFloat64(a, result *float64, n int)
func avxDotProductFloat64(a, b *float64, n int) float64

// AVX implementation overrides for AVXProvider

func (p *AVXProvider) AddFloat64(a, b, result []float64, n int) {
	if n == 0 || len(a) == 0 || len(b) == 0 || len(result) == 0 {
		return
	}

	// Ensure we don't exceed slice bounds
	if n > len(a) {
		n = len(a)
	}
	if n > len(b) {
		n = len(b)
	}
	if n > len(result) {
		n = len(result)
	}

	// Check alignment and size threshold
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	resultPtr := unsafe.Pointer(&result[0])

	if ShouldUseSIMD(n, aPtr, bPtr, resultPtr) {
		DebugVerbose("Using AVX for AddFloat64: n=%d", n)
		avxAddFloat64(&a[0], &b[0], &result[0], n)
		IncrementOperations()
	} else {
		DebugVerbose("Using scalar for AddFloat64: n=%d", n)
		p.ScalarProvider.AddFloat64(a, b, result, n)
	}
}

func (p *AVXProvider) MulFloat64(a, b, result []float64, n int) {
	if n == 0 || len(a) == 0 || len(b) == 0 || len(result) == 0 {
		return
	}

	// Ensure we don't exceed slice bounds
	if n > len(a) {
		n = len(a)
	}
	if n > len(b) {
		n = len(b)
	}
	if n > len(result) {
		n = len(result)
	}

	// Check alignment and size threshold
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	resultPtr := unsafe.Pointer(&result[0])

	if ShouldUseSIMD(n, aPtr, bPtr, resultPtr) {
		DebugVerbose("Using AVX for MulFloat64: n=%d", n)
		avxMulFloat64(&a[0], &b[0], &result[0], n)
		IncrementOperations()
	} else {
		DebugVerbose("Using scalar for MulFloat64: n=%d", n)
		p.ScalarProvider.MulFloat64(a, b, result, n)
	}
}

func (p *AVXProvider) SumFloat64(a []float64, n int) float64 {
	if n == 0 || len(a) == 0 {
		return 0.0
	}

	// Ensure we don't exceed slice bounds
	if n > len(a) {
		n = len(a)
	}

	// Check alignment and size threshold
	aPtr := unsafe.Pointer(&a[0])

	if ShouldUseSIMD(n, aPtr) {
		DebugVerbose("Using AVX for SumFloat64: n=%d", n)
		result := avxSumFloat64(&a[0], n)
		IncrementOperations()
		return result
	} else {
		DebugVerbose("Using scalar for SumFloat64: n=%d", n)
		return p.ScalarProvider.SumFloat64(a, n)
	}
}

func (p *AVXProvider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	if n == 0 || len(a) == 0 || len(result) == 0 {
		return
	}

	// Ensure we don't exceed slice bounds
	if n > len(a) {
		n = len(a)
	}
	if n > len(result) {
		n = len(result)
	}

	// Check alignment and size threshold
	aPtr := unsafe.Pointer(&a[0])
	resultPtr := unsafe.Pointer(&result[0])

	if ShouldUseSIMD(n, aPtr, resultPtr) {
		DebugVerbose("Using AVX for AddScalarFloat64: n=%d", n)
		avxAddScalarFloat64(&a[0], scalar, &result[0], n)
		IncrementOperations()
	} else {
		DebugVerbose("Using scalar for AddScalarFloat64: n=%d", n)
		p.ScalarProvider.AddScalarFloat64(a, scalar, result, n)
	}
}

func (p *AVXProvider) MulScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	// For now, use scalar implementation with potential for future AVX optimization
	p.ScalarProvider.MulScalarFloat64(a, scalar, result, n)
}

func (p *AVXProvider) SqrtFloat64(a, result []float64, n int) {
	if n == 0 || len(a) == 0 || len(result) == 0 {
		return
	}

	// Ensure we don't exceed slice bounds
	if n > len(a) {
		n = len(a)
	}
	if n > len(result) {
		n = len(result)
	}

	// Check alignment and size threshold
	aPtr := unsafe.Pointer(&a[0])
	resultPtr := unsafe.Pointer(&result[0])

	if ShouldUseSIMD(n, aPtr, resultPtr) {
		DebugVerbose("Using AVX for SqrtFloat64: n=%d", n)
		avxSqrtFloat64(&a[0], &result[0], n)
		IncrementOperations()
	} else {
		DebugVerbose("Using scalar for SqrtFloat64: n=%d", n)
		p.ScalarProvider.SqrtFloat64(a, result, n)
	}
}

func (p *AVXProvider) DotProductFloat64(a, b []float64, n int) float64 {
	if n == 0 || len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	// Ensure we don't exceed slice bounds
	if n > len(a) {
		n = len(a)
	}
	if n > len(b) {
		n = len(b)
	}

	// Check alignment and size threshold
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])

	if ShouldUseSIMD(n, aPtr, bPtr) {
		DebugVerbose("Using AVX for DotProductFloat64: n=%d", n)
		result := avxDotProductFloat64(&a[0], &b[0], n)
		IncrementOperations()
		return result
	} else {
		DebugVerbose("Using scalar for DotProductFloat64: n=%d", n)
		return p.ScalarProvider.DotProductFloat64(a, b, n)
	}
}

// AVX2Provider inherits AVX implementations and can override specific functions
func (p *AVX2Provider) AddFloat64(a, b, result []float64, n int) {
	// Use AVX implementation for now - can be optimized further with AVX2 specific instructions
	p.AVXProvider.AddFloat64(a, b, result, n)
}

func (p *AVX2Provider) MulFloat64(a, b, result []float64, n int) {
	// Use AVX implementation for now
	p.AVXProvider.MulFloat64(a, b, result, n)
}

func (p *AVX2Provider) SumFloat64(a []float64, n int) float64 {
	// Use AVX implementation for now
	return p.AVXProvider.SumFloat64(a, n)
}

func (p *AVX2Provider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	// Use AVX implementation for now
	p.AVXProvider.AddScalarFloat64(a, scalar, result, n)
}

// AVX512Provider inherits AVX2 implementations
func (p *AVX512Provider) AddFloat64(a, b, result []float64, n int) {
	// Use AVX2 implementation for now - can be optimized with AVX-512
	p.AVX2Provider.AddFloat64(a, b, result, n)
}

func (p *AVX512Provider) MulFloat64(a, b, result []float64, n int) {
	// Use AVX2 implementation for now
	p.AVX2Provider.MulFloat64(a, b, result, n)
}

func (p *AVX512Provider) SumFloat64(a []float64, n int) float64 {
	// Use AVX2 implementation for now
	return p.AVX2Provider.SumFloat64(a, n)
}

func (p *AVX512Provider) AddScalarFloat64(a []float64, scalar float64, result []float64, n int) {
	// Use AVX2 implementation for now
	p.AVX2Provider.AddScalarFloat64(a, scalar, result, n)
}
