//go:build !arm64 && !arm

package internal

import "unsafe"

// NewNEONProvider is not available on non-ARM architectures
// This fallback returns a scalar provider
func NewNEONProvider() SIMDProvider {
	return NewScalarProvider()
}

// NEON function stubs for non-ARM architectures
// These functions are never called but need to be declared for the linker

func addFloat64NEON(a, b, c unsafe.Pointer, n int)                                   {}
func subFloat64NEON(a, b, c unsafe.Pointer, n int)                                   {}
func mulFloat64NEON(a, b, c unsafe.Pointer, n int)                                   {}
func divFloat64NEON(a, b, c unsafe.Pointer, n int)                                   {}
func addFloat32NEON(a, b, c unsafe.Pointer, n int)                                   {}
func subFloat32NEON(a, b, c unsafe.Pointer, n int)                                   {}
func mulFloat32NEON(a, b, c unsafe.Pointer, n int)                                   {}
func divFloat32NEON(a, b, c unsafe.Pointer, n int)                                   {}
func addScalarFloat64NEON(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int) {}
func mulScalarFloat64NEON(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int) {}
func addScalarFloat32NEON(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) {}
func mulScalarFloat32NEON(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) {}
func sqrtFloat64NEON(a, c unsafe.Pointer, n int)                                     {}
func sqrtFloat32NEON(a, c unsafe.Pointer, n int)                                     {}
func sumFloat64NEON(a unsafe.Pointer, n int) float64                                 { return 0 }
func sumFloat32NEON(a unsafe.Pointer, n int) float32                                 { return 0 }
func dotProductFloat64NEON(a, b unsafe.Pointer, n int) float64                       { return 0 }
func varianceFloat64NEON(a unsafe.Pointer, mean float64, n int) float64              { return 0 }
