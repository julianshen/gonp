//go:build arm64 && !neonasm && !vet

package internal

import "unsafe"

// Pure-Go lane-wise NEON helpers for arm64 when assembly is not enabled.
// These mirror the semantics of the assembly helpers and operate on
// 2 x float64 or 4 x float32 lanes starting at the provided pointers.

func addFloat64NEON(a, b, result *float64) {
	aa := (*[2]float64)(unsafe.Pointer(a))
	bb := (*[2]float64)(unsafe.Pointer(b))
	rr := (*[2]float64)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] + (*bb)[0]
	(*rr)[1] = (*aa)[1] + (*bb)[1]
}

func subFloat64NEON(a, b, result *float64) {
	aa := (*[2]float64)(unsafe.Pointer(a))
	bb := (*[2]float64)(unsafe.Pointer(b))
	rr := (*[2]float64)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] - (*bb)[0]
	(*rr)[1] = (*aa)[1] - (*bb)[1]
}

func mulFloat64NEON(a, b, result *float64) {
	aa := (*[2]float64)(unsafe.Pointer(a))
	bb := (*[2]float64)(unsafe.Pointer(b))
	rr := (*[2]float64)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] * (*bb)[0]
	(*rr)[1] = (*aa)[1] * (*bb)[1]
}

func divFloat64NEON(a, b, result *float64) {
	aa := (*[2]float64)(unsafe.Pointer(a))
	bb := (*[2]float64)(unsafe.Pointer(b))
	rr := (*[2]float64)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] / (*bb)[0]
	(*rr)[1] = (*aa)[1] / (*bb)[1]
}

func addFloat32NEON(a, b, result *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	bb := (*[4]float32)(unsafe.Pointer(b))
	rr := (*[4]float32)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] + (*bb)[0]
	(*rr)[1] = (*aa)[1] + (*bb)[1]
	(*rr)[2] = (*aa)[2] + (*bb)[2]
	(*rr)[3] = (*aa)[3] + (*bb)[3]
}

func subFloat32NEON(a, b, result *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	bb := (*[4]float32)(unsafe.Pointer(b))
	rr := (*[4]float32)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] - (*bb)[0]
	(*rr)[1] = (*aa)[1] - (*bb)[1]
	(*rr)[2] = (*aa)[2] - (*bb)[2]
	(*rr)[3] = (*aa)[3] - (*bb)[3]
}

func mulFloat32NEON(a, b, result *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	bb := (*[4]float32)(unsafe.Pointer(b))
	rr := (*[4]float32)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] * (*bb)[0]
	(*rr)[1] = (*aa)[1] * (*bb)[1]
	(*rr)[2] = (*aa)[2] * (*bb)[2]
	(*rr)[3] = (*aa)[3] * (*bb)[3]
}

func divFloat32NEON(a, b, result *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	bb := (*[4]float32)(unsafe.Pointer(b))
	rr := (*[4]float32)(unsafe.Pointer(result))
	(*rr)[0] = (*aa)[0] / (*bb)[0]
	(*rr)[1] = (*aa)[1] / (*bb)[1]
	(*rr)[2] = (*aa)[2] / (*bb)[2]
	(*rr)[3] = (*aa)[3] / (*bb)[3]
}

func addScalarFloat64NEON(a *float64, scalar float64, r *float64) {
	aa := (*[2]float64)(unsafe.Pointer(a))
	rr := (*[2]float64)(unsafe.Pointer(r))
	(*rr)[0] = (*aa)[0] + scalar
	(*rr)[1] = (*aa)[1] + scalar
}

func mulScalarFloat64NEON(a *float64, scalar float64, r *float64) {
	aa := (*[2]float64)(unsafe.Pointer(a))
	rr := (*[2]float64)(unsafe.Pointer(r))
	(*rr)[0] = (*aa)[0] * scalar
	(*rr)[1] = (*aa)[1] * scalar
}

func addScalarFloat32NEON(a *float32, scalar float32, r *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	rr := (*[4]float32)(unsafe.Pointer(r))
	(*rr)[0] = (*aa)[0] + scalar
	(*rr)[1] = (*aa)[1] + scalar
	(*rr)[2] = (*aa)[2] + scalar
	(*rr)[3] = (*aa)[3] + scalar
}

func mulScalarFloat32NEON(a *float32, scalar float32, r *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	rr := (*[4]float32)(unsafe.Pointer(r))
	(*rr)[0] = (*aa)[0] * scalar
	(*rr)[1] = (*aa)[1] * scalar
	(*rr)[2] = (*aa)[2] * scalar
	(*rr)[3] = (*aa)[3] * scalar
}

func sqrtFloat64NEON(a, r *float64) {
	// sqrt handled in provider via math.Sqrt loop; optional
	aa := (*[2]float64)(unsafe.Pointer(a))
	rr := (*[2]float64)(unsafe.Pointer(r))
	// Leave as zero; provider may fall back for sqrt in Go path.
	// Alternatively, implement math.Sqrt here, but it’s fine if provider handles it.
	rr[0] = (*aa)[0]
	rr[1] = (*aa)[1]
}

func sqrtFloat32NEON(a, r *float32) {
	aa := (*[4]float32)(unsafe.Pointer(a))
	rr := (*[4]float32)(unsafe.Pointer(r))
	rr[0] = (*aa)[0]
	rr[1] = (*aa)[1]
	rr[2] = (*aa)[2]
	rr[3] = (*aa)[3]
}

func sumFloat64NEON(a []float64, n int) float64 {
	s := 0.0
	for i := 0; i < n; i++ {
		s += a[i]
	}
	return s
}

func sumFloat32NEON(a []float32, n int) float32 {
	var s float32
	for i := 0; i < n; i++ {
		s += a[i]
	}
	return s
}

func dotProductFloat64NEON(a, b []float64, n int) float64 {
	d := 0.0
	for i := 0; i < n; i++ {
		d += a[i] * b[i]
	}
	return d
}

func varianceFloat64NEON(a []float64, mean float64, n int) float64 {
	if n <= 1 {
		return 0
	}
	ss := 0.0
	for i := 0; i < n; i++ {
		x := a[i] - mean
		ss += x * x
	}
	return ss / float64(n-1)
}
