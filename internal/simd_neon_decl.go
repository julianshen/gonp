//go:build arm64 && !vet

package internal

// Assembly function declarations (implemented in simd_neon_arm64.s)
func addFloat64NEON(a, b, result *float64)
func subFloat64NEON(a, b, result *float64)
func mulFloat64NEON(a, b, result *float64)
func divFloat64NEON(a, b, result *float64)

func addFloat32NEON(a, b, result *float32)
func subFloat32NEON(a, b, result *float32)
func mulFloat32NEON(a, b, result *float32)
func divFloat32NEON(a, b, result *float32)

func addScalarFloat64NEON(a *float64, scalar float64, result *float64)
func mulScalarFloat64NEON(a *float64, scalar float64, result *float64)
func addScalarFloat32NEON(a *float32, scalar float32, result *float32)
func mulScalarFloat32NEON(a *float32, scalar float32, result *float32)

func sqrtFloat64NEON(a, result *float64)
func sqrtFloat32NEON(a, result *float32)

func sumFloat64NEON(a []float64, n int) float64
func sumFloat32NEON(a []float32, n int) float32
func dotProductFloat64NEON(a, b []float64, n int) float64
func varianceFloat64NEON(a []float64, mean float64, n int) float64
