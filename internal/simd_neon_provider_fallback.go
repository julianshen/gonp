//go:build (!arm64 && !arm) || vet

package internal

// NewNEONProvider provides a safe fallback on non-ARM or vet builds
func NewNEONProvider() SIMDProvider {
	return NewScalarProvider()
}
