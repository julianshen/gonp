//go:build !arm64 && !arm

package internal

// detectARMFeatures is a no-op for non-ARM architectures
func detectARMFeatures(features *CPUFeatures) {
	// No ARM features on non-ARM architectures
}
