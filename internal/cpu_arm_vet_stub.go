//go:build vet

package internal

// During `go vet`, include a no-op implementation to avoid build-tag
// resolution issues in certain sandboxed environments.
func detectARMFeatures(features *CPUFeatures) {
	// Simulate ARM64 NEON availability during vet/test with -tags vet
	features.HasNEON = true
	features.HasNEONFP = true
	features.HasASIMD = true
	features.HasFP = true
}
