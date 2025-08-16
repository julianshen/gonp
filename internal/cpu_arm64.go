//go:build arm64 && !vet

package internal

// detectARMFeatures detects ARM64 SIMD capabilities
func detectARMFeatures(features *CPUFeatures) {
	// ARM64 has NEON (Advanced SIMD) by default
	features.HasNEON = true
	features.HasNEONFP = true
	features.HasASIMD = true
	features.HasFP = true

	// Assume CRC32 and Crypto extensions available on modern ARM64; adjust as needed
	features.HasCRC32 = true
	features.HasCrypto = true

	DebugInfo("ARM64 Features detected: NEON=%v, NEONFP=%v, ASIMD=%v, FP=%v",
		features.HasNEON, features.HasNEONFP, features.HasASIMD, features.HasFP)
}
