//go:build arm64 || arm

package internal

import (
	"runtime"
	"strings"
)

// detectARMFeatures detects ARM SIMD capabilities
func detectARMFeatures(features *CPUFeatures) {
	// For ARM64, most features are available by default
	if runtime.GOARCH == "arm64" {
		// ARM64 has NEON (Advanced SIMD) by default
		features.HasNEON = true
		features.HasNEONFP = true
		features.HasASIMD = true
		features.HasFP = true

		// Detect additional ARM64 features through runtime checks
		detectAdvancedARMFeatures(features)
	} else if runtime.GOARCH == "arm" {
		// ARM32 requires more careful detection
		detectARM32Features(features)
	}

	DebugInfo("ARM Features detected: NEON=%v, NEONFP=%v, ASIMD=%v, FP=%v",
		features.HasNEON, features.HasNEONFP, features.HasASIMD, features.HasFP)
}

// detectAdvancedARMFeatures detects advanced ARM features
func detectAdvancedARMFeatures(features *CPUFeatures) {
	// Try to detect crypto extensions and other advanced features
	// This is platform-specific and may require reading /proc/cpuinfo on Linux

	// For now, assume modern ARM64 processors have these features
	// In a real implementation, you would read /proc/cpuinfo or use OS-specific APIs
	features.HasCRC32 = true
	features.HasCrypto = true
}

// detectARM32Features detects ARM32 (32-bit) specific features
func detectARM32Features(features *CPUFeatures) {
	// ARM32 NEON detection is more complex and platform-dependent
	// For now, we'll use a conservative approach

	// Check for NEON availability (simplified)
	// In a real implementation, this would involve:
	// 1. Reading /proc/cpuinfo on Linux
	// 2. Using OS-specific APIs on other platforms
	// 3. Checking CPUID-equivalent information

	features.HasNEON = checkARM32NEON()
	features.HasNEONFP = features.HasNEON // Usually comes together
	features.HasFP = true                 // Most ARM32 processors have FP
}

// checkARM32NEON checks if ARM32 NEON is available
// This is a simplified implementation - production code would be more sophisticated
func checkARM32NEON() bool {
	// Simplified detection - in reality this would check:
	// - /proc/cpuinfo for "neon" flag on Linux
	// - Specific CPU model detection
	// - Runtime capability checking
	return true // Assume NEON is available for now
}

// isARMCPUModelSupported checks if a specific ARM CPU model supports NEON
func isARMCPUModelSupported(cpuModel string) bool {
	// List of ARM CPU models known to support NEON
	supportedModels := []string{
		"cortex-a7", "cortex-a8", "cortex-a9", "cortex-a15", "cortex-a17",
		"cortex-a53", "cortex-a57", "cortex-a72", "cortex-a73", "cortex-a75",
		"cortex-a76", "cortex-a77", "cortex-a78", "cortex-x1", "cortex-x2",
	}

	cpuLower := strings.ToLower(cpuModel)
	for _, model := range supportedModels {
		if strings.Contains(cpuLower, model) {
			return true
		}
	}

	return false
}
