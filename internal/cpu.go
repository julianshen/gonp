package internal

import (
	"runtime"
	"sync"
	"unsafe"
)

// CPUFeatures represents available CPU SIMD instruction sets
type CPUFeatures struct {
	// x86/x86_64 features
	HasSSE      bool
	HasSSE2     bool
	HasSSE3     bool
	HasSSSE3    bool
	HasSSE41    bool
	HasSSE42    bool
	HasAVX      bool
	HasAVX2     bool
	HasAVX512F  bool
	HasAVX512DQ bool
	HasAVX512BW bool
	HasAVX512VL bool
	HasFMA      bool

	// ARM features
	HasNEON   bool
	HasNEONFP bool // NEON floating-point support
	HasCRC32  bool // CRC32 instructions
	HasCrypto bool // Crypto extensions
	HasFP     bool // Floating-point unit
	HasASIMD  bool // Advanced SIMD (NEON on AArch64)

	Architecture string
}

// Global CPU features - detected once at startup
var (
	cpuFeatures     *CPUFeatures
	cpuFeaturesOnce sync.Once
)

// GetCPUFeatures returns the detected CPU features (thread-safe, cached)
func GetCPUFeatures() *CPUFeatures {
	cpuFeaturesOnce.Do(func() {
		cpuFeatures = detectCPUFeatures()
	})
	return cpuFeatures
}

// detectCPUFeatures detects available SIMD instruction sets
func detectCPUFeatures() *CPUFeatures {
	features := &CPUFeatures{
		Architecture: runtime.GOARCH,
	}

	// Detect CPU features based on architecture
	switch runtime.GOARCH {
	case "amd64", "386":
		detectX86Features(features)
	case "arm64", "arm":
		detectARMFeatures(features)
	}

	DebugInfo("CPU Features detected: SSE=%v, SSE2=%v, AVX=%v, AVX2=%v, AVX512F=%v",
		features.HasSSE, features.HasSSE2, features.HasAVX, features.HasAVX2, features.HasAVX512F)

	return features
}

// detectX86Features detects x86/x86_64 specific features using CPUID
func detectX86Features(features *CPUFeatures) {
	// Get basic CPUID info
	maxID := cpuid(0, 0)
	if maxID == 0 {
		return
	}

	// CPUID function 1: Processor Info and Feature Bits
	if maxID >= 1 {
		_, _, ecx1, edx1 := cpuidFunc(1)

		// EDX register features
		features.HasSSE = (edx1 & (1 << 25)) != 0
		features.HasSSE2 = (edx1 & (1 << 26)) != 0

		// ECX register features
		features.HasSSE3 = (ecx1 & (1 << 0)) != 0
		features.HasSSSE3 = (ecx1 & (1 << 9)) != 0
		features.HasSSE41 = (ecx1 & (1 << 19)) != 0
		features.HasSSE42 = (ecx1 & (1 << 20)) != 0
		features.HasAVX = (ecx1 & (1 << 28)) != 0
		features.HasFMA = (ecx1 & (1 << 12)) != 0
	}

	// CPUID function 7: Extended Features
	if maxID >= 7 {
		_, ebx7, ecx7, _ := cpuidFunc7(0)

		// EBX register features
		features.HasAVX2 = (ebx7 & (1 << 5)) != 0
		features.HasAVX512F = (ebx7 & (1 << 16)) != 0
		features.HasAVX512DQ = (ebx7 & (1 << 17)) != 0
		features.HasAVX512BW = (ebx7 & (1 << 30)) != 0

		// ECX register features
		features.HasAVX512VL = (ecx7 & (1 << 31)) != 0
	}
}

// SIMDCapability represents the best available SIMD instruction set
type SIMDCapability int

const (
	SIMDNone SIMDCapability = iota
	// x86/x86_64 SIMD capabilities
	SIMDSSE
	SIMDSSE2
	SIMDSSE3
	SIMDSSE41
	SIMDSSE42
	SIMDAVX
	SIMDAVX2
	SIMDAVX512
	// ARM SIMD capabilities
	SIMDNEON   // ARM NEON (32-bit and 64-bit)
	SIMDNEONV8 // ARMv8 Advanced SIMD
)

// String returns string representation of SIMD capability
func (s SIMDCapability) String() string {
	switch s {
	case SIMDNone:
		return "None"
	case SIMDSSE:
		return "SSE"
	case SIMDSSE2:
		return "SSE2"
	case SIMDSSE3:
		return "SSE3"
	case SIMDSSE41:
		return "SSE4.1"
	case SIMDSSE42:
		return "SSE4.2"
	case SIMDAVX:
		return "AVX"
	case SIMDAVX2:
		return "AVX2"
	case SIMDAVX512:
		return "AVX-512"
	case SIMDNEON:
		return "NEON"
	case SIMDNEONV8:
		return "NEON-v8"
	default:
		return "Unknown"
	}
}

// GetBestSIMDCapability returns the highest available SIMD instruction set
func GetBestSIMDCapability() SIMDCapability {
	features := GetCPUFeatures()

	// Check x86/x86_64 capabilities in order (highest to lowest)
	if features.HasAVX512F && features.HasAVX512DQ && features.HasAVX512BW && features.HasAVX512VL {
		return SIMDAVX512
	}
	if features.HasAVX2 {
		return SIMDAVX2
	}
	if features.HasAVX {
		return SIMDAVX
	}
	if features.HasSSE42 {
		return SIMDSSE42
	}
	if features.HasSSE41 {
		return SIMDSSE41
	}
	if features.HasSSE3 {
		return SIMDSSE3
	}
	if features.HasSSE2 {
		return SIMDSSE2
	}
	if features.HasSSE {
		return SIMDSSE
	}

	// Check ARM capabilities
	if features.HasASIMD || (features.HasNEON && features.HasNEONFP) {
		return SIMDNEONV8
	}
	if features.HasNEON {
		return SIMDNEON
	}

	return SIMDNone
}

// GetVectorWidth returns the optimal vector width in bytes for the best SIMD capability
func GetVectorWidth() int {
	capability := GetBestSIMDCapability()

	switch capability {
	case SIMDAVX512:
		return 64 // 512 bits = 64 bytes
	case SIMDAVX, SIMDAVX2:
		return 32 // 256 bits = 32 bytes
	case SIMDSSE, SIMDSSE2, SIMDSSE3, SIMDSSE41, SIMDSSE42:
		return 16 // 128 bits = 16 bytes
	default:
		return 8 // Fallback to scalar operations
	}
}

// GetElementsPerVector returns number of elements that fit in a SIMD vector for given data type
func GetElementsPerVector(dtype DType) int {
	vectorWidth := GetVectorWidth()

	switch dtype {
	case Float64:
		return vectorWidth / 8 // 8 bytes per float64
	case Float32:
		return vectorWidth / 4 // 4 bytes per float32
	case Int64:
		return vectorWidth / 8 // 8 bytes per int64
	case Int32:
		return vectorWidth / 4 // 4 bytes per int32
	case Int16:
		return vectorWidth / 2 // 2 bytes per int16
	case Int8:
		return vectorWidth / 1 // 1 byte per int8
	case Bool:
		return vectorWidth * 8 // 1 bit per bool (packed)
	default:
		return 1 // Fallback for unsupported types
	}
}

// IsAligned checks if a pointer is aligned to the required boundary for SIMD operations
func IsAligned(ptr unsafe.Pointer, alignment int) bool {
	return uintptr(ptr)%uintptr(alignment) == 0
}

// AlignmentRequirement returns the required memory alignment for optimal SIMD performance
func AlignmentRequirement() int {
	capability := GetBestSIMDCapability()

	switch capability {
	case SIMDAVX512:
		return 64 // AVX-512 prefers 64-byte alignment
	case SIMDAVX, SIMDAVX2:
		return 32 // AVX prefers 32-byte alignment
	case SIMDSSE, SIMDSSE2, SIMDSSE3, SIMDSSE41, SIMDSSE42:
		return 16 // SSE requires 16-byte alignment
	default:
		return 8 // Default alignment
	}
}

// SIMDInfo provides comprehensive information about SIMD capabilities
type SIMDInfo struct {
	BestCapability    SIMDCapability
	VectorWidth       int
	AlignmentRequired int
	Features          *CPUFeatures
}

// GetSIMDInfo returns comprehensive SIMD information
func GetSIMDInfo() *SIMDInfo {
	return &SIMDInfo{
		BestCapability:    GetBestSIMDCapability(),
		VectorWidth:       GetVectorWidth(),
		AlignmentRequired: AlignmentRequirement(),
		Features:          GetCPUFeatures(),
	}
}

// PrintSIMDInfo prints detailed SIMD capability information
func PrintSIMDInfo() {
	info := GetSIMDInfo()
	features := info.Features

	DebugInfo("=== SIMD Capabilities ===")
	DebugInfo("Architecture: %s", features.Architecture)
	DebugInfo("Best SIMD: %s", info.BestCapability.String())
	DebugInfo("Vector Width: %d bytes", info.VectorWidth)
	DebugInfo("Alignment Required: %d bytes", info.AlignmentRequired)
	DebugInfo("Available Features:")
	DebugInfo("  SSE: %v, SSE2: %v, SSE3: %v", features.HasSSE, features.HasSSE2, features.HasSSE3)
	DebugInfo("  SSSE3: %v, SSE4.1: %v, SSE4.2: %v", features.HasSSSE3, features.HasSSE41, features.HasSSE42)
	DebugInfo("  AVX: %v, AVX2: %v, FMA: %v", features.HasAVX, features.HasAVX2, features.HasFMA)
	DebugInfo("  AVX-512F: %v, AVX-512DQ: %v, AVX-512BW: %v, AVX-512VL: %v",
		features.HasAVX512F, features.HasAVX512DQ, features.HasAVX512BW, features.HasAVX512VL)
	DebugInfo("Elements per vector (float64): %d", GetElementsPerVector(Float64))
	DebugInfo("Elements per vector (float32): %d", GetElementsPerVector(Float32))
	DebugInfo("========================")
}
