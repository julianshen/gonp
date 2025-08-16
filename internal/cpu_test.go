package internal

import (
	"testing"
)

func TestCPUFeatureDetection(t *testing.T) {
	PrintSIMDInfo()

	info := GetSIMDInfo()
	if info == nil {
		t.Error("GetSIMDInfo returned nil")
		return
	}

	t.Logf("Best SIMD capability: %s", info.BestCapability.String())
	t.Logf("Vector width: %d bytes", info.VectorWidth)
	t.Logf("Alignment required: %d bytes", info.AlignmentRequired)

	// Cross-arch determinism: do not assume any particular SIMD is available
	// on the build host. Validate internal consistency instead.
	if info.BestCapability == SIMDNone {
		t.Log("No SIMD capabilities detected on this system; proceeding with scalar fallback checks")
	}

	features := info.Features
	t.Logf("SSE2: %v, AVX: %v, AVX2: %v, AVX-512F: %v",
		features.HasSSE2, features.HasAVX, features.HasAVX2, features.HasAVX512F)

	// Test element counts
	float64Count := GetElementsPerVector(Float64)
	float32Count := GetElementsPerVector(Float32)

	t.Logf("Elements per vector - float64: %d, float32: %d", float64Count, float32Count)

	if float64Count <= 0 || float32Count <= 0 {
		t.Error("Invalid elements per vector count")
	}
}

func TestSIMDCapabilityString(t *testing.T) {
	tests := []struct {
		cap      SIMDCapability
		expected string
	}{
		{SIMDNone, "None"},
		{SIMDSSE, "SSE"},
		{SIMDSSE2, "SSE2"},
		{SIMDAVX, "AVX"},
		{SIMDAVX2, "AVX2"},
		{SIMDAVX512, "AVX-512"},
		{SIMDNEON, "NEON"},
		{SIMDNEONV8, "NEON-v8"},
	}

	for _, tt := range tests {
		result := tt.cap.String()
		if result != tt.expected {
			t.Errorf("SIMDCapability(%d).String() = %s, expected %s", tt.cap, result, tt.expected)
		}
	}
}

func TestVectorWidthCalculation(t *testing.T) {
	width := GetVectorWidth()
	t.Logf("Vector width: %d bytes", width)

	// Should be one of the expected values
	validWidths := []int{8, 16, 32, 64}
	found := false
	for _, validWidth := range validWidths {
		if width == validWidth {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Unexpected vector width: %d", width)
	}
}
