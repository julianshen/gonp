//go:build !amd64
// +build !amd64

package internal

// cpuid fallback for non-x86 architectures
func cpuid(eaxIn, ecxIn uint32) uint32 {
	return 0
}

// cpuidFunc fallback for non-x86 architectures
func cpuidFunc(eaxIn uint32) (eax, ebx, ecx, edx uint32) {
	return 0, 0, 0, 0
}

// cpuidFunc7 fallback for non-x86 architectures
func cpuidFunc7(ecxIn uint32) (eax, ebx, ecx, edx uint32) {
	return 0, 0, 0, 0
}
