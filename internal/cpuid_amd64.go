//go:build amd64
// +build amd64

package internal

// cpuid executes the CPUID instruction with the given input values
func cpuid(eaxIn, ecxIn uint32) uint32

// cpuidFunc executes CPUID and returns all register values
func cpuidFunc(eaxIn uint32) (eax, ebx, ecx, edx uint32)

// cpuidFunc7 executes CPUID function 7 (Extended Features)
func cpuidFunc7(ecxIn uint32) (eax, ebx, ecx, edx uint32)
