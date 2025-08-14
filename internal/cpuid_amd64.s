// +build amd64

#include "textflag.h"

// func cpuid(eaxIn, ecxIn uint32) uint32
TEXT ·cpuid(SB), NOSPLIT, $0-12
	MOVL eaxIn+0(FP), AX
	MOVL ecxIn+4(FP), CX
	CPUID
	MOVL AX, ret+8(FP)
	RET

// func cpuidFunc(eaxIn uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuidFunc(SB), NOSPLIT, $0-24
	MOVL eaxIn+0(FP), AX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func cpuidFunc7(ecxIn uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuidFunc7(SB), NOSPLIT, $0-24
	MOVL $7, AX
	MOVL ecxIn+0(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET
