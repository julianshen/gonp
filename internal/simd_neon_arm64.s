//go:build arm64

// ARM64 NEON assembly implementations
// These functions provide vectorized operations using ARM64 Advanced SIMD (NEON)

#include "textflag.h"

// addFloat64NEON adds two float64 vectors using NEON
// func addFloat64NEON(a, b, result *float64)
TEXT ·addFloat64NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    // Load two float64 values into vector registers
    VLD1 (R0), [V0.D2]    // Load a[0], a[1] into V0
    VLD1 (R1), [V1.D2]    // Load b[0], b[1] into V1
    
    // Perform vectorized addition
    VFADD V1.D2, V0.D2, V2.D2
    
    // Store result
    VST1 [V2.D2], (R2)
    RET

// subFloat64NEON subtracts two float64 vectors using NEON
// func subFloat64NEON(a, b, result *float64)
TEXT ·subFloat64NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.D2]    // Load a[0], a[1]
    VLD1 (R1), [V1.D2]    // Load b[0], b[1]
    
    // Perform vectorized subtraction
    VFSUB V1.D2, V0.D2, V2.D2
    
    VST1 [V2.D2], (R2)
    RET

// mulFloat64NEON multiplies two float64 vectors using NEON
// func mulFloat64NEON(a, b, result *float64)
TEXT ·mulFloat64NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.D2]
    VLD1 (R1), [V1.D2]
    
    // Perform vectorized multiplication
    VFMUL V1.D2, V0.D2, V2.D2
    
    VST1 [V2.D2], (R2)
    RET

// divFloat64NEON divides two float64 vectors using NEON
// func divFloat64NEON(a, b, result *float64)
TEXT ·divFloat64NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.D2]
    VLD1 (R1), [V1.D2]
    
    // Perform vectorized division
    VFDIV V1.D2, V0.D2, V2.D2
    
    VST1 [V2.D2], (R2)
    RET

// addFloat32NEON adds four float32 vectors using NEON
// func addFloat32NEON(a, b, result *float32)
TEXT ·addFloat32NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.S4]    // Load 4 float32 values
    VLD1 (R1), [V1.S4]
    
    VFADD V1.S4, V0.S4, V2.S4
    
    VST1 [V2.S4], (R2)
    RET

// subFloat32NEON subtracts four float32 vectors using NEON
// func subFloat32NEON(a, b, result *float32)
TEXT ·subFloat32NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.S4]
    VLD1 (R1), [V1.S4]
    
    VFSUB V1.S4, V0.S4, V2.S4
    
    VST1 [V2.S4], (R2)
    RET

// mulFloat32NEON multiplies four float32 vectors using NEON
// func mulFloat32NEON(a, b, result *float32)
TEXT ·mulFloat32NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.S4]
    VLD1 (R1), [V1.S4]
    
    VFMUL V1.S4, V0.S4, V2.S4
    
    VST1 [V2.S4], (R2)
    RET

// divFloat32NEON divides four float32 vectors using NEON
// func divFloat32NEON(a, b, result *float32)
TEXT ·divFloat32NEON(SB), NOSPLIT, $0-24
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    
    VLD1 (R0), [V0.S4]
    VLD1 (R1), [V1.S4]
    
    VFDIV V1.S4, V0.S4, V2.S4
    
    VST1 [V2.S4], (R2)
    RET

// Scalar operations with vectors

// addScalarFloat64NEON adds scalar to float64 vector using NEON
// func addScalarFloat64NEON(a *float64, scalar float64, result *float64)
TEXT ·addScalarFloat64NEON(SB), NOSPLIT, $0-32
    MOVD a+0(FP), R0
    FMOVD scalar+8(FP), F0
    MOVD result+24(FP), R1
    
    VLD1 (R0), [V1.D2]      // Load vector a
    VDUP F0, V2.D2          // Broadcast scalar to vector
    
    VFADD V2.D2, V1.D2, V3.D2
    
    VST1 [V3.D2], (R1)
    RET

// mulScalarFloat64NEON multiplies scalar with float64 vector using NEON
// func mulScalarFloat64NEON(a *float64, scalar float64, result *float64)
TEXT ·mulScalarFloat64NEON(SB), NOSPLIT, $0-32
    MOVD a+0(FP), R0
    FMOVD scalar+8(FP), F0
    MOVD result+24(FP), R1
    
    VLD1 (R0), [V1.D2]
    VDUP F0, V2.D2
    
    VFMUL V2.D2, V1.D2, V3.D2
    
    VST1 [V3.D2], (R1)
    RET

// addScalarFloat32NEON adds scalar to float32 vector using NEON
// func addScalarFloat32NEON(a *float32, scalar float32, result *float32)
TEXT ·addScalarFloat32NEON(SB), NOSPLIT, $0-28
    MOVD a+0(FP), R0
    FMOVS scalar+8(FP), F0
    MOVD result+20(FP), R1
    
    VLD1 (R0), [V1.S4]
    VDUP F0, V2.S4
    
    VFADD V2.S4, V1.S4, V3.S4
    
    VST1 [V3.S4], (R1)
    RET

// mulScalarFloat32NEON multiplies scalar with float32 vector using NEON
// func mulScalarFloat32NEON(a *float32, scalar float32, result *float32)
TEXT ·mulScalarFloat32NEON(SB), NOSPLIT, $0-28
    MOVD a+0(FP), R0
    FMOVS scalar+8(FP), F0
    MOVD result+20(FP), R1
    
    VLD1 (R0), [V1.S4]
    VDUP F0, V2.S4
    
    VFMUL V2.S4, V1.S4, V3.S4
    
    VST1 [V3.S4], (R1)
    RET

// sqrtFloat64NEON computes sqrt of float64 vector using NEON
// func sqrtFloat64NEON(a, result *float64)
TEXT ·sqrtFloat64NEON(SB), NOSPLIT, $0-16
    MOVD a+0(FP), R0
    MOVD result+8(FP), R1
    
    VLD1 (R0), [V0.D2]
    
    VFSQRT V0.D2, V1.D2
    
    VST1 [V1.D2], (R1)
    RET

// sqrtFloat32NEON computes sqrt of float32 vector using NEON
// func sqrtFloat32NEON(a, result *float32)
TEXT ·sqrtFloat32NEON(SB), NOSPLIT, $0-16
    MOVD a+0(FP), R0
    MOVD result+8(FP), R1
    
    VLD1 (R0), [V0.S4]
    
    VFSQRT V0.S4, V1.S4
    
    VST1 [V1.S4], (R1)
    RET

// Statistical operations

// sumFloat64NEON computes sum of float64 slice using NEON
// func sumFloat64NEON(a []float64, n int) float64
TEXT ·sumFloat64NEON(SB), NOSPLIT, $0-40
    MOVD a_base+0(FP), R0
    MOVD n+16(FP), R1
    
    VEOR V0.B16, V0.B16, V0.B16  // Clear accumulator
    MOVD $2, R3                   // Vector size (2 float64s)
    MOVD R1, R2                   // Copy n to R2
    
    // Process pairs of elements
loop_sum64:
    CMP R3, R2
    BLT remainder_sum64
    
    VLD1 (R0), [V1.D2]          // Load 2 elements
    VFADD V1.D2, V0.D2, V0.D2   // Add to accumulator
    
    ADD $16, R0                  // Advance pointer (2 * 8 bytes)
    SUB R3, R2                   // Decrement counter
    B loop_sum64

remainder_sum64:
    // Handle remaining elements
    CBZ R2, finish_sum64
    
    FMOVD (R0), F1
    VFADD V0.D[0], F1, F1
    FMOVD F1, V0.D[0]
    
finish_sum64:
    // Sum the two elements in the accumulator
    VFADD V0.D[1], V0.D[0], F0
    FMOVD F0, ret+32(FP)
    RET

// sumFloat32NEON computes sum of float32 slice using NEON
// func sumFloat32NEON(a []float32, n int) float32
TEXT ·sumFloat32NEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD n+8(FP), R1
    
    VEOR V0.B16, V0.B16, V0.B16  // Clear accumulator
    MOVD $4, R3                   // Vector size (4 float32s)
    MOVD R1, R2
    
loop_sum32:
    CMP R3, R2
    BLT remainder_sum32
    
    VLD1 (R0), [V1.S4]
    VFADD V1.S4, V0.S4, V0.S4
    
    ADD $16, R0                  // Advance pointer (4 * 4 bytes)
    SUB R3, R2
    B loop_sum32

remainder_sum32:
    CBZ R2, finish_sum32
    
    FMOVS (R0), F1
    VFADD V0.S[0], F1, F1
    FMOVS F1, V0.S[0]
    
finish_sum32:
    // Sum all 4 elements in the accumulator
    VADDP V0.S4, V0.S4, V1.S4    // Pairwise add
    VADDP V1.S4, V1.S4, V2.S4
    FMOVS V2.S[0], F0
    FMOVS F0, ret+24(FP)
    RET

// dotProductFloat64NEON computes dot product using NEON
// func dotProductFloat64NEON(a, b []float64, n int) float64
TEXT ·dotProductFloat64NEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD b_base+24(FP), R1
    MOVD n+40(FP), R2
    
    VEOR V0.B16, V0.B16, V0.B16  // Clear accumulator
    MOVD $2, R3                   // Vector size
    
loop_dot64:
    CMP R3, R2
    BLT remainder_dot64
    
    VLD1 (R0), [V1.D2]          // Load a
    VLD1 (R1), [V2.D2]          // Load b
    
    VFMLA V2.D2, V1.D2, V0.D2   // Multiply-accumulate
    
    ADD $16, R0
    ADD $16, R1
    SUB R3, R2
    B loop_dot64

remainder_dot64:
    CBZ R2, finish_dot64
    
    FMOVD (R0), F1
    FMOVD (R1), F2
    VFMLA F2, F1, V0.D[0]
    
finish_dot64:
    // Sum the accumulator
    VFADD V0.D[1], V0.D[0], F0
    FMOVD F0, ret+48(FP)
    RET

// varianceFloat64NEON computes variance using NEON
// func varianceFloat64NEON(a []float64, mean float64, n int) float64
TEXT ·varianceFloat64NEON(SB), NOSPLIT, $0-48
    MOVD a_base+0(FP), R0
    FMOVD mean+24(FP), F1
    MOVD n+32(FP), R2
    
    VEOR V0.B16, V0.B16, V0.B16  // Clear accumulator
    VDUP F1, V3.D2               // Broadcast mean
    MOVD $2, R3                   // Vector size
    
loop_var64:
    CMP R3, R2
    BLT remainder_var64
    
    VLD1 (R0), [V1.D2]          // Load data
    VFSUB V3.D2, V1.D2, V2.D2   // Subtract mean
    VFMLA V2.D2, V2.D2, V0.D2   // Square and accumulate
    
    ADD $16, R0
    SUB R3, R2
    B loop_var64

remainder_var64:
    CBZ R2, finish_var64
    
    FMOVD (R0), F2
    VFSUB F1, F2, F2
    VFMLA F2, F2, V0.D[0]
    
finish_var64:
    // Sum accumulator and divide by (n-1)
    VFADD V0.D[1], V0.D[0], F0
    MOVD n+32(FP), R2
    SUB $1, R2
    SCVTF R2, F2
    VFDIV F2, F0, F0
    FMOVD F0, ret+40(FP)
    RET