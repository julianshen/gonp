// +build amd64

#include "textflag.h"

// func avxAddFloat64(a, b, result *float64, n int)
TEXT ·avxAddFloat64(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), AX          // Load pointer to a
    MOVQ b+8(FP), BX          // Load pointer to b  
    MOVQ result+16(FP), CX    // Load pointer to result
    MOVQ n+24(FP), DX         // Load count

    // Check if we have at least 4 elements (256 bits / 64 bits per float64)
    CMPQ DX, $4
    JL scalar_add_float64

    // Calculate number of vector iterations (n / 4)
    MOVQ DX, R8
    SHRQ $2, R8              // R8 = n / 4
    
    // Vector loop
vector_loop_add_float64:
    TESTQ R8, R8
    JZ remainder_add_float64
    
    // Load 4 float64 values from a and b
    VMOVUPD (AX), Y0         // Load a[i:i+4] into Y0
    VMOVUPD (BX), Y1         // Load b[i:i+4] into Y1
    
    // Perform addition
    VADDPD Y1, Y0, Y2        // Y2 = Y0 + Y1
    
    // Store result
    VMOVUPD Y2, (CX)         // Store result[i:i+4]
    
    // Advance pointers
    ADDQ $32, AX             // a += 4 * 8 bytes
    ADDQ $32, BX             // b += 4 * 8 bytes  
    ADDQ $32, CX             // result += 4 * 8 bytes
    
    DECQ R8
    JMP vector_loop_add_float64

remainder_add_float64:
    // Handle remaining elements (n % 4)
    ANDQ $3, DX              // DX = n % 4
    
scalar_add_float64:
    TESTQ DX, DX
    JZ done_add_float64
    
scalar_loop_add_float64:
    // Scalar addition for remaining elements
    MOVSD (AX), X0           // Load a[i]
    MOVSD (BX), X1           // Load b[i]
    ADDSD X1, X0             // X0 = a[i] + b[i]
    MOVSD X0, (CX)           // Store result[i]
    
    ADDQ $8, AX              // a++
    ADDQ $8, BX              // b++
    ADDQ $8, CX              // result++
    
    DECQ DX
    JNZ scalar_loop_add_float64

done_add_float64:
    VZEROUPPER               // Clear upper bits of YMM registers
    RET

// func avxMulFloat64(a, b, result *float64, n int)
TEXT ·avxMulFloat64(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), AX          // Load pointer to a
    MOVQ b+8(FP), BX          // Load pointer to b
    MOVQ result+16(FP), CX    // Load pointer to result
    MOVQ n+24(FP), DX         // Load count

    // Check if we have at least 4 elements
    CMPQ DX, $4
    JL scalar_mul_float64

    // Calculate number of vector iterations
    MOVQ DX, R8
    SHRQ $2, R8              // R8 = n / 4
    
vector_loop_mul_float64:
    TESTQ R8, R8
    JZ remainder_mul_float64
    
    // Load 4 float64 values
    VMOVUPD (AX), Y0         // Load a[i:i+4]
    VMOVUPD (BX), Y1         // Load b[i:i+4]
    
    // Perform multiplication
    VMULPD Y1, Y0, Y2        // Y2 = Y0 * Y1
    
    // Store result
    VMOVUPD Y2, (CX)         // Store result[i:i+4]
    
    // Advance pointers
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    
    DECQ R8
    JMP vector_loop_mul_float64

remainder_mul_float64:
    // Handle remaining elements
    ANDQ $3, DX
    
scalar_mul_float64:
    TESTQ DX, DX
    JZ done_mul_float64
    
scalar_loop_mul_float64:
    MOVSD (AX), X0           // Load a[i]
    MOVSD (BX), X1           // Load b[i]
    MULSD X1, X0             // X0 = a[i] * b[i]
    MOVSD X0, (CX)           // Store result[i]
    
    ADDQ $8, AX
    ADDQ $8, BX
    ADDQ $8, CX
    
    DECQ DX
    JNZ scalar_loop_mul_float64

done_mul_float64:
    VZEROUPPER
    RET

// func avxSumFloat64(a *float64, n int) float64
TEXT ·avxSumFloat64(SB), NOSPLIT, $0-24
    MOVQ a+0(FP), AX         // Load pointer to a
    MOVQ n+8(FP), DX         // Load count
    
    // Initialize sum vector to zero
    VXORPD Y0, Y0, Y0        // Y0 = 0 (4 x float64)
    
    // Check if we have at least 4 elements
    CMPQ DX, $4
    JL scalar_sum_float64

    // Calculate number of vector iterations
    MOVQ DX, R8
    SHRQ $2, R8              // R8 = n / 4
    
vector_loop_sum_float64:
    TESTQ R8, R8
    JZ remainder_sum_float64
    
    // Load 4 float64 values and accumulate
    VMOVUPD (AX), Y1         // Load a[i:i+4]
    VADDPD Y1, Y0, Y0        // Y0 += Y1
    
    ADDQ $32, AX             // a += 4 * 8 bytes
    DECQ R8
    JMP vector_loop_sum_float64

remainder_sum_float64:
    // Handle remaining elements
    ANDQ $3, DX
    
    // Horizontal add of Y0 to get final sum
    VEXTRACTF128 $1, Y0, X1  // Extract upper 128 bits
    VADDPD X1, X0, X0        // Add upper and lower parts
    VHADDPD X0, X0, X0       // Horizontal add within 128 bits
    
scalar_sum_float64:
    TESTQ DX, DX
    JZ done_sum_float64
    
scalar_loop_sum_float64:
    MOVSD (AX), X1           // Load a[i]
    ADDSD X1, X0             // X0 += a[i]
    
    ADDQ $8, AX
    DECQ DX
    JNZ scalar_loop_sum_float64

done_sum_float64:
    VZEROUPPER
    MOVSD X0, ret+16(FP)     // Store result
    RET

// func avxAddScalarFloat64(a *float64, scalar float64, result *float64, n int)
TEXT ·avxAddScalarFloat64(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), AX         // Load pointer to a
    MOVSD scalar+8(FP), X0   // Load scalar
    MOVQ result+16(FP), CX   // Load pointer to result
    MOVQ n+24(FP), DX        // Load count
    
    // Broadcast scalar to all elements in YMM register
    VBROADCASTSD X0, Y0      // Y0 = [scalar, scalar, scalar, scalar]
    
    // Check if we have at least 4 elements
    CMPQ DX, $4
    JL scalar_add_scalar_float64

    // Calculate number of vector iterations
    MOVQ DX, R8
    SHRQ $2, R8              // R8 = n / 4
    
vector_loop_add_scalar_float64:
    TESTQ R8, R8
    JZ remainder_add_scalar_float64
    
    // Load 4 float64 values and add scalar
    VMOVUPD (AX), Y1         // Load a[i:i+4]
    VADDPD Y0, Y1, Y2        // Y2 = Y1 + Y0 (scalar)
    VMOVUPD Y2, (CX)         // Store result[i:i+4]
    
    ADDQ $32, AX             // a += 4 * 8 bytes
    ADDQ $32, CX             // result += 4 * 8 bytes
    
    DECQ R8
    JMP vector_loop_add_scalar_float64

remainder_add_scalar_float64:
    // Handle remaining elements
    ANDQ $3, DX
    
scalar_add_scalar_float64:
    TESTQ DX, DX
    JZ done_add_scalar_float64
    
scalar_loop_add_scalar_float64:
    MOVSD (AX), X1           // Load a[i]
    ADDSD X0, X1             // X1 = a[i] + scalar
    MOVSD X1, (CX)           // Store result[i]
    
    ADDQ $8, AX
    ADDQ $8, CX
    
    DECQ DX
    JNZ scalar_loop_add_scalar_float64

done_add_scalar_float64:
    VZEROUPPER
    RET

// Advanced mathematical functions

// func avxSqrtFloat64(a, result *float64, n int)
TEXT ·avxSqrtFloat64(SB), NOSPLIT, $0-24
    MOVQ a+0(FP), AX         // Load pointer to a
    MOVQ result+8(FP), CX    // Load pointer to result
    MOVQ n+16(FP), DX        // Load count

    // Check if we have at least 4 elements
    CMPQ DX, $4
    JL scalar_sqrt_float64

    // Calculate number of vector iterations
    MOVQ DX, R8
    SHRQ $2, R8              // R8 = n / 4
    
vector_loop_sqrt_float64:
    TESTQ R8, R8
    JZ remainder_sqrt_float64
    
    // Load 4 float64 values
    VMOVUPD (AX), Y0         // Load a[i:i+4]
    
    // Perform square root (AVX has native sqrt)
    VSQRTPD Y0, Y1           // Y1 = sqrt(Y0)
    
    // Store result
    VMOVUPD Y1, (CX)         // Store result[i:i+4]
    
    // Advance pointers
    ADDQ $32, AX             // a += 4 * 8 bytes
    ADDQ $32, CX             // result += 4 * 8 bytes
    
    DECQ R8
    JMP vector_loop_sqrt_float64

remainder_sqrt_float64:
    // Handle remaining elements
    ANDQ $3, DX
    
scalar_sqrt_float64:
    TESTQ DX, DX
    JZ done_sqrt_float64
    
scalar_loop_sqrt_float64:
    MOVSD (AX), X0           // Load a[i]
    SQRTSD X0, X1            // X1 = sqrt(a[i])
    MOVSD X1, (CX)           // Store result[i]
    
    ADDQ $8, AX
    ADDQ $8, CX
    
    DECQ DX
    JNZ scalar_loop_sqrt_float64

done_sqrt_float64:
    VZEROUPPER
    RET

// func avxDotProductFloat64(a, b *float64, n int) float64
TEXT ·avxDotProductFloat64(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), AX         // Load pointer to a
    MOVQ b+8(FP), BX         // Load pointer to b
    MOVQ n+16(FP), DX        // Load count
    
    // Initialize accumulator to zero
    VXORPD Y0, Y0, Y0        // Y0 = 0 (4 x float64)
    
    // Check if we have at least 4 elements
    CMPQ DX, $4
    JL scalar_dot_float64

    // Calculate number of vector iterations
    MOVQ DX, R8
    SHRQ $2, R8              // R8 = n / 4
    
vector_loop_dot_float64:
    TESTQ R8, R8
    JZ remainder_dot_float64
    
    // Load 4 float64 values from a and b
    VMOVUPD (AX), Y1         // Load a[i:i+4]
    VMOVUPD (BX), Y2         // Load b[i:i+4]
    
    // Multiply and accumulate
    VMULPD Y2, Y1, Y3        // Y3 = Y1 * Y2
    VADDPD Y3, Y0, Y0        // Y0 += Y3
    
    // Advance pointers
    ADDQ $32, AX             // a += 4 * 8 bytes
    ADDQ $32, BX             // b += 4 * 8 bytes
    
    DECQ R8
    JMP vector_loop_dot_float64

remainder_dot_float64:
    // Handle remaining elements
    ANDQ $3, DX
    
    // Horizontal add of Y0 to get final sum
    VEXTRACTF128 $1, Y0, X1  // Extract upper 128 bits
    VADDPD X1, X0, X0        // Add upper and lower parts
    VHADDPD X0, X0, X0       // Horizontal add within 128 bits
    
scalar_dot_float64:
    TESTQ DX, DX
    JZ done_dot_float64
    
scalar_loop_dot_float64:
    MOVSD (AX), X1           // Load a[i]
    MOVSD (BX), X2           // Load b[i]
    MULSD X2, X1             // X1 = a[i] * b[i]
    ADDSD X1, X0             // X0 += a[i] * b[i]
    
    ADDQ $8, AX
    ADDQ $8, BX
    
    DECQ DX
    JNZ scalar_loop_dot_float64

done_dot_float64:
    VZEROUPPER
    MOVSD X0, ret+24(FP)     // Store result
    RET
