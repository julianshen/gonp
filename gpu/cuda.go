// Package gpu provides CUDA hardware integration
//
// This module implements actual CUDA device detection and initialization
// using cgo bindings to the CUDA Driver API.
//
// Build Requirements:
//   - CUDA Toolkit installed on the system
//   - CGO_ENABLED=1
//   - Proper CUDA library paths
//
// Usage:
//   devices, err := DetectCUDADevices()
//   if err == nil {
//       device := devices[0]
//       // Use CUDA device for computations
//   }
//
//go:build cuda
// +build cuda

package gpu

/*
#cgo LDFLAGS: -lcuda
#include <cuda.h>
#include <stdlib.h>

// Error string helpers
const char* cuErrName(CUresult r) {
    const char* n = NULL;
    cuGetErrorName(r, &n);
    return n ? n : "UNKNOWN";
}
const char* cuErrStr(CUresult r) {
    const char* s = NULL;
    cuGetErrorString(r, &s);
    return s ? s : "";
}

// Primary context helpers per device
int retainSetPrimaryCtx(int device, CUcontext* out) {
    CUdevice dev;
    CUcontext ctx;
    CUresult res = cuDeviceGet(&dev, device);
    if (res != CUDA_SUCCESS) return (int)res;
    res = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (res != CUDA_SUCCESS) return (int)res;
    res = cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) { cuDevicePrimaryCtxRelease(dev); return (int)res; }
    if (out) *out = ctx;
    return 0;
}
int releasePrimaryCtx(int device) {
    CUdevice dev;
    CUresult res = cuDeviceGet(&dev, device);
    if (res != CUDA_SUCCESS) return (int)res;
    cuCtxSetCurrent(NULL);
    return (int)cuDevicePrimaryCtxRelease(dev);
}

// Helper function to get device name
int getCudaDeviceName(int device, char* name, int len) {
    CUdevice cuDevice;
    CUresult result = cuDeviceGet(&cuDevice, device);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    result = cuDeviceGetName(name, len, cuDevice);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    return 0;
}

// Helper function to get device memory
int getCudaDeviceMemory(int device, size_t* bytes) {
    CUdevice cuDevice;
    CUresult result = cuDeviceGet(&cuDevice, device);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    result = cuDeviceTotalMem(bytes, cuDevice);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    return 0;
}

// Helper function to get compute capability
int getCudaDeviceCapability(int device, int* major, int* minor) {
    CUdevice cuDevice;
    CUresult result = cuDeviceGet(&cuDevice, device);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    result = cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    result = cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
    if (result != CUDA_SUCCESS) {
        return -1;
    }

    return 0;
}
*/
import "C"

import (
    "fmt"
    "runtime"
    "sync"
    "unsafe"
)

// CUDADevice represents a CUDA GPU device
type CUDADevice struct {
    deviceID int
    name     string
    memory   int64
    major    int
    minor    int
}

// Manage primary contexts per device to avoid retain/release overhead
type cudaDeviceCtx struct {
    dev     C.CUdevice
    ctx     C.CUcontext
    retained bool
    mod     C.CUmodule
    fnVAdd  C.CUfunction
    kernelLoaded bool
    modSum  C.CUmodule
    fnSum   C.CUfunction
    kernelLoadedSum bool
    modMat C.CUmodule
    fnMat  C.CUfunction
    kernelLoadedMat bool
    modReduce C.CUmodule
    fnReduce  C.CUfunction
    kernelLoadedReduce bool
    modTree C.CUmodule
    fnTree  C.CUfunction
    kernelLoadedTree bool
}

var cudaCtxRegistry = struct {
    mu sync.Mutex
    m  map[int]*cudaDeviceCtx
}{m: make(map[int]*cudaDeviceCtx)}

func ensurePrimaryContext(deviceID int) error {
    cudaCtxRegistry.mu.Lock()
    entry, ok := cudaCtxRegistry.m[deviceID]
    if !ok {
        entry = &cudaDeviceCtx{}
        cudaCtxRegistry.m[deviceID] = entry
    }
    // If not retained yet, retain the primary context once
    if !entry.retained {
        var dev C.CUdevice
        var ctx C.CUcontext
        if res := C.cuDeviceGet(&dev, C.int(deviceID)); res != C.CUDA_SUCCESS {
            cudaCtxRegistry.mu.Unlock()
            return NewGPUError(fmt.Sprintf("cuDeviceGet failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorDeviceNotFound)
        }
        if res := C.cuDevicePrimaryCtxRetain(&ctx, dev); res != C.CUDA_SUCCESS {
            cudaCtxRegistry.mu.Unlock()
            return NewGPUError(fmt.Sprintf("cuDevicePrimaryCtxRetain failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorDeviceNotFound)
        }
        entry.dev = dev
        entry.ctx = ctx
        entry.retained = true
    }
    ctx := entry.ctx
    cudaCtxRegistry.mu.Unlock()

    // Set current for this thread
    if res := C.cuCtxSetCurrent(ctx); res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuCtxSetCurrent failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return nil
}

// PTX for double-precision vector add with grid-stride loop
const vectorAddPTX = `
.version 7.0
.target sm_50
.address_size 64

.visible .entry vadd_f64(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 N
) {
    .reg .pred p;
    .reg .b32 rIdx, rN, rBD, rGD, rStride, rTmp;
    .reg .b64 rdA, rdB, rdC, rdOff, rdAp, rdBp, rdCp;
    .reg .f64 fA, fB, fC;

    ld.param.u64 rdA, [A];
    ld.param.u64 rdB, [B];
    ld.param.u64 rdC, [C];
    ld.param.u32 rN, [N];

    mov.u32 rTmp, %ntid.x;
    mov.u32 rBD, %ctaid.x;
    mov.u32 rIdx, %tid.x;
    mad.lo.s32 rIdx, rBD, rTmp, rIdx;

    mov.u32 rGD, %nctaid.x;
    mul.lo.s32 rStride, rGD, rTmp;

LOOP:
    setp.ge.u32 p, rIdx, rN;
    @p bra DONE;

    mul.wide.u32 rdOff, rIdx, 8;
    add.s64 rdAp, rdA, rdOff;
    add.s64 rdBp, rdB, rdOff;
    add.s64 rdCp, rdC, rdOff;

    ld.global.f64 fA, [rdAp];
    ld.global.f64 fB, [rdBp];
    add.f64 fC, fA, fB;
    st.global.f64 [rdCp], fC;

    add.s32 rIdx, rIdx, rStride;
    bra LOOP;

DONE:
    ret;
}`

func ensureVectorAddKernel(deviceID int) (C.CUfunction, error) {
    // Ensure context retained and current
    if err := ensurePrimaryContext(deviceID); err != nil {
        return nil, err
    }
    // Load or reuse module/function
    cudaCtxRegistry.mu.Lock()
    entry := cudaCtxRegistry.m[deviceID]
    if entry.kernelLoaded {
        fn := entry.fnVAdd
        cudaCtxRegistry.mu.Unlock()
        return fn, nil
    }
    cudaCtxRegistry.mu.Unlock()

    cstr := C.CString(vectorAddPTX)
    defer C.free(unsafe.Pointer(cstr))

    var mod C.CUmodule
    if res := C.cuModuleLoadData(&mod, unsafe.Pointer(cstr)); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleLoadData failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    var fn C.CUfunction
    cname := C.CString("vadd_f64")
    defer C.free(unsafe.Pointer(cname))
    if res := C.cuModuleGetFunction(&fn, mod, cname); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleGetFunction failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    cudaCtxRegistry.mu.Lock()
    entry = cudaCtxRegistry.m[deviceID]
    entry.mod = mod
    entry.fnVAdd = fn
    entry.kernelLoaded = true
    cudaCtxRegistry.mu.Unlock()
    return fn, nil
}

// PTX for partial sum: each thread produces one partial sum into Out[global_tid]
const sumPartialsPTX = `
.version 7.0
.target sm_50
.address_size 64

.visible .entry sum_partials_f64(
    .param .u64 In,
    .param .u64 Out,
    .param .u32 N
) {
    .reg .pred p;
    .reg .b32 rIdx, rN, rBDX, rBDY, rBDZ, rTIX, rTIY, rTIZ, rGDX, rGDY, rGDZ, rTmp, rStride;
    .reg .b64 rdIn, rdOut, rdOff, rdPtr;
    .reg .f64 fSum, fVal;

    ld.param.u64 rdIn, [In];
    ld.param.u64 rdOut, [Out];
    ld.param.u32 rN, [N];

    mov.u32 rTmp, %ntid.x;     // blockDim.x
    mov.u32 rBDX, %ctaid.x;    // blockIdx.x
    mov.u32 rTIX, %tid.x;      // threadIdx.x
    mad.lo.s32 rIdx, rBDX, rTmp, rTIX; // global thread index in 1D grid

    mov.u32 rGDX, %nctaid.x;   // gridDim.x
    mul.lo.s32 rStride, rGDX, rTmp; // total threads

    mov.f64 fSum, 0d0;

LOOP_S:
    setp.ge.u32 p, rIdx, rN;
    @p bra DONE_S;
    mul.wide.u32 rdOff, rIdx, 8;
    add.s64 rdPtr, rdIn, rdOff;
    ld.global.f64 fVal, [rdPtr];
    add.f64 fSum, fSum, fVal;
    add.s32 rIdx, rIdx, rStride;
    bra LOOP_S;

DONE_S:
    // write partial to Out[global_tid_original]
    // recompute original global tid: blockIdx.x*blockDim.x + threadIdx.x
    mov.u32 rTmp, %ntid.x;
    mov.u32 rBDX, %ctaid.x;
    mov.u32 rTIX, %tid.x;
    mad.lo.s32 rIdx, rBDX, rTmp, rTIX;
    mul.wide.u32 rdOff, rIdx, 8;
    add.s64 rdPtr, rdOut, rdOff;
    st.global.f64 [rdPtr], fSum;
    ret;
}`

func ensureSumKernel(deviceID int) (C.CUfunction, error) {
    if err := ensurePrimaryContext(deviceID); err != nil { return nil, err }
    cudaCtxRegistry.mu.Lock()
    entry := cudaCtxRegistry.m[deviceID]
    if entry.kernelLoadedSum {
        fn := entry.fnSum
        cudaCtxRegistry.mu.Unlock()
        return fn, nil
    }
    cudaCtxRegistry.mu.Unlock()

    cstr := C.CString(sumPartialsPTX)
    defer C.free(unsafe.Pointer(cstr))
    var mod C.CUmodule
    if res := C.cuModuleLoadData(&mod, unsafe.Pointer(cstr)); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleLoadData(sum) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    var fn C.CUfunction
    cname := C.CString("sum_partials_f64")
    defer C.free(unsafe.Pointer(cname))
    if res := C.cuModuleGetFunction(&fn, mod, cname); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleGetFunction(sum) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    cudaCtxRegistry.mu.Lock()
    entry = cudaCtxRegistry.m[deviceID]
    entry.modSum = mod
    entry.fnSum = fn
    entry.kernelLoadedSum = true
    cudaCtxRegistry.mu.Unlock()
    return fn, nil
}

// PTX for single-block final reduction of doubles from In[L] to Out[1]
// Assumes launch with threads <= 256; uses 2048B shared memory
const finalReducePTX = `
.version 7.0
.target sm_50
.address_size 64

.shared .align 8 .b8 sdata[2048];

.visible .entry final_reduce_f64(
    .param .u64 In,
    .param .u64 Out,
    .param .u32 L
) {
    .reg .pred p, p2;
    .reg .b32 rTid, rBD, rIdx, rL, rStride;
    .reg .b64 rdIn, rdOut, rdOff, rdPtr, rdS;
    .reg .f64 fSum, fTmp;

    ld.param.u64 rdIn, [In];
    ld.param.u64 rdOut, [Out];
    ld.param.u32 rL, [L];

    mov.u32 rTid, %tid.x;
    mov.u32 rBD, %ntid.x; // threads per block

    // grid-stride accumulation (single block so stride==blockDim)
    mov.u32 rIdx, rTid;
    mov.f64 fSum, 0d0;
LOOP_ACC:
    setp.ge.u32 p, rIdx, rL;
    @p bra END_ACC;
    mul.wide.u32 rdOff, rIdx, 8;
    add.s64 rdPtr, rdIn, rdOff;
    ld.global.f64 fTmp, [rdPtr];
    add.f64 fSum, fSum, fTmp;
    add.s32 rIdx, rIdx, rBD;
    bra LOOP_ACC;
END_ACC:

    // write to shared
    mul.wide.u32 rdOff, rTid, 8;
    cvta.to.shared.u64 rdS, sdata;
    add.s64 rdS, rdS, rdOff;
    st.shared.f64 [rdS], fSum;

    // reduction in shared memory: s = 128,64,32,16,8,4,2,1
    bar.sync 0;

    // s=128
    setp.lt.u32 p, rTid, 128;
    @p {
        // sdata[tid] += sdata[tid+128]
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 1024; // 128*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=64
    setp.lt.u32 p, rTid, 64;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 512; // 64*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=32
    setp.lt.u32 p, rTid, 32;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 256; // 32*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=16
    setp.lt.u32 p, rTid, 16;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 128; // 16*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=8
    setp.lt.u32 p, rTid, 8;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 64; // 8*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=4
    setp.lt.u32 p, rTid, 4;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 32; // 4*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=2
    setp.lt.u32 p, rTid, 2;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 16; // 2*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;

    // s=1
    setp.eq.u32 p, rTid, 0;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 8; // 1*8
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.global.f64 [rdOut], fSum;
    }
    ret;
}`

func ensureFinalReduceKernel(deviceID int) (C.CUfunction, error) {
    if err := ensurePrimaryContext(deviceID); err != nil { return nil, err }
    cudaCtxRegistry.mu.Lock()
    entry := cudaCtxRegistry.m[deviceID]
    if entry.kernelLoadedReduce {
        fn := entry.fnReduce
        cudaCtxRegistry.mu.Unlock()
        return fn, nil
    }
    cudaCtxRegistry.mu.Unlock()
    cstr := C.CString(finalReducePTX)
    defer C.free(unsafe.Pointer(cstr))
    var mod C.CUmodule
    if res := C.cuModuleLoadData(&mod, unsafe.Pointer(cstr)); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleLoadData(reduce) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    var fn C.CUfunction
    cname := C.CString("final_reduce_f64")
    defer C.free(unsafe.Pointer(cname))
    if res := C.cuModuleGetFunction(&fn, mod, cname); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleGetFunction(reduce) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    cudaCtxRegistry.mu.Lock()
    entry = cudaCtxRegistry.m[deviceID]
    entry.modReduce = mod
    entry.fnReduce = fn
    entry.kernelLoadedReduce = true
    cudaCtxRegistry.mu.Unlock()
    return fn, nil
}

// PTX for multi-block tree reduction stage: each block reduces a chunk to one value
// Grid-stride across input; shared-memory intra-block reduction; writes Out[blockIdx.x]
const treeReducePTX = `
.version 7.0
.target sm_50
.address_size 64

.shared .align 8 .b8 sdata[2048]; // 256 doubles

.visible .entry tree_reduce_stage_f64(
    .param .u64 In,
    .param .u64 Out,
    .param .u32 L
) {
    .reg .pred p;
    .reg .b32 rTid, rBD, rGD, rIdx, rL;
    .reg .b64 rdIn, rdOut, rdOff, rdPtr, rdS, rdOutPtr;
    .reg .f64 fSum, fTmp;

    ld.param.u64 rdIn, [In];
    ld.param.u64 rdOut, [Out];
    ld.param.u32 rL, [L];

    mov.u32 rTid, %tid.x;       // 0..255
    mov.u32 rBD, %ntid.x;       // 256
    mov.u32 rGD, %nctaid.x;     // gridDim.x

    // global thread id across grid: blockIdx.x*blockDim.x + threadIdx.x
    .reg .b32 rBlk;
    mov.u32 rBlk, %ctaid.x;
    mad.lo.s32 rIdx, rBlk, rBD, rTid;

    // grid-stride to accumulate
    mov.f64 fSum, 0d0;
LOOP_ACC:
    setp.ge.u32 p, rIdx, rL;
    @p bra END_ACC;
    mul.wide.u32 rdOff, rIdx, 8;
    add.s64 rdPtr, rdIn, rdOff;
    ld.global.f64 fTmp, [rdPtr];
    add.f64 fSum, fSum, fTmp;
    mad.lo.s32 rIdx, rGD, rBD, rIdx; // rIdx += gridDim*blockDim
    bra LOOP_ACC;
END_ACC:

    // write partial to shared
    mul.wide.u32 rdOff, rTid, 8;
    cvta.to.shared.u64 rdS, sdata;
    add.s64 rdS, rdS, rdOff;
    st.shared.f64 [rdS], fSum;
    bar.sync 0;

    // reduce in shared: 128..1 (same pattern as final)
    // s=128
    setp.lt.u32 p, rTid, 128;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 1024;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=64
    setp.lt.u32 p, rTid, 64;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 512;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=32
    setp.lt.u32 p, rTid, 32;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 256;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=16
    setp.lt.u32 p, rTid, 16;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 128;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=8
    setp.lt.u32 p, rTid, 8;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 64;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=4
    setp.lt.u32 p, rTid, 4;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 32;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=2
    setp.lt.u32 p, rTid, 2;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 16;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        st.shared.f64 [rdS], fSum;
    }
    bar.sync 0;
    // s=1 -> write block result
    setp.eq.u32 p, rTid, 0;
    @p {
        ld.shared.f64 fSum, [rdS];
        add.s64 rdOff, rdS, 8;
        ld.shared.f64 fTmp, [rdOff];
        add.f64 fSum, fSum, fTmp;
        // Out[blockIdx.x] = fSum
        .reg .b32 rBlkIdx;
        mov.u32 rBlkIdx, %ctaid.x;
        mul.wide.u32 rdOff, rBlkIdx, 8;
        add.s64 rdOutPtr, rdOut, rdOff;
        st.global.f64 [rdOutPtr], fSum;
    }
    ret;
}`

func ensureTreeReduceKernel(deviceID int) (C.CUfunction, error) {
    if err := ensurePrimaryContext(deviceID); err != nil { return nil, err }
    cudaCtxRegistry.mu.Lock()
    entry := cudaCtxRegistry.m[deviceID]
    if entry.kernelLoadedTree {
        fn := entry.fnTree
        cudaCtxRegistry.mu.Unlock()
        return fn, nil
    }
    cudaCtxRegistry.mu.Unlock()
    cstr := C.CString(treeReducePTX)
    defer C.free(unsafe.Pointer(cstr))
    var mod C.CUmodule
    if res := C.cuModuleLoadData(&mod, unsafe.Pointer(cstr)); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleLoadData(tree) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    var fn C.CUfunction
    cname := C.CString("tree_reduce_stage_f64")
    defer C.free(unsafe.Pointer(cname))
    if res := C.cuModuleGetFunction(&fn, mod, cname); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleGetFunction(tree) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    cudaCtxRegistry.mu.Lock()
    entry = cudaCtxRegistry.m[deviceID]
    entry.modTree = mod
    entry.fnTree = fn
    entry.kernelLoadedTree = true
    cudaCtxRegistry.mu.Unlock()
    return fn, nil
}

// PTX for tiled row-major double MatMul: C[M,N] = A[M,K]*B[K,N]
// 16x16 tiles in shared memory for A and B
const matMulPTX = `
.version 7.0
.target sm_50
.address_size 64

.shared .align 8 .b8 sA[2048]; // 16*16*8
.shared .align 8 .b8 sB[2048]; // 16*16*8

.visible .entry matmul_tiled_f64(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 M,
    .param .u32 N,
    .param .u32 K
) {
    .reg .pred pOOB, pLoop, pLdA, pLdB;
    .reg .b32 rM, rN, rK, rI, rJ, rTx, rTy, rBx, rBy, rBDx, rBDy, rT, rTEnd;
    .reg .b64 rdA, rdB, rdC, rdAp, rdBp, rdCp, rdOff, rdSA, rdSB;
    .reg .f64 fSum, fA, fB;

    ld.param.u64 rdA, [A];
    ld.param.u64 rdB, [B];
    ld.param.u64 rdC, [C];
    ld.param.u32 rM, [M];
    ld.param.u32 rN, [N];
    ld.param.u32 rK, [K];

    mov.u32 rTx, %tid.x;
    mov.u32 rTy, %tid.y;
    mov.u32 rBx, %ctaid.x;
    mov.u32 rBy, %ctaid.y;
    mov.u32 rBDx, %ntid.x; // 16
    mov.u32 rBDy, %ntid.y; // 16

    mad.lo.s32 rJ, rBx, rBDx, rTx; // col j
    mad.lo.s32 rI, rBy, rBDy, rTy; // row i

    // Initialize sum
    mov.f64 fSum, 0d0;

    // Compute C base address for (i,j)
    mul.wide.u32 rdOff, rI, rN; // i*N
    add.s64 rdOff, rdOff, rJ;   // i*N+j
    mul.wide.u32 rdOff, rdOff, 8; // *8
    add.s64 rdCp, rdC, rdOff;

    // t loop over tiles of K dimension
    mov.u32 rT, 0;
    mov.u32 rTEnd, rK;
TILE_LOOP:
    setp.ge.u32 pLoop, rT, rTEnd;
    @pLoop bra TILE_END;

    // Load A tile element A[i, rT + rTx] -> sA[rTy, rTx]
    // Compute global col for A
    .reg .b32 rColA;
    add.s32 rColA, rT, rTx;
    // Bounds check for A
    setp.lt.u32 pLdA, rI, rM;
    @pLdA setp.lt.u32 pLdA, rColA, rK;
    @pLdA {
        mul.wide.u32 rdOff, rI, rK; // i*K
        add.s64 rdOff, rdOff, rColA;
        mul.wide.u32 rdOff, rdOff, 8;
        add.s64 rdAp, rdA, rdOff;
        ld.global.f64 fA, [rdAp];
    }
    @!pLdA mov.f64 fA, 0d0;
    // Store to shared sA[ rTy*16 + rTx ]
    mul.lo.u32 rBDx, rTy, 16;
    add.s32 rBDx, rBDx, rTx;
    mul.wide.u32 rdOff, rBDx, 8;
    cvta.to.shared.u64 rdSA, sA;
    add.s64 rdSA, rdSA, rdOff;
    st.shared.f64 [rdSA], fA;

    // Load B tile element B[rT + rTy, j] -> sB[rTy, rTx]
    .reg .b32 rRowB;
    add.s32 rRowB, rT, rTy;
    // Bounds check for B
    setp.lt.u32 pLdB, rRowB, rK;
    @pLdB setp.lt.u32 pLdB, rJ, rN;
    @pLdB {
        mul.wide.u32 rdOff, rRowB, rN; // (rT+ty)*N
        add.s64 rdOff, rdOff, rJ;      // + j
        mul.wide.u32 rdOff, rdOff, 8;
        add.s64 rdBp, rdB, rdOff;
        ld.global.f64 fB, [rdBp];
    }
    @!pLdB mov.f64 fB, 0d0;
    // Store to shared sB[ rTy*16 + rTx ]
    mul.lo.u32 rBDy, rTy, 16;
    add.s32 rBDy, rBDy, rTx;
    mul.wide.u32 rdOff, rBDy, 8;
    cvta.to.shared.u64 rdSB, sB;
    add.s64 rdSB, rdSB, rdOff;
    st.shared.f64 [rdSB], fB;

    // Sync threads in block
    bar.sync 0;

    // Compute partial products for this tile
    .reg .b32 rP;
    mov.u32 rP, 0;
INNER:
    setp.ge.u32 pLoop, rP, 16;
    @pLoop bra AFTER_INNER;
    // Load sA[ty, p]
    mul.lo.u32 rBDx, rTy, 16;
    add.s32 rBDx, rBDx, rP;
    mul.wide.u32 rdOff, rBDx, 8;
    cvta.to.shared.u64 rdSA, sA;
    add.s64 rdSA, rdSA, rdOff;
    ld.shared.f64 fA, [rdSA];
    // Load sB[p, tx]
    mul.lo.u32 rBDy, rP, 16;
    add.s32 rBDy, rBDy, rTx;
    mul.wide.u32 rdOff, rBDy, 8;
    cvta.to.shared.u64 rdSB, sB;
    add.s64 rdSB, rdSB, rdOff;
    ld.shared.f64 fB, [rdSB];
    // Accumulate
    fma.rn.f64 fSum, fA, fB, fSum;
    add.s32 rP, rP, 1;
    bra INNER;
AFTER_INNER:

    // Sync before next tile
    bar.sync 0;
    // Advance tile start by 16
    add.s32 rT, rT, 16;
    bra TILE_LOOP;

TILE_END:
    // Bounds check store
    setp.lt.u32 pOOB, rI, rM;
    @pOOB setp.lt.u32 pOOB, rJ, rN;
    @pOOB st.global.f64 [rdCp], fSum;
    ret;
}`

func ensureMatMulKernel(deviceID int) (C.CUfunction, error) {
    if err := ensurePrimaryContext(deviceID); err != nil { return nil, err }
    cudaCtxRegistry.mu.Lock()
    entry := cudaCtxRegistry.m[deviceID]
    if entry.kernelLoadedMat {
        fn := entry.fnMat
        cudaCtxRegistry.mu.Unlock()
        return fn, nil
    }
    cudaCtxRegistry.mu.Unlock()
    cstr := C.CString(matMulPTX)
    defer C.free(unsafe.Pointer(cstr))
    var mod C.CUmodule
    if res := C.cuModuleLoadData(&mod, unsafe.Pointer(cstr)); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleLoadData(matmul) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    var fn C.CUfunction
    cname := C.CString("matmul_tiled_f64")
    defer C.free(unsafe.Pointer(cname))
    if res := C.cuModuleGetFunction(&fn, mod, cname); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuModuleGetFunction(matmul) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    cudaCtxRegistry.mu.Lock()
    entry = cudaCtxRegistry.m[deviceID]
    entry.modMat = mod
    entry.fnMat = fn
    entry.kernelLoadedMat = true
    cudaCtxRegistry.mu.Unlock()
    return fn, nil
}

func (d *CUDADevice) Name() string {
	return d.name
}

func (d *CUDADevice) ComputeCapability() (major, minor int) {
	return d.major, d.minor
}

func (d *CUDADevice) MemorySize() int64 {
	return d.memory
}

func (d *CUDADevice) IsAvailable() bool {
	return true // If device was detected, it's available
}

func (d *CUDADevice) GetBackend() string {
	return "CUDA"
}

// CUDA-specific methods

func (d *CUDADevice) AllocateBuffer(size int64) (GPUBuffer, error) {
    if size <= 0 {
        return nil, NewGPUError("allocation size must be positive", GPUErrorInvalidParameter)
    }
    if err := ensurePrimaryContext(d.deviceID); err != nil {
        return nil, err
    }
    var dptr C.CUdeviceptr
    res := C.cuMemAlloc(&dptr, C.size_t(size))
    if res != C.CUDA_SUCCESS {
        errType := GPUErrorKernelFailed
        if res == C.CUDA_ERROR_OUT_OF_MEMORY {
            errType = GPUErrorOutOfMemory
        }
        return nil, NewGPUError(fmt.Sprintf("cuMemAlloc(%d) failed: %s (%s)", size, C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), errType)
    }
    return &CUDABuffer{size: size, deviceID: d.deviceID, dptr: dptr}, nil
}

func (d *CUDADevice) CreateStream() (GPUStream, error) {
    if err := ensurePrimaryContext(d.deviceID); err != nil {
        return nil, err
    }
    var str C.CUstream
    res := C.cuStreamCreate(&str, 0)
    if res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuStreamCreate failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return &CUDAStream{deviceID: d.deviceID, stream: str}, nil
}

func (d *CUDADevice) Reset() error {
    // Release primary context for this device if retained
    cudaCtxRegistry.mu.Lock()
    entry, ok := cudaCtxRegistry.m[d.deviceID]
    if !ok || !entry.retained {
        cudaCtxRegistry.mu.Unlock()
        return nil
    }
    dev := entry.dev
    entry.retained = false
    entry.ctx = nil
    cudaCtxRegistry.mu.Unlock()
    if res := C.cuDevicePrimaryCtxRelease(dev); res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuDevicePrimaryCtxRelease failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return nil
}

func (d *CUDADevice) ExecuteVectorAdd(a, b []float64) ([]float64, error) {
    if len(a) != len(b) {
        return nil, NewGPUError("Array size mismatch", GPUErrorInvalidParameter)
    }
    n := len(a)
    if n == 0 {
        return []float64{}, nil
    }
    if err := ensurePrimaryContext(d.deviceID); err != nil {
        return nil, err
    }
    fn, err := ensureVectorAddKernel(d.deviceID)
    if err != nil {
        return nil, err
    }
    // Allocate device buffers
    bufA, err := d.AllocateBuffer(int64(n * 8))
    if err != nil { return nil, err }
    defer bufA.Free()
    bufB, err := d.AllocateBuffer(int64(n * 8))
    if err != nil { return nil, err }
    defer bufB.Free()
    bufC, err := d.AllocateBuffer(int64(n * 8))
    if err != nil { return nil, err }
    defer bufC.Free()

    // Copy inputs
    if err := bufA.CopyFromHost(a); err != nil { return nil, err }
    if err := bufB.CopyFromHost(b); err != nil { return nil, err }

    // Extract CUdeviceptrs
    dA := bufA.(*CUDABuffer).dptr
    dB := bufB.(*CUDABuffer).dptr
    dC := bufC.(*CUDABuffer).dptr
    nCu := C.uint(n)

    // Kernel parameters array
    params := []unsafe.Pointer{
        unsafe.Pointer(&dA),
        unsafe.Pointer(&dB),
        unsafe.Pointer(&dC),
        unsafe.Pointer(&nCu),
    }

    // Launch configuration
    threads := C.uint(256)
    blocks := C.uint((n + 255) / 256)

    // Launch kernel on default stream
    if res := C.cuLaunchKernel(fn,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        nil,
        (**C.void)(unsafe.Pointer(&params[0])),
        nil); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuLaunchKernel failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    if res := C.cuCtxSynchronize(); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuCtxSynchronize failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }

    // Copy result back
    out := make([]float64, n)
    if err := bufC.CopyToHost(out); err != nil { return nil, err }
    return out, nil
}

func (d *CUDADevice) ExecuteMatMul(A, B []float64, M, N, K int) ([]float64, error) {
    if M < 0 || N < 0 || K < 0 || len(A) != M*K || len(B) != K*N {
        return nil, NewGPUError("Matrix dimensions mismatch", GPUErrorInvalidParameter)
    }
    if M == 0 || N == 0 || K == 0 {
        return []float64{}, nil
    }
    if err := ensurePrimaryContext(d.deviceID); err != nil { return nil, err }
    fn, err := ensureMatMulKernel(d.deviceID)
    if err != nil { return nil, err }

    // Allocate device buffers
    bytesA := int64(M*K*8)
    bytesB := int64(K*N*8)
    bytesC := int64(M*N*8)
    dA, err := d.AllocateBuffer(bytesA); if err != nil { return nil, err }
    defer dA.Free()
    dB, err := d.AllocateBuffer(bytesB); if err != nil { return nil, err }
    defer dB.Free()
    dC, err := d.AllocateBuffer(bytesC); if err != nil { return nil, err }
    defer dC.Free()

    if err := dA.CopyFromHost(A); err != nil { return nil, err }
    if err := dB.CopyFromHost(B); err != nil { return nil, err }

    cuA := dA.(*CUDABuffer).dptr
    cuB := dB.(*CUDABuffer).dptr
    cuC := dC.(*CUDABuffer).dptr
    cuM := C.uint(M)
    cuN := C.uint(N)
    cuK := C.uint(K)

    params := []unsafe.Pointer{
        unsafe.Pointer(&cuA),
        unsafe.Pointer(&cuB),
        unsafe.Pointer(&cuC),
        unsafe.Pointer(&cuM),
        unsafe.Pointer(&cuN),
        unsafe.Pointer(&cuK),
    }

    // Launch config: 16x16 threads, enough blocks to cover MxN
    tx := C.uint(16)
    ty := C.uint(16)
    bx := C.uint((N + 15) / 16)
    by := C.uint((M + 15) / 16)

    if res := C.cuLaunchKernel(fn,
        bx, by, 1,
        tx, ty, 1,
        0,
        nil,
        (**C.void)(unsafe.Pointer(&params[0])),
        nil); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuLaunchKernel(matmul) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    if res := C.cuCtxSynchronize(); res != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("cuCtxSynchronize(matmul) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }

    out := make([]float64, M*N)
    if err := dC.CopyToHost(out); err != nil { return nil, err }
    return out, nil
}

func (d *CUDADevice) ExecuteSum(data []float64) (float64, error) {
    if len(data) == 0 {
        return 0, NewGPUError("Cannot sum empty array", GPUErrorInvalidParameter)
    }
    n := len(data)
    if err := ensurePrimaryContext(d.deviceID); err != nil { return 0, err }
    fn, err := ensureSumKernel(d.deviceID)
    if err != nil { return 0, err }

    // Allocate input and partials buffers
    bufIn, err := d.AllocateBuffer(int64(n*8)); if err != nil { return 0, err }
    defer bufIn.Free()
    threads := 256
    blocks := (n + threads - 1) / threads
    if blocks <= 0 { blocks = 1 }
    totalThreads := blocks * threads
    bufOut, err := d.AllocateBuffer(int64(totalThreads*8)); if err != nil { return 0, err }
    defer bufOut.Free()

    if err := bufIn.CopyFromHost(data); err != nil { return 0, err }

    dIn := bufIn.(*CUDABuffer).dptr
    dOut := bufOut.(*CUDABuffer).dptr
    cuN := C.uint(n)
    params := []unsafe.Pointer{
        unsafe.Pointer(&dIn),
        unsafe.Pointer(&dOut),
        unsafe.Pointer(&cuN),
    }

    if res := C.cuLaunchKernel(fn,
        C.uint(blocks), 1, 1,
        C.uint(threads), 1, 1,
        0,
        nil,
        (**C.void)(unsafe.Pointer(&params[0])),
        nil); res != C.CUDA_SUCCESS {
        return 0, NewGPUError(fmt.Sprintf("cuLaunchKernel(sum) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    if res := C.cuCtxSynchronize(); res != C.CUDA_SUCCESS {
        return 0, NewGPUError(fmt.Sprintf("cuCtxSynchronize(sum) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }

    // Tree reduction: apply staged reductions until length <= 256, then final reduce
    fnTree, err := ensureTreeReduceKernel(d.deviceID)
    if err != nil { return 0, err }
    curPtr := bufOut.(*CUDABuffer).dptr
    curLen := totalThreads
    // Temporary buffer sized to at most ceil(curLen/(2*threads)) doubles; allocate generously once
    // Max stages halves length each time; allocate half of initial
    tmpCapacity := (curLen + 1) / 2
    if tmpCapacity < 1 { tmpCapacity = 1 }
    tmpBuf, err := d.AllocateBuffer(int64(tmpCapacity*8))
    if err != nil { return 0, err }
    defer tmpBuf.Free()
    tmpPtr := tmpBuf.(*CUDABuffer).dptr

    for curLen > 256 {
        // Each block reduces ~2*threads elements; number of blocks is ceil(curLen/(2*threads))
        blocks := (curLen + (2*threads) - 1) / (2*threads)
        cuL := C.uint(curLen)
        params := []unsafe.Pointer{
            unsafe.Pointer(&curPtr),
            unsafe.Pointer(&tmpPtr),
            unsafe.Pointer(&cuL),
        }
        if res := C.cuLaunchKernel(fnTree,
            C.uint(blocks), 1, 1,
            C.uint(threads), 1, 1,
            0,
            nil,
            (**C.void)(unsafe.Pointer(&params[0])),
            nil); res != C.CUDA_SUCCESS {
            return 0, NewGPUError(fmt.Sprintf("cuLaunchKernel(tree reduce) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
        }
        if res := C.cuCtxSynchronize(); res != C.CUDA_SUCCESS {
            return 0, NewGPUError(fmt.Sprintf("cuCtxSynchronize(tree reduce) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
        }
        // Next stage: input becomes tmp, length becomes blocks
        curPtr, tmpPtr = tmpPtr, curPtr
        curLen = blocks
    }

    // Final reduction on GPU using single-block shared-memory kernel
    fn2, err := ensureFinalReduceKernel(d.deviceID)
    if err != nil { return 0, err }
    // Allocate single-double output
    dOut, err := d.AllocateBuffer(8)
    if err != nil { return 0, err }
    defer dOut.Free()
    dOutPtr := dOut.(*CUDABuffer).dptr
    cuL := C.uint(curLen)
    params2 := []unsafe.Pointer{
        unsafe.Pointer(&curPtr),
        unsafe.Pointer(&dOutPtr),
        unsafe.Pointer(&cuL),
    }
    if res := C.cuLaunchKernel(fn2,
        1, 1, 1,
        256, 1, 1,
        0,
        nil,
        (**C.void)(unsafe.Pointer(&params2[0])),
        nil); res != C.CUDA_SUCCESS {
        return 0, NewGPUError(fmt.Sprintf("cuLaunchKernel(final reduce) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    if res := C.cuCtxSynchronize(); res != C.CUDA_SUCCESS {
        return 0, NewGPUError(fmt.Sprintf("cuCtxSynchronize(final reduce) failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    out := make([]float64, 1)
    if err := dOut.CopyToHost(out); err != nil { return 0, err }
    return out[0], nil
}

// CUDABuffer represents CUDA GPU memory buffer
type CUDABuffer struct {
    size     int64
    deviceID int
    dptr     C.CUdeviceptr
}

func (b *CUDABuffer) Size() int64 {
	return b.size
}

func (b *CUDABuffer) CopyFromHost(data interface{}) error {
    if b == nil || b.dptr == 0 {
        return NewGPUError("invalid CUDA buffer", GPUErrorInvalidParameter)
    }
    bytes, err := interfaceToBytes(data)
    if err != nil {
        return err
    }
    if int64(len(bytes)) > b.size {
        return NewGPUError("host data larger than device buffer", GPUErrorInvalidParameter)
    }
    if len(bytes) == 0 {
        return nil
    }
    if err := ensurePrimaryContext(b.deviceID); err != nil {
        return err
    }
    res := C.cuMemcpyHtoD(b.dptr, unsafe.Pointer(&bytes[0]), C.size_t(len(bytes)))
    if res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuMemcpyHtoD failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return nil
}

func (b *CUDABuffer) CopyFromHostAsync(data interface{}, stream GPUStream) error {
    if b == nil || b.dptr == 0 {
        return NewGPUError("invalid CUDA buffer", GPUErrorInvalidParameter)
    }
    bytes, err := interfaceToBytes(data)
    if err != nil {
        return err
    }
    if int64(len(bytes)) > b.size {
        return NewGPUError("host data larger than device buffer", GPUErrorInvalidParameter)
    }
    if len(bytes) == 0 {
        return nil
    }
    s, ok := stream.(*CUDAStream)
    if !ok || s == nil {
        return NewGPUError("invalid CUDA stream", GPUErrorInvalidParameter)
    }
    if err := ensurePrimaryContext(b.deviceID); err != nil {
        return err
    }
    res := C.cuMemcpyHtoDAsync(b.dptr, unsafe.Pointer(&bytes[0]), C.size_t(len(bytes)), s.stream)
    if res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuMemcpyHtoDAsync failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return nil
}

func (b *CUDABuffer) CopyToHost(data interface{}) error {
    if b == nil || b.dptr == 0 {
        return NewGPUError("invalid CUDA buffer", GPUErrorInvalidParameter)
    }
    bytes, err := interfaceToBytes(data)
    if err != nil {
        return err
    }
    if int64(len(bytes)) > b.size {
        return NewGPUError("destination smaller than device buffer", GPUErrorInvalidParameter)
    }
    if len(bytes) == 0 {
        return nil
    }
    if err := ensurePrimaryContext(b.deviceID); err != nil {
        return err
    }
    res := C.cuMemcpyDtoH(unsafe.Pointer(&bytes[0]), b.dptr, C.size_t(len(bytes)))
    if res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuMemcpyDtoH failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return nil
}

func (b *CUDABuffer) Free() error {
    if b == nil || b.dptr == 0 {
        return nil
    }
    if err := ensurePrimaryContext(b.deviceID); err != nil {
        return err
    }
    res := C.cuMemFree(b.dptr)
    if res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuMemFree failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    b.dptr = 0
    return nil
}

// CUDAStream represents CUDA computation stream
type CUDAStream struct {
    deviceID int
    stream   C.CUstream
}

func (s *CUDAStream) Synchronize() error {
    if s == nil || s.stream == nil {
        return NewGPUError("invalid CUDA stream", GPUErrorInvalidParameter)
    }
    if err := ensurePrimaryContext(s.deviceID); err != nil {
        return err
    }
    res := C.cuStreamSynchronize(s.stream)
    if res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuStreamSynchronize failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    return nil
}

func (s *CUDAStream) Destroy() error {
    if s == nil || s.stream == nil {
        return nil
    }
    if err := ensurePrimaryContext(s.deviceID); err != nil {
        return err
    }
    res := C.cuStreamDestroy(s.stream)
    if res != C.CUDA_SUCCESS {
        return NewGPUError(fmt.Sprintf("cuStreamDestroy failed: %s (%s)", C.GoString(C.cuErrName(res)), C.GoString(C.cuErrStr(res))), GPUErrorKernelFailed)
    }
    s.stream = nil
    return nil
}

// DetectCUDADevices detects and initializes CUDA devices
func DetectCUDADevices() ([]HardwareGPUDevice, error) {
	// Initialize CUDA
    result := C.cuInit(0)
    if result != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("Failed to initialize CUDA: %s (%s)", C.GoString(C.cuErrName(result)), C.GoString(C.cuErrStr(result))), GPUErrorDeviceNotFound)
    }

	// Get device count
	var deviceCount C.int
    result = C.cuDeviceGetCount(&deviceCount)
    if result != C.CUDA_SUCCESS {
        return nil, NewGPUError(fmt.Sprintf("Failed to get CUDA device count: %s (%s)", C.GoString(C.cuErrName(result)), C.GoString(C.cuErrStr(result))), GPUErrorDeviceNotFound)
    }

	if deviceCount == 0 {
		return []HardwareGPUDevice{}, nil
	}

    devices := make([]HardwareGPUDevice, 0, int(deviceCount))

	for i := 0; i < int(deviceCount); i++ {
		// Get device name
		nameBuffer := make([]C.char, 256)
		namePtr := (*C.char)(unsafe.Pointer(&nameBuffer[0]))

		if C.getCudaDeviceName(C.int(i), namePtr, 256) != 0 {
			continue // Skip devices we can't query
		}

		name := C.GoString(namePtr)

		// Get device memory
		var memBytes C.size_t
		if C.getCudaDeviceMemory(C.int(i), &memBytes) != 0 {
			continue // Skip devices we can't query
		}

		// Get compute capability
		var major, minor C.int
		if C.getCudaDeviceCapability(C.int(i), &major, &minor) != 0 {
			continue // Skip devices we can't query
		}

		device := &CUDADevice{
			deviceID: i,
			name:     name,
			memory:   int64(memBytes),
			major:    int(major),
			minor:    int(minor),
		}

		devices = append(devices, device)
	}

	return devices, nil
}

// CUDAManager implements HardwareGPUManager for CUDA devices
type CUDAManager struct {
	devices []HardwareGPUDevice
}

func NewCUDAManager() (*CUDAManager, error) {
	devices, err := DetectCUDADevices()
	if err != nil {
		return nil, err
	}

	return &CUDAManager{devices: devices}, nil
}

func (m *CUDAManager) EnumerateDevices() ([]HardwareGPUDevice, error) {
	return m.devices, nil
}

func (m *CUDAManager) GetDefaultDevice() (HardwareGPUDevice, error) {
	if len(m.devices) == 0 {
		return nil, NewGPUError("No CUDA devices available", GPUErrorDeviceNotFound)
	}
	return m.devices[0], nil
}

func (m *CUDAManager) ExecuteMultiGPUSum(data []float64, devices []HardwareGPUDevice) (float64, error) {
	// For now, just use the first device
	if len(devices) == 0 {
		return 0, NewGPUError("No devices provided", GPUErrorInvalidParameter)
	}

	return devices[0].ExecuteSum(data)
}

// Lock goroutine to OS thread for CUDA operations
func init() {
	runtime.LockOSThread()
}
