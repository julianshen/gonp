// Package gpu provides OpenCL hardware integration
//
// This module implements actual OpenCL device detection and initialization
// using cgo bindings to the OpenCL API.
//
// Build Requirements:
//   - OpenCL SDK installed on the system
//   - CGO_ENABLED=1
//   - Proper OpenCL library paths
//   - OpenCL headers available
//
// Usage:
//   devices, err := DetectOpenCLDevices()
//   if err == nil {
//       device := devices[0]
//       // Use OpenCL device for computations
//   }
//
//go:build opencl
// +build opencl

package gpu

/*
#cgo CFLAGS: -I/usr/include/CL -I/usr/local/include/CL -I/opt/rocm/include/CL -I/System/Library/Frameworks/OpenCL.framework/Headers
#cgo linux LDFLAGS: -lOpenCL
#cgo darwin LDFLAGS: -framework OpenCL
#cgo windows LDFLAGS: -lOpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <string.h>

// Helper function to get platform information
int getOpenCLPlatforms(cl_platform_id **platforms, cl_uint *num_platforms) {
    cl_int err = clGetPlatformIDs(0, NULL, num_platforms);
    if (err != CL_SUCCESS || *num_platforms == 0) {
        return -1;
    }

    *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * (*num_platforms));
    err = clGetPlatformIDs(*num_platforms, *platforms, NULL);
    if (err != CL_SUCCESS) {
        free(*platforms);
        return -1;
    }

    return 0;
}

// Helper function to get devices for a platform
int getOpenCLDevices(cl_platform_id platform, cl_device_id **devices, cl_uint *num_devices) {
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, num_devices);
    if (err != CL_SUCCESS || *num_devices == 0) {
        // Try all device types if no GPU found
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, num_devices);
        if (err != CL_SUCCESS || *num_devices == 0) {
            return -1;
        }
    }

    *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * (*num_devices));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, *num_devices, *devices, NULL);
    if (err != CL_SUCCESS) {
        // Fallback to all devices
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *num_devices, *devices, NULL);
        if (err != CL_SUCCESS) {
            free(*devices);
            return -1;
        }
    }

    return 0;
}

// Helper function to get device name
int getOpenCLDeviceName(cl_device_id device, char *name, size_t name_size) {
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, name, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

// Helper function to get device memory
int getOpenCLDeviceMemory(cl_device_id device, cl_ulong *memory) {
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), memory, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

// Helper function to get device compute units
int getOpenCLDeviceComputeUnits(cl_device_id device, cl_uint *compute_units) {
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), compute_units, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

// Helper function to get device type
int getOpenCLDeviceType(cl_device_id device, cl_device_type *device_type) {
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), device_type, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

// Simple kernel source for vector addition
const char* vector_add_kernel_source =
"__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {\n"
"    int i = get_global_id(0);\n"
"    C[i] = A[i] + B[i];\n"
"}\n";

// Simple kernel source for sum reduction
const char* sum_reduction_kernel_source =
"__kernel void sum_reduction(__global const float* input, __global float* output, __local float* temp) {\n"
"    int global_id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    \n"
"    temp[local_id] = input[global_id];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    \n"
"    for (int stride = local_size / 2; stride > 0; stride /= 2) {\n"
"        if (local_id < stride) {\n"
"            temp[local_id] += temp[local_id + stride];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    if (local_id == 0) {\n"
"        output[get_group_id(0)] = temp[0];\n"
"    }\n"
"}\n";
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// OpenCLDevice represents an OpenCL GPU device
type OpenCLDevice struct {
	deviceID     C.cl_device_id
	platformID   C.cl_platform_id
	context      C.cl_context
	commandQueue C.cl_command_queue
	name         string
	memory       int64
	computeUnits int
	deviceType   string
}

func (d *OpenCLDevice) Name() string {
	return d.name
}

func (d *OpenCLDevice) ComputeCapability() (major, minor int) {
	// OpenCL doesn't have compute capability like CUDA
	// Return version-like information based on device type
	if d.deviceType == "GPU" {
		return 2, 0 // OpenCL 2.0 GPU
	}
	return 1, 2 // OpenCL 1.2 other
}

func (d *OpenCLDevice) MemorySize() int64 {
	return d.memory
}

func (d *OpenCLDevice) IsAvailable() bool {
	return d.deviceID != nil
}

func (d *OpenCLDevice) GetBackend() string {
	return "OpenCL"
}

// OpenCL-specific methods

func (d *OpenCLDevice) AllocateBuffer(size int64) (GPUBuffer, error) {
	if d.context == nil {
		return nil, NewGPUError("OpenCL context not initialized", GPUErrorDeviceNotFound)
	}

	var err C.cl_int
	buffer := C.clCreateBuffer(d.context, C.CL_MEM_READ_WRITE, C.size_t(size), nil, &err)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError(fmt.Sprintf("Failed to allocate OpenCL buffer: %d", int(err)), GPUErrorOutOfMemory)
	}

	return &OpenCLBuffer{
		buffer: buffer,
		size:   size,
		device: d,
	}, nil
}

func (d *OpenCLDevice) CreateStream() (GPUStream, error) {
	// OpenCL uses command queues instead of streams
	return &OpenCLStream{
		commandQueue: d.commandQueue,
		device:       d,
	}, nil
}

func (d *OpenCLDevice) Reset() error {
	// For now, return nil (would implement device reset)
	return nil
}

func (d *OpenCLDevice) ExecuteVectorAdd(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, NewGPUError("Array size mismatch", GPUErrorInvalidParameter)
	}

	if len(a) == 0 {
		return []float64{}, nil
	}

	size := len(a)
	result := make([]float64, size)

	// Convert to float32 for OpenCL kernel
	aFloat32 := make([]float32, size)
	bFloat32 := make([]float32, size)

	for i := 0; i < size; i++ {
		aFloat32[i] = float32(a[i])
		bFloat32[i] = float32(b[i])
	}

	// Create OpenCL buffers
	var err C.cl_int
	bufferA := C.clCreateBuffer(d.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(size*4), unsafe.Pointer(&aFloat32[0]), &err)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to create buffer A", GPUErrorOutOfMemory)
	}
	defer C.clReleaseMemObject(bufferA)

	bufferB := C.clCreateBuffer(d.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(size*4), unsafe.Pointer(&bFloat32[0]), &err)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to create buffer B", GPUErrorOutOfMemory)
	}
	defer C.clReleaseMemObject(bufferB)

	bufferC := C.clCreateBuffer(d.context, C.CL_MEM_WRITE_ONLY, C.size_t(size*4), nil, &err)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to create buffer C", GPUErrorOutOfMemory)
	}
	defer C.clReleaseMemObject(bufferC)

	// Create and build program
	kernelSource := C.CString(C.vector_add_kernel_source)
	defer C.free(unsafe.Pointer(kernelSource))

	program := C.clCreateProgramWithSource(d.context, 1, &kernelSource, nil, &err)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to create program", GPUErrorKernelFailed)
	}
	defer C.clReleaseProgram(program)

	err = C.clBuildProgram(program, 1, &d.deviceID, nil, nil, nil)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to build program", GPUErrorKernelFailed)
	}

	// Create kernel
	kernelName := C.CString("vector_add")
	defer C.free(unsafe.Pointer(kernelName))

	kernel := C.clCreateKernel(program, kernelName, &err)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to create kernel", GPUErrorKernelFailed)
	}
	defer C.clReleaseKernel(kernel)

	// Set kernel arguments
	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(bufferA)), unsafe.Pointer(&bufferA))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(bufferB)), unsafe.Pointer(&bufferB))
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(bufferC)), unsafe.Pointer(&bufferC))

	// Execute kernel
	globalSize := C.size_t(size)
	err = C.clEnqueueNDRangeKernel(d.commandQueue, kernel, 1, nil, &globalSize, nil, 0, nil, nil)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to execute kernel", GPUErrorKernelFailed)
	}

	// Read result
	resultFloat32 := make([]float32, size)
	err = C.clEnqueueReadBuffer(d.commandQueue, bufferC, C.CL_TRUE, 0, C.size_t(size*4),
		unsafe.Pointer(&resultFloat32[0]), 0, nil, nil)
	if err != C.CL_SUCCESS {
		return nil, NewGPUError("Failed to read result", GPUErrorKernelFailed)
	}

	// Convert back to float64
	for i := 0; i < size; i++ {
		result[i] = float64(resultFloat32[i])
	}

	return result, nil
}

func (d *OpenCLDevice) ExecuteMatMul(A, B []float64, M, N, K int) ([]float64, error) {
	// For now, fallback to CPU implementation
	if len(A) != M*K || len(B) != K*N {
		return nil, NewGPUError("Matrix dimensions mismatch", GPUErrorInvalidParameter)
	}

	result := make([]float64, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			result[i*N+j] = sum
		}
	}
	return result, nil
}

func (d *OpenCLDevice) ExecuteSum(data []float64) (float64, error) {
	if len(data) == 0 {
		return 0, NewGPUError("Cannot sum empty array", GPUErrorInvalidParameter)
	}

	// For small arrays, use CPU
	if len(data) < 1024 {
		sum := 0.0
		for _, v := range data {
			sum += v
		}
		return sum, nil
	}

	// Use OpenCL reduction for larger arrays
	size := len(data)

	// Convert to float32
	dataFloat32 := make([]float32, size)
	for i := 0; i < size; i++ {
		dataFloat32[i] = float32(data[i])
	}

	// Create input buffer
	var err C.cl_int
	inputBuffer := C.clCreateBuffer(d.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(size*4), unsafe.Pointer(&dataFloat32[0]), &err)
	if err != C.CL_SUCCESS {
		// Fallback to CPU
		sum := 0.0
		for _, v := range data {
			sum += v
		}
		return sum, nil
	}
	defer C.clReleaseMemObject(inputBuffer)

	// For simplicity, use CPU sum for now
	// (Full OpenCL reduction implementation would be more complex)
	sum := 0.0
	for _, v := range data {
		sum += v
	}

	return sum, nil
}

// OpenCLBuffer represents OpenCL GPU memory buffer
type OpenCLBuffer struct {
	buffer C.cl_mem
	size   int64
	device *OpenCLDevice
}

func (b *OpenCLBuffer) Size() int64 {
	return b.size
}

func (b *OpenCLBuffer) CopyFromHost(data interface{}) error {
	if b.buffer == nil {
		return NewGPUError("Buffer not allocated", GPUErrorInvalidParameter)
	}

	// For now, just validate the operation
	return nil
}

func (b *OpenCLBuffer) CopyFromHostAsync(data interface{}, stream GPUStream) error {
	return b.CopyFromHost(data)
}

func (b *OpenCLBuffer) CopyToHost(data interface{}) error {
	if b.buffer == nil {
		return NewGPUError("Buffer not allocated", GPUErrorInvalidParameter)
	}

	// For now, just validate the operation
	return nil
}

func (b *OpenCLBuffer) Free() error {
	if b.buffer != nil {
		C.clReleaseMemObject(b.buffer)
		b.buffer = nil
	}
	return nil
}

// OpenCLStream represents OpenCL command queue
type OpenCLStream struct {
	commandQueue C.cl_command_queue
	device       *OpenCLDevice
}

func (s *OpenCLStream) Synchronize() error {
	if s.commandQueue != nil {
		C.clFinish(s.commandQueue)
	}
	return nil
}

func (s *OpenCLStream) Destroy() error {
	// Command queue is managed by device, don't release here
	return nil
}

// DetectOpenCLDevices detects and initializes OpenCL devices
func DetectOpenCLDevices() ([]HardwareGPUDevice, error) {
	var platforms *C.cl_platform_id
	var numPlatforms C.cl_uint

	// Get platforms
	if C.getOpenCLPlatforms(&platforms, &numPlatforms) != 0 {
		return nil, NewGPUError("No OpenCL platforms found", GPUErrorDeviceNotFound)
	}
	defer C.free(unsafe.Pointer(platforms))

	if numPlatforms == 0 {
		return []HardwareGPUDevice{}, nil
	}

	var devices []HardwareGPUDevice

	// Iterate through platforms
	platformSlice := (*[1 << 10]C.cl_platform_id)(unsafe.Pointer(platforms))[:numPlatforms:numPlatforms]

	for _, platform := range platformSlice {
		var platformDevices *C.cl_device_id
		var numDevices C.cl_uint

		if C.getOpenCLDevices(platform, &platformDevices, &numDevices) != 0 {
			continue // Skip platforms with no devices
		}

		if numDevices == 0 {
			C.free(unsafe.Pointer(platformDevices))
			continue
		}

		deviceSlice := (*[1 << 10]C.cl_device_id)(unsafe.Pointer(platformDevices))[:numDevices:numDevices]

		for _, deviceID := range deviceSlice {
			// Get device name
			nameBuffer := make([]C.char, 256)
			if C.getOpenCLDeviceName(deviceID, &nameBuffer[0], 256) != 0 {
				continue
			}
			name := C.GoString(&nameBuffer[0])

			// Get device memory
			var memory C.cl_ulong
			if C.getOpenCLDeviceMemory(deviceID, &memory) != 0 {
				continue
			}

			// Get compute units
			var computeUnits C.cl_uint
			if C.getOpenCLDeviceComputeUnits(deviceID, &computeUnits) != 0 {
				computeUnits = 1 // Default
			}

			// Get device type
			var deviceType C.cl_device_type
			deviceTypeStr := "Unknown"
			if C.getOpenCLDeviceType(deviceID, &deviceType) == 0 {
				switch deviceType {
				case C.CL_DEVICE_TYPE_GPU:
					deviceTypeStr = "GPU"
				case C.CL_DEVICE_TYPE_CPU:
					deviceTypeStr = "CPU"
				case C.CL_DEVICE_TYPE_ACCELERATOR:
					deviceTypeStr = "Accelerator"
				}
			}

			// Create context
			var err C.cl_int
			context := C.clCreateContext(nil, 1, &deviceID, nil, nil, &err)
			if err != C.CL_SUCCESS {
				continue
			}

			// Create command queue
			commandQueue := C.clCreateCommandQueue(context, deviceID, 0, &err)
			if err != C.CL_SUCCESS {
				C.clReleaseContext(context)
				continue
			}

			device := &OpenCLDevice{
				deviceID:     deviceID,
				platformID:   platform,
				context:      context,
				commandQueue: commandQueue,
				name:         name,
				memory:       int64(memory),
				computeUnits: int(computeUnits),
				deviceType:   deviceTypeStr,
			}

			devices = append(devices, device)
		}

		C.free(unsafe.Pointer(platformDevices))
	}

	return devices, nil
}

// OpenCLManager implements HardwareGPUManager for OpenCL devices
type OpenCLManager struct {
	devices []HardwareGPUDevice
}

func NewOpenCLManager() (*OpenCLManager, error) {
	devices, err := DetectOpenCLDevices()
	if err != nil {
		return nil, err
	}

	return &OpenCLManager{devices: devices}, nil
}

func (m *OpenCLManager) EnumerateDevices() ([]HardwareGPUDevice, error) {
	return m.devices, nil
}

func (m *OpenCLManager) GetDefaultDevice() (HardwareGPUDevice, error) {
	if len(m.devices) == 0 {
		return nil, NewGPUError("No OpenCL devices available", GPUErrorDeviceNotFound)
	}
	return m.devices[0], nil
}

func (m *OpenCLManager) ExecuteMultiGPUSum(data []float64, devices []HardwareGPUDevice) (float64, error) {
	if len(devices) == 0 {
		return 0, NewGPUError("No devices provided", GPUErrorInvalidParameter)
	}

	// For now, just use the first device
	return devices[0].ExecuteSum(data)
}

// Lock goroutine to OS thread for OpenCL operations (similar to CUDA)
func init() {
	runtime.LockOSThread()
}
