// Package gpu provides GPU hardware types and interfaces
//
// This module defines the common types and interfaces used across
// all GPU hardware implementations (CUDA, OpenCL, etc.).

package gpu

// HardwareGPUDevice represents a real GPU device interface
type HardwareGPUDevice interface {
	Name() string
	ComputeCapability() (major, minor int)
	MemorySize() int64
	IsAvailable() bool
	GetBackend() string

	// Memory management
	AllocateBuffer(size int64) (GPUBuffer, error)
	CreateStream() (GPUStream, error)
	Reset() error

	// Kernel execution
	ExecuteVectorAdd(a, b []float64) ([]float64, error)
	ExecuteMatMul(A, B []float64, M, N, K int) ([]float64, error)
	ExecuteSum(data []float64) (float64, error)
}

// GPUBuffer represents GPU memory buffer
type GPUBuffer interface {
	Size() int64
	CopyFromHost(data interface{}) error
	CopyFromHostAsync(data interface{}, stream GPUStream) error
	CopyToHost(data interface{}) error
	Free() error
}

// GPUStream represents GPU computation stream
type GPUStream interface {
	Synchronize() error
	Destroy() error
}

// HardwareGPUManager manages real GPU devices
type HardwareGPUManager interface {
	EnumerateDevices() ([]HardwareGPUDevice, error)
	GetDefaultDevice() (HardwareGPUDevice, error)
	ExecuteMultiGPUSum(data []float64, devices []HardwareGPUDevice) (float64, error)
}

// GPU Error types and handling

type GPUErrorType int

const (
	GPUErrorNotImplemented GPUErrorType = iota
	GPUErrorOutOfMemory
	GPUErrorInvalidParameter
	GPUErrorDeviceNotFound
	GPUErrorKernelFailed
)

type GPUError struct {
	Message string
	Type    GPUErrorType
}

func (e *GPUError) Error() string {
	return e.Message
}

func NewGPUError(message string, errorType GPUErrorType) error {
	return &GPUError{Message: message, Type: errorType}
}

func IsOutOfMemoryError(err error) bool {
	if gpuErr, ok := err.(*GPUError); ok {
		return gpuErr.Type == GPUErrorOutOfMemory
	}
	return false
}
