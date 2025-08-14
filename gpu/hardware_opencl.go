// Package gpu provides hardware GPU integration with OpenCL support
//
// This module provides OpenCL-specific implementations when built with OpenCL support.
//
// Build Requirements:
//   - Build with: go build -tags opencl
//   - OpenCL SDK installed (AMD, Intel, NVIDIA)
//   - CGO_ENABLED=1
//
//go:build opencl
// +build opencl

package gpu

// NewHardwareGPUManager creates a hardware GPU manager with OpenCL support
func NewHardwareGPUManager() (HardwareGPUManager, error) {
	// Try OpenCL first
	openclManager, err := NewOpenCLManager()
	if err == nil {
		return openclManager, nil
	}

	// If OpenCL fails, try CUDA (if available)
	devices, cudaErr := DetectCUDADevices()
	if cudaErr == nil && len(devices) > 0 {
		return &HardwareManager{devices: devices, backend: "CUDA"}, nil
	}

	// Return the original OpenCL error if both fail
	return nil, err
}

// DetectCUDADevices provides CUDA detection (stub for OpenCL build)
func DetectCUDADevices() ([]HardwareGPUDevice, error) {
	return nil, NewGPUError("CUDA not available in OpenCL build", GPUErrorNotImplemented)
}
