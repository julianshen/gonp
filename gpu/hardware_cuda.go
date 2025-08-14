// Package gpu provides hardware GPU integration with CUDA support
//
// This module provides CUDA-specific implementations when built with CUDA support.
//
// Build Requirements:
//   - Build with: go build -tags cuda
//   - CUDA Toolkit installed
//   - CGO_ENABLED=1
//
//go:build cuda
// +build cuda

package gpu

// NewHardwareGPUManager creates a hardware GPU manager with CUDA support
func NewHardwareGPUManager() (HardwareGPUManager, error) {
	// Try CUDA first
	cudaManager, err := NewCUDAManager()
	if err == nil {
		return cudaManager, nil
	}

	// If CUDA fails, try OpenCL (if available)
	devices, openclErr := DetectOpenCLDevices()
	if openclErr == nil && len(devices) > 0 {
		return &HardwareManager{devices: devices, backend: "OpenCL"}, nil
	}

	// Return the original CUDA error if both fail
	return nil, err
}

// DetectOpenCLDevices provides OpenCL detection (stub for CUDA build)
func DetectOpenCLDevices() ([]HardwareGPUDevice, error) {
	return nil, NewGPUError("OpenCL not available in CUDA build", GPUErrorNotImplemented)
}
