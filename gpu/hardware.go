// Package gpu provides hardware GPU integration fallback
//
// This module provides fallback implementations when GPU hardware
// is not available or CUDA/OpenCL are not installed.
//
// Build tags:
//   - Default build: No GPU support, returns appropriate errors
//   - cuda build tag: Enables CUDA support
//   - opencl build tag: Enables OpenCL support
//
//go:build !cuda && !opencl
// +build !cuda,!opencl

package gpu

import (
	"fmt"
)

// Fallback implementations for when GPU hardware is not available

// DetectCUDADevices returns an error when CUDA is not available
func DetectCUDADevices() ([]HardwareGPUDevice, error) {
	return nil, NewGPUError("CUDA support not compiled in (use -tags cuda)", GPUErrorNotImplemented)
}

// DetectOpenCLDevices returns an error when OpenCL is not available
func DetectOpenCLDevices() ([]HardwareGPUDevice, error) {
	return nil, NewGPUError("OpenCL support not compiled in (use -tags opencl)", GPUErrorNotImplemented)
}

// NewHardwareGPUManager creates a hardware GPU manager
func NewHardwareGPUManager() (HardwareGPUManager, error) {
	// Try to create managers in order of preference

	// Try CUDA first (if available)
	if cudaManager, err := tryCreateCUDAManager(); err == nil {
		return cudaManager, nil
	}

	// Try OpenCL second (if available)
	if openclManager, err := tryCreateOpenCLManager(); err == nil {
		return openclManager, nil
	}

	// No hardware GPU available
	return nil, NewGPUError("No GPU hardware available (install CUDA/OpenCL and rebuild with appropriate tags)", GPUErrorDeviceNotFound)
}

func tryCreateCUDAManager() (HardwareGPUManager, error) {
	devices, err := DetectCUDADevices()
	if err != nil {
		return nil, err
	}

	if len(devices) == 0 {
		return nil, NewGPUError("No CUDA devices found", GPUErrorDeviceNotFound)
	}

	return &HardwareManager{devices: devices, backend: "CUDA"}, nil
}

func tryCreateOpenCLManager() (HardwareGPUManager, error) {
	devices, err := DetectOpenCLDevices()
	if err != nil {
		return nil, err
	}

	if len(devices) == 0 {
		return nil, NewGPUError("No OpenCL devices found", GPUErrorDeviceNotFound)
	}

	return &HardwareManager{devices: devices, backend: "OpenCL"}, nil
}

// HardwareManager provides a unified interface for GPU hardware
type HardwareManager struct {
	devices []HardwareGPUDevice
	backend string
}

func (m *HardwareManager) EnumerateDevices() ([]HardwareGPUDevice, error) {
	return m.devices, nil
}

func (m *HardwareManager) GetDefaultDevice() (HardwareGPUDevice, error) {
	if len(m.devices) == 0 {
		return nil, NewGPUError("No devices available", GPUErrorDeviceNotFound)
	}
	return m.devices[0], nil
}

func (m *HardwareManager) ExecuteMultiGPUSum(data []float64, devices []HardwareGPUDevice) (float64, error) {
	if len(devices) == 0 {
		return 0, NewGPUError("No devices provided", GPUErrorInvalidParameter)
	}

	if len(data) == 0 {
		return 0, NewGPUError("Cannot sum empty array", GPUErrorInvalidParameter)
	}

	// For multi-GPU, distribute work across devices
	if len(devices) == 1 {
		return devices[0].ExecuteSum(data)
	}

	// Simple distribution: split data across devices
	chunkSize := len(data) / len(devices)
	results := make(chan float64, len(devices))
	errors := make(chan error, len(devices))

	for i, device := range devices {
		go func(dev HardwareGPUDevice, start int) {
			end := start + chunkSize
			if start >= len(data) {
				results <- 0.0
				errors <- nil
				return
			}
			if end > len(data) {
				end = len(data)
			}

			chunk := data[start:end]
			sum, err := dev.ExecuteSum(chunk)
			results <- sum
			errors <- err
		}(device, i*chunkSize)
	}

	// Collect results
	totalSum := 0.0
	for i := 0; i < len(devices); i++ {
		sum := <-results
		err := <-errors
		if err != nil {
			return 0, fmt.Errorf("device %d failed: %v", i, err)
		}
		totalSum += sum
	}

	return totalSum, nil
}

// Utility functions for build-specific features

func GPUBackendsAvailable() []string {
	var backends []string

	// Check what was compiled in
	if _, err := DetectCUDADevices(); err == nil {
		backends = append(backends, "CUDA")
	}

	if _, err := DetectOpenCLDevices(); err == nil {
		backends = append(backends, "OpenCL")
	}

	if len(backends) == 0 {
		backends = append(backends, "None (CPU fallback)")
	}

	return backends
}

func IsGPUAvailable() bool {
	_, err := NewHardwareGPUManager()
	return err == nil
}
