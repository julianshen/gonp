//go:build linux

package io

import (
	"fmt"
	"os"
	"syscall"
)

// PrefetchFile prefetches file content into system cache (Linux implementation)
func PrefetchFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	fd := file.Fd()
	stat, err := file.Stat()
	if err != nil {
		return err
	}

	// POSIX_FADV_SEQUENTIAL = 2
	_, _, errno := syscall.Syscall6(
		syscall.SYS_FADVISE64,
		uintptr(fd),
		0,
		uintptr(stat.Size()),
		uintptr(2),
		0, 0,
	)
	if errno != 0 {
		return fmt.Errorf("fadvise failed: %v", errno)
	}
	return nil
}
