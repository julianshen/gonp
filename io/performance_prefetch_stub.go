//go:build !linux

package io

// PrefetchFile is a no-op on non-Linux platforms.
func PrefetchFile(filename string) error { return nil }
