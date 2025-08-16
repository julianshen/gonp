package internal

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"sync/atomic"
	"time"
)

// DebugConfig holds debug configuration settings
type DebugConfig struct {
	Enabled       bool
	LogLevel      DebugLevel
	ProfileMemory bool
	ProfileTiming bool
	LogToFile     bool
	LogFile       string
}

// DebugLevel represents different debug levels
type DebugLevel int

const (
	DebugOff DebugLevel = iota
	DebugLevelError
	DebugLevelWarn
	DebugLevelInfo
	DebugLevelVerbose
)

// String returns string representation of debug level
func (d DebugLevel) String() string {
	switch d {
	case DebugOff:
		return "OFF"
	case DebugLevelError:
		return "ERROR"
	case DebugLevelWarn:
		return "WARN"
	case DebugLevelInfo:
		return "INFO"
	case DebugLevelVerbose:
		return "VERBOSE"
	default:
		return "UNKNOWN"
	}
}

// Global debug configuration
var globalDebugConfig = &DebugConfig{
	Enabled:       false,
	LogLevel:      DebugOff,
	ProfileMemory: false,
	ProfileTiming: false,
	LogToFile:     false,
	LogFile:       "gonp_debug.log",
}

// Statistics counters
var (
	arrayCreationCount   int64
	operationCount       int64
	memoryAllocCount     int64
	totalMemoryAllocated int64
	errorCount           int64
)

// Initialize debug mode from environment variables at startup
func init() {
	initDebugFromEnv()
}

// initDebugFromEnv initializes debug configuration from environment variables
func initDebugFromEnv() {
	// Check if debug mode is enabled
	if os.Getenv("GONP_DEBUG") == "1" || os.Getenv("GONP_DEBUG") == "true" {
		globalDebugConfig.Enabled = true
	}

	// Set debug level
	if levelStr := os.Getenv("GONP_DEBUG_LEVEL"); levelStr != "" {
		switch levelStr {
		case "ERROR", "error":
			globalDebugConfig.LogLevel = DebugLevelError
		case "WARN", "warn":
			globalDebugConfig.LogLevel = DebugLevelWarn
		case "INFO", "info":
			globalDebugConfig.LogLevel = DebugLevelInfo
		case "VERBOSE", "verbose":
			globalDebugConfig.LogLevel = DebugLevelVerbose
		}
	}

	// Enable memory profiling
	if os.Getenv("GONP_PROFILE_MEMORY") == "1" {
		globalDebugConfig.ProfileMemory = true
	}

	// Enable timing profiling
	if os.Getenv("GONP_PROFILE_TIMING") == "1" {
		globalDebugConfig.ProfileTiming = true
	}

	// Set log file
	if logFile := os.Getenv("GONP_LOG_FILE"); logFile != "" {
		globalDebugConfig.LogToFile = true
		globalDebugConfig.LogFile = logFile
	}
}

// SetDebugConfig updates the global debug configuration
func SetDebugConfig(config *DebugConfig) {
	*globalDebugConfig = *config
}

// GetDebugConfig returns a copy of the current debug configuration
func GetDebugConfig() DebugConfig {
	return *globalDebugConfig
}

// IsDebugEnabled returns true if debug mode is enabled
func IsDebugEnabled() bool {
	return globalDebugConfig.Enabled
}

// LogLevel returns the current log level
func LogLevel() DebugLevel {
	return globalDebugConfig.LogLevel
}

// Debug logging functions

// DebugLog logs a message at the specified level
func DebugLog(level DebugLevel, format string, args ...interface{}) {
	if !globalDebugConfig.Enabled || level > globalDebugConfig.LogLevel {
		return
	}

	message := fmt.Sprintf("[%s] %s", level.String(), fmt.Sprintf(format, args...))

	if globalDebugConfig.LogToFile {
		logToFile(message)
	} else {
		log.Println(message)
	}
}

// DebugError logs an error message
func DebugError(format string, args ...interface{}) {
	DebugLog(DebugLevelError, format, args...)
	atomic.AddInt64(&errorCount, 1)
}

// DebugWarn logs a warning message
func DebugWarn(format string, args ...interface{}) {
	DebugLog(DebugLevelWarn, format, args...)
}

// DebugInfo logs an info message
func DebugInfo(format string, args ...interface{}) {
	DebugLog(DebugLevelInfo, format, args...)
}

// DebugVerbose logs a verbose message
func DebugVerbose(format string, args ...interface{}) {
	DebugLog(DebugLevelVerbose, format, args...)
}

// Performance profiling functions

// ProfilerContext holds profiling information
type ProfilerContext struct {
	Name      string
	StartTime time.Time
	StartMem  runtime.MemStats
}

// StartProfiler begins profiling an operation
func StartProfiler(name string) *ProfilerContext {
	if !globalDebugConfig.Enabled || (!globalDebugConfig.ProfileTiming && !globalDebugConfig.ProfileMemory) {
		return nil
	}

	ctx := &ProfilerContext{
		Name:      name,
		StartTime: time.Now(),
	}

	if globalDebugConfig.ProfileMemory {
		runtime.ReadMemStats(&ctx.StartMem)
	}

	DebugVerbose("Started profiling: %s", name)
	return ctx
}

// EndProfiler ends profiling and logs results
func (ctx *ProfilerContext) EndProfiler() {
	if ctx == nil || !globalDebugConfig.Enabled {
		return
	}

	var timingInfo, memoryInfo string

	if globalDebugConfig.ProfileTiming {
		duration := time.Since(ctx.StartTime)
		timingInfo = fmt.Sprintf("Duration: %v", duration)
	}

	if globalDebugConfig.ProfileMemory {
		var endMem runtime.MemStats
		runtime.ReadMemStats(&endMem)

		memDiff := endMem.Alloc - ctx.StartMem.Alloc
		memoryInfo = fmt.Sprintf("Memory: %d bytes", memDiff)

		if memDiff > 0 {
			atomic.AddInt64(&totalMemoryAllocated, int64(memDiff))
		}
	}

	if timingInfo != "" && memoryInfo != "" {
		DebugInfo("Profiling result [%s]: %s, %s", ctx.Name, timingInfo, memoryInfo)
	} else if timingInfo != "" {
		DebugInfo("Profiling result [%s]: %s", ctx.Name, timingInfo)
	} else if memoryInfo != "" {
		DebugInfo("Profiling result [%s]: %s", ctx.Name, memoryInfo)
	}
}

// Statistics functions

// IncrementArrayCreations increments the array creation counter
func IncrementArrayCreations() {
	if globalDebugConfig.Enabled {
		atomic.AddInt64(&arrayCreationCount, 1)
	}
}

// IncrementOperations increments the operation counter
func IncrementOperations() {
	if globalDebugConfig.Enabled {
		atomic.AddInt64(&operationCount, 1)
	}
}

// IncrementMemoryAllocs increments the memory allocation counter
func IncrementMemoryAllocs(size int64) {
	if globalDebugConfig.Enabled {
		atomic.AddInt64(&memoryAllocCount, 1)
		atomic.AddInt64(&totalMemoryAllocated, size)
	}
}

// GetStatistics returns current debug statistics
func GetStatistics() map[string]int64 {
	return map[string]int64{
		"array_creations":    atomic.LoadInt64(&arrayCreationCount),
		"operations":         atomic.LoadInt64(&operationCount),
		"memory_allocs":      atomic.LoadInt64(&memoryAllocCount),
		"total_memory_bytes": atomic.LoadInt64(&totalMemoryAllocated),
		"errors":             atomic.LoadInt64(&errorCount),
	}
}

// ResetStatistics resets all debug statistics
func ResetStatistics() {
	atomic.StoreInt64(&arrayCreationCount, 0)
	atomic.StoreInt64(&operationCount, 0)
	atomic.StoreInt64(&memoryAllocCount, 0)
	atomic.StoreInt64(&totalMemoryAllocated, 0)
	atomic.StoreInt64(&errorCount, 0)
}

// PrintStatistics prints current statistics to stdout
func PrintStatistics() {
	if !globalDebugConfig.Enabled {
		fmt.Println("Debug mode is disabled")
		return
	}

	stats := GetStatistics()
	fmt.Println("=== GoNP Debug Statistics ===")
	fmt.Printf("Array Creations: %d\n", stats["array_creations"])
	fmt.Printf("Operations: %d\n", stats["operations"])
	fmt.Printf("Memory Allocations: %d\n", stats["memory_allocs"])
	fmt.Printf("Total Memory Allocated: %d bytes (%.2f MB)\n",
		stats["total_memory_bytes"],
		float64(stats["total_memory_bytes"])/1024/1024)
	fmt.Printf("Errors: %d\n", stats["errors"])
	fmt.Println("============================")
}

// Utility functions

// logToFile writes a message to the debug log file
func logToFile(message string) {
	// Use restrictive permissions for debug logs
	file, err := os.OpenFile(globalDebugConfig.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		log.Printf("Failed to open debug log file: %v", err)
		return
	}
	defer file.Close()

	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Fprintf(file, "[%s] %s\n", timestamp, message)
}

// EnableDebugMode enables debug mode with default settings
func EnableDebugMode() {
	globalDebugConfig.Enabled = true
	globalDebugConfig.LogLevel = DebugLevelInfo
	DebugInfo("Debug mode enabled")
}

// DisableDebugMode disables debug mode
func DisableDebugMode() {
	DebugInfo("Debug mode disabled")
	globalDebugConfig.Enabled = false
}

// EnableMemoryProfiling enables memory profiling
func EnableMemoryProfiling() {
	globalDebugConfig.ProfileMemory = true
	DebugInfo("Memory profiling enabled")
}

// EnableTimingProfiling enables timing profiling
func EnableTimingProfiling() {
	globalDebugConfig.ProfileTiming = true
	DebugInfo("Timing profiling enabled")
}

// SetLogLevel sets the debug log level
func SetLogLevel(level DebugLevel) {
	globalDebugConfig.LogLevel = level
	DebugInfo("Log level set to: %s", level.String())
}

// Environmental variable documentation for users:
//
// GONP_DEBUG=1                 - Enable debug mode
// GONP_DEBUG_LEVEL=INFO        - Set log level (ERROR, WARN, INFO, VERBOSE)
// GONP_PROFILE_MEMORY=1        - Enable memory profiling
// GONP_PROFILE_TIMING=1        - Enable timing profiling
// GONP_LOG_FILE=debug.log      - Log to file instead of stdout
