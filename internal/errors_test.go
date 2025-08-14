package internal

import (
	"errors"
	"strings"
	"testing"
)

// TestStructuredErrorHierarchy tests the comprehensive error hierarchy system
func TestStructuredErrorHierarchy(t *testing.T) {

	t.Run("Error classification and categorization", func(t *testing.T) {
		// Test error category creation
		memoryCategory := NewErrorCategory("MEMORY", "Memory management errors")
		if memoryCategory.Code() != "MEMORY" {
			t.Errorf("Expected category code 'MEMORY', got '%s'", memoryCategory.Code())
		}

		computeCategory := NewErrorCategory("COMPUTE", "Computational errors")
		if computeCategory.Description() != "Computational errors" {
			t.Errorf("Expected description 'Computational errors', got '%s'", computeCategory.Description())
		}

		// Test error type creation within categories
		oomError := NewErrorType(memoryCategory, "OUT_OF_MEMORY", "System out of memory")
		if oomError.Category() != memoryCategory {
			t.Error("Error type should belong to correct category")
		}

		if oomError.Code() != "MEMORY.OUT_OF_MEMORY" {
			t.Errorf("Expected full code 'MEMORY.OUT_OF_MEMORY', got '%s'", oomError.Code())
		}

		// Test severity levels
		criticalSeverity := SeverityCritical
		if criticalSeverity.String() != "CRITICAL" {
			t.Errorf("Expected severity 'CRITICAL', got '%s'", criticalSeverity.String())
		}

		// Test error instance creation
		err := NewStructuredError(oomError, SeverityCritical, "Failed to allocate 1GB memory block")
		if err.Type() != oomError {
			t.Error("Structured error should have correct type")
		}

		if err.Severity() != SeverityCritical {
			t.Error("Structured error should have correct severity")
		}

		if !strings.Contains(err.Error(), "Failed to allocate 1GB memory block") {
			t.Error("Error message should contain original message")
		}
	})

	t.Run("Error context and metadata attachment", func(t *testing.T) {
		category := NewErrorCategory("ARRAY", "Array operation errors")
		errorType := NewErrorType(category, "DIMENSION_MISMATCH", "Array dimensions don't match")

		// Create error with context
		err := NewStructuredError(errorType, SeverityError, "Cannot multiply arrays")
		err.WithContext("operation", "matrix_multiplication")
		err.WithContext("array1_shape", "[3, 4]")
		err.WithContext("array2_shape", "[2, 3]")
		err.WithContext("function", "array.MatMul")
		err.WithContext("line", 245)

		// Test context retrieval
		op, exists := err.GetContext("operation")
		if !exists || op != "matrix_multiplication" {
			t.Error("Context should be retrievable")
		}

		shape1, exists := err.GetContext("array1_shape")
		if !exists || shape1 != "[3, 4]" {
			t.Error("Array shape context should be stored correctly")
		}

		// Test context in error message
		fullMsg := err.DetailedError()
		if !strings.Contains(fullMsg, "matrix_multiplication") {
			t.Error("Detailed error should include context")
		}

		if !strings.Contains(fullMsg, "array.MatMul:245") {
			t.Error("Detailed error should include function and line context")
		}
	})

	t.Run("Error chaining and cause tracking", func(t *testing.T) {
		category := NewErrorCategory("IO", "Input/output errors")
		fileError := NewErrorType(category, "FILE_NOT_FOUND", "File not found")
		parseError := NewErrorType(category, "PARSE_FAILED", "Failed to parse data")

		// Create chain of errors
		rootCause := NewStructuredError(fileError, SeverityError, "Cannot open 'data.csv'")
		rootCause.WithContext("filepath", "/tmp/data.csv")
		rootCause.WithContext("permissions", "755")

		wrappedError := NewStructuredError(parseError, SeverityError, "Failed to read CSV data")
		wrappedError.WithCause(rootCause)
		wrappedError.WithContext("parser", "csv_reader")

		finalError := NewStructuredError(parseError, SeverityCritical, "Dataset loading failed")
		finalError.WithCause(wrappedError)
		finalError.WithContext("dataset", "training_data")

		// Test cause chain traversal
		if finalError.RootCause() != rootCause {
			t.Error("Root cause should be retrievable through chain")
		}

		causes := finalError.CauseChain()
		if len(causes) != 2 {
			t.Errorf("Expected 2 causes in chain, got %d", len(causes))
		}

		// Test error unwrapping
		if !errors.Is(finalError, rootCause) {
			t.Error("errors.Is should work with cause chain")
		}
	})

	t.Run("Error recovery suggestions and actions", func(t *testing.T) {
		category := NewErrorCategory("VALIDATION", "Input validation errors")
		rangeError := NewErrorType(category, "VALUE_OUT_OF_RANGE", "Value outside acceptable range")

		err := NewStructuredError(rangeError, SeverityWarning, "Array index out of bounds")
		err.WithContext("index", 15)
		err.WithContext("array_length", 10)
		err.WithContext("operation", "element_access")

		// Add recovery suggestions
		err.AddRecoverySuggestion("Check array bounds before accessing elements")
		err.AddRecoverySuggestion("Use array.Len() to verify array size")
		err.AddRecoverySuggestion("Consider using bounds checking: if index < array.Len()")

		// Add recovery actions
		err.AddRecoveryAction(RecoveryAction{
			ActionType:  ActionTypeRetry,
			Description: "Retry with corrected index",
			AutoApply:   false,
			Handler: func() error {
				// Mock handler - would normally fix the issue
				return nil
			},
		})

		err.AddRecoveryAction(RecoveryAction{
			ActionType:  ActionTypeFallback,
			Description: "Use default value for out-of-bounds access",
			AutoApply:   true,
			Handler: func() error {
				return nil
			},
		})

		// Test suggestions retrieval
		suggestions := err.GetRecoverySuggestions()
		if len(suggestions) != 3 {
			t.Errorf("Expected 3 recovery suggestions, got %d", len(suggestions))
		}

		if !strings.Contains(suggestions[0], "Check array bounds") {
			t.Error("First suggestion should mention bounds checking")
		}

		// Test recovery actions
		actions := err.GetRecoveryActions()
		if len(actions) != 2 {
			t.Errorf("Expected 2 recovery actions, got %d", len(actions))
		}

		retryAction := actions[0]
		if retryAction.ActionType != ActionTypeRetry {
			t.Error("First action should be retry type")
		}

		fallbackAction := actions[1]
		if !fallbackAction.AutoApply {
			t.Error("Fallback action should be auto-apply")
		}

		// Test action execution
		if err := fallbackAction.Handler(); err != nil {
			t.Errorf("Recovery action should execute without error: %v", err)
		}
	})

	t.Run("Error serialization and deserialization", func(t *testing.T) {
		category := NewErrorCategory("NETWORK", "Network communication errors")
		timeoutError := NewErrorType(category, "TIMEOUT", "Operation timed out")

		originalError := NewStructuredError(timeoutError, SeverityError, "Connection timeout after 30s")
		originalError.WithContext("host", "example.com")
		originalError.WithContext("port", 443)
		originalError.WithContext("timeout_ms", 30000)
		originalError.AddRecoverySuggestion("Increase timeout duration")
		originalError.AddRecoverySuggestion("Check network connectivity")

		// Test JSON serialization
		jsonData, err := originalError.ToJSON()
		if err != nil {
			t.Fatalf("Failed to serialize error to JSON: %v", err)
		}

		if !strings.Contains(string(jsonData), "NETWORK.TIMEOUT") {
			t.Error("JSON should contain full error code")
		}

		if !strings.Contains(string(jsonData), "example.com") {
			t.Error("JSON should contain context data")
		}

		// Test JSON deserialization
		deserializedError, err := StructuredErrorFromJSON(jsonData)
		if err != nil {
			t.Fatalf("Failed to deserialize error from JSON: %v", err)
		}

		if deserializedError.Type().Code() != "NETWORK.TIMEOUT" {
			t.Error("Deserialized error should have same type")
		}

		if deserializedError.Severity() != SeverityError {
			t.Error("Deserialized error should have same severity")
		}

		host, exists := deserializedError.GetContext("host")
		if !exists || host != "example.com" {
			t.Error("Deserialized error should preserve context")
		}

		suggestions := deserializedError.GetRecoverySuggestions()
		if len(suggestions) != 2 {
			t.Error("Deserialized error should preserve recovery suggestions")
		}

		// Test string representation formats
		shortForm := originalError.Error()
		if len(shortForm) == 0 {
			t.Error("Short form should not be empty")
		}

		detailedForm := originalError.DetailedError()
		if len(detailedForm) <= len(shortForm) {
			t.Error("Detailed form should be longer than short form")
		}

		debugForm := originalError.DebugString()
		if !strings.Contains(debugForm, "Context:") {
			t.Error("Debug form should include context section")
		}

		if !strings.Contains(debugForm, "Recovery:") {
			t.Error("Debug form should include recovery section")
		}
	})
}

// TestErrorAggregationAndBatching tests error collection and batch processing
func TestErrorAggregationAndBatching(t *testing.T) {

	t.Run("Error collector with multiple errors", func(t *testing.T) {
		collector := NewErrorCollector()

		// Collect multiple errors
		category := NewErrorCategory("COMPUTE", "Computational errors")
		divByZero := NewErrorType(category, "DIVISION_BY_ZERO", "Division by zero")
		overflow := NewErrorType(category, "NUMERIC_OVERFLOW", "Numeric overflow")

		err1 := NewStructuredError(divByZero, SeverityError, "Cannot divide by zero")
		err1.WithContext("operation", "vector_division")
		err1.WithContext("element_index", 5)

		err2 := NewStructuredError(overflow, SeverityWarning, "Value exceeds maximum")
		err2.WithContext("value", "1.79769e+309")
		err2.WithContext("max_value", "1.79769e+308")

		err3 := NewStructuredError(divByZero, SeverityCritical, "Matrix inversion failed")
		err3.WithContext("matrix_size", "1000x1000")
		err3.WithContext("condition_number", "inf")

		collector.Add(err1)
		collector.Add(err2)
		collector.Add(err3)

		// Test error collection
		allErrors := collector.GetAll()
		if len(allErrors) != 3 {
			t.Errorf("Expected 3 collected errors, got %d", len(allErrors))
		}

		// Test error filtering by severity
		criticalErrors := collector.GetBySeverity(SeverityCritical)
		if len(criticalErrors) != 1 {
			t.Errorf("Expected 1 critical error, got %d", len(criticalErrors))
		}

		// Test error filtering by type
		divisionErrors := collector.GetByType(divByZero)
		if len(divisionErrors) != 2 {
			t.Errorf("Expected 2 division errors, got %d", len(divisionErrors))
		}

		// Test error count by category
		computeCount := collector.CountByCategory(category)
		if computeCount != 3 {
			t.Errorf("Expected 3 compute errors, got %d", computeCount)
		}

		// Test error summary
		summary := collector.GetSummary()
		if summary.TotalErrors != 3 {
			t.Error("Summary should show total error count")
		}

		if summary.CriticalCount != 1 {
			t.Error("Summary should show critical error count")
		}

		if len(summary.ByCategory) == 0 {
			t.Error("Summary should include category breakdown")
		}

		// Test batch error creation
		batchError := collector.ToBatchError("Multiple computational errors occurred")
		if batchError.ErrorCount() != 3 {
			t.Error("Batch error should contain all collected errors")
		}

		if batchError.HighestSeverity() != SeverityCritical {
			t.Error("Batch error should report highest severity")
		}
	})

	t.Run("Error rate limiting and throttling", func(t *testing.T) {
		limiter := NewErrorRateLimiter(3, 1000) // 3 errors per 1000ms

		category := NewErrorCategory("RATE_TEST", "Rate limiting test")
		testError := NewErrorType(category, "TEST_ERROR", "Test error for rate limiting")

		// Test rate limiting
		for i := 0; i < 5; i++ {
			err := NewStructuredError(testError, SeverityError, "Test error")
			err.WithContext("iteration", i)

			allowed := limiter.ShouldReport(err)
			if i < 3 && !allowed {
				t.Errorf("Error %d should be allowed", i)
			}
			if i >= 3 && allowed {
				t.Errorf("Error %d should be rate limited", i)
			}
		}

		// Test rate limiter stats
		stats := limiter.GetStats()
		if stats.TotalErrors != 5 {
			t.Errorf("Expected 5 total errors, got %d", stats.TotalErrors)
		}

		if stats.AllowedErrors != 3 {
			t.Errorf("Expected 3 allowed errors, got %d", stats.AllowedErrors)
		}

		if stats.ThrottledErrors != 2 {
			t.Errorf("Expected 2 throttled errors, got %d", stats.ThrottledErrors)
		}
	})
}

// TestErrorReportingAndLogging tests error reporting system integration
func TestErrorReportingAndLogging(t *testing.T) {

	t.Run("Error reporter with multiple handlers", func(t *testing.T) {
		reporter := NewErrorReporter()

		// Add mock handlers
		loggedErrors := make([]StructuredError, 0)
		sentEmails := make([]StructuredError, 0)

		logHandler := func(err StructuredError) error {
			loggedErrors = append(loggedErrors, err)
			return nil
		}

		emailHandler := func(err StructuredError) error {
			if err.Severity() >= SeverityError {
				sentEmails = append(sentEmails, err)
			}
			return nil
		}

		reporter.AddHandler("logger", logHandler)
		reporter.AddHandler("email_alerts", emailHandler)

		// Test error reporting
		category := NewErrorCategory("SERVICE", "Service errors")
		serviceError := NewErrorType(category, "SERVICE_DOWN", "Service unavailable")

		err1 := NewStructuredError(serviceError, SeverityWarning, "Service responding slowly")
		err2 := NewStructuredError(serviceError, SeverityError, "Service completely down")
		err3 := NewStructuredError(serviceError, SeverityCritical, "Service crashed")

		reporter.Report(err1)
		reporter.Report(err2)
		reporter.Report(err3)

		// Verify handler calls
		if len(loggedErrors) != 3 {
			t.Errorf("Expected 3 logged errors, got %d", len(loggedErrors))
		}

		if len(sentEmails) != 2 {
			t.Errorf("Expected 2 email alerts (error+ severity), got %d", len(sentEmails))
		}

		// Test handler failure handling
		failingHandler := func(err StructuredError) error {
			return errors.New("handler failed")
		}

		reporter.AddHandler("failing_handler", failingHandler)

		// Should not panic on handler failure
		err4 := NewStructuredError(serviceError, SeverityInfo, "Service info")
		if err := reporter.Report(err4); err == nil {
			t.Error("Reporter should return error when handler fails")
		}

		// Test async reporting
		asyncReporter := NewAsyncErrorReporter(10) // Buffer size 10
		defer asyncReporter.Close()

		processedAsync := make([]StructuredError, 0)
		asyncHandler := func(err StructuredError) error {
			processedAsync = append(processedAsync, err)
			return nil
		}

		asyncReporter.AddHandler("async_logger", asyncHandler)

		// Send multiple errors async
		for i := 0; i < 5; i++ {
			asyncErr := NewStructuredError(serviceError, SeverityInfo, "Async test error")
			asyncErr.WithContext("iteration", i)
			asyncReporter.ReportAsync(asyncErr)
		}

		// Wait for async processing
		asyncReporter.Flush()

		if len(processedAsync) != 5 {
			t.Errorf("Expected 5 async processed errors, got %d", len(processedAsync))
		}
	})
}

// TestProductionErrorHandling tests error handling in production scenarios
func TestProductionErrorHandling(t *testing.T) {

	t.Run("Memory management error integration", func(t *testing.T) {
		// Test integration with existing memory pool errors
		if !IsOutOfMemoryError(ErrOutOfMemory) {
			t.Error("ErrOutOfMemory should be recognized as OOM error")
		}

		if !IsInvalidSizeError(ErrInvalidSize) {
			t.Error("ErrInvalidSize should be recognized as invalid size error")
		}

		// Test structured error wrapping of existing errors
		wrappedOOM := WrapError(ErrOutOfMemory, "Failed to allocate vector")
		if wrappedOOM.Type().Code() != "MEMORY.OUT_OF_MEMORY" {
			t.Error("Wrapped OOM error should have correct structured type")
		}

		// Test error context enhancement
		enhanced := EnhanceError(ErrInvalidSize, map[string]interface{}{
			"requested_size": -1,
			"function":       "array.NewArray",
			"line":           123,
		})

		size, exists := enhanced.GetContext("requested_size")
		if !exists || size != -1 {
			t.Error("Enhanced error should include context")
		}
	})

	t.Run("Error handling performance under load", func(t *testing.T) {
		// Test error creation performance
		category := NewErrorCategory("PERF_TEST", "Performance test errors")
		errorType := NewErrorType(category, "LOAD_TEST", "Load test error")

		errorCount := 1000
		start := make([]StructuredError, 0, errorCount)

		// Measure error creation time
		for i := 0; i < errorCount; i++ {
			err := NewStructuredError(errorType, SeverityInfo, "Load test error")
			err.WithContext("iteration", i)
			err.WithContext("timestamp", "2023-01-01T00:00:00Z")
			start = append(start, err)
		}

		if len(start) != errorCount {
			t.Errorf("Should create %d errors efficiently", errorCount)
		}

		// Test collector performance
		collector := NewErrorCollector()
		for _, err := range start {
			collector.Add(err)
		}

		summary := collector.GetSummary()
		if summary.TotalErrors != errorCount {
			t.Error("Collector should handle large number of errors efficiently")
		}

		// Test serialization performance
		batchError := collector.ToBatchError("Performance test batch")
		jsonData, err := batchError.ToJSON()
		if err != nil {
			t.Errorf("Should serialize large error batch: %v", err)
		}

		if len(jsonData) == 0 {
			t.Error("Serialized data should not be empty")
		}
	})
}
