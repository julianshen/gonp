package internal

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestMonitoringAndObservability tests the comprehensive monitoring and observability system
func TestMonitoringAndObservability(t *testing.T) {

	t.Run("Metrics collection and aggregation", func(t *testing.T) {
		// Test metrics registry creation
		registry := NewMetricsRegistry()
		if registry == nil {
			t.Error("Should create metrics registry")
		}

		// Test counter metrics
		counter := registry.NewCounter("gonp_operations_total", "Total number of operations")
		counter.Inc()
		counter.Add(5)

		if counter.Value() != 6 {
			t.Errorf("Expected counter value 6, got %f", counter.Value())
		}

		// Test gauge metrics
		gauge := registry.NewGauge("gonp_memory_usage_bytes", "Current memory usage in bytes")
		gauge.Set(1048576) // 1MB

		if gauge.Value() != 1048576 {
			t.Errorf("Expected gauge value 1048576, got %f", gauge.Value())
		}

		gauge.Add(1024)
		if gauge.Value() != 1049600 {
			t.Errorf("Expected gauge value 1049600 after add, got %f", gauge.Value())
		}

		// Test histogram metrics
		histogram := registry.NewHistogram("gonp_operation_duration_seconds", "Operation duration distribution")
		histogram.Observe(0.1)
		histogram.Observe(0.5)
		histogram.Observe(1.2)

		count := histogram.Count()
		if count != 3 {
			t.Errorf("Expected histogram count 3, got %d", count)
		}

		sum := histogram.Sum()
		if sum != 1.8 {
			t.Errorf("Expected histogram sum 1.8, got %f", sum)
		}

		// Test summary metrics with quantiles
		summary := registry.NewSummary("gonp_request_size_bytes", "Request size distribution")
		for i := 0; i < 100; i++ {
			summary.Observe(float64(i))
		}

		p50 := summary.Quantile(0.5)
		p95 := summary.Quantile(0.95)
		p99 := summary.Quantile(0.99)

		if p50 < 45 || p50 > 55 {
			t.Errorf("Expected p50 around 50, got %f", p50)
		}

		if p95 < 90 || p95 > 99 {
			t.Errorf("Expected p95 around 95, got %f", p95)
		}

		if p99 < 95 || p99 > 99 {
			t.Errorf("Expected p99 around 99, got %f", p99)
		}
	})

	t.Run("Health check system", func(t *testing.T) {
		// Test health checker creation
		checker := NewHealthChecker()
		if checker == nil {
			t.Error("Should create health checker")
		}

		// Test individual health checks
		memoryCheck := &HealthCheck{
			Name:        "memory",
			Description: "Memory usage check",
			CheckFunc: func(ctx context.Context) HealthStatus {
				// Simulate memory check
				return HealthStatus{
					Status:    HealthStatusHealthy,
					Message:   "Memory usage within limits",
					Timestamp: time.Now(),
					Details: map[string]interface{}{
						"used_bytes":  1048576,
						"limit_bytes": 1073741824,
					},
				}
			},
			Timeout:  5 * time.Second,
			Critical: true,
		}

		checker.RegisterCheck(memoryCheck)

		cpuCheck := &HealthCheck{
			Name:        "cpu",
			Description: "CPU usage check",
			CheckFunc: func(ctx context.Context) HealthStatus {
				return HealthStatus{
					Status:    HealthStatusHealthy,
					Message:   "CPU usage normal",
					Timestamp: time.Now(),
					Details: map[string]interface{}{
						"usage_percent": 25.5,
						"load_average":  1.2,
					},
				}
			},
			Timeout:  3 * time.Second,
			Critical: false,
		}

		checker.RegisterCheck(cpuCheck)

		// Test overall health status
		ctx := context.Background()
		overallStatus := checker.CheckHealth(ctx)

		if overallStatus.Status != HealthStatusHealthy {
			t.Errorf("Expected healthy status, got %s", overallStatus.Status)
		}

		if len(overallStatus.Checks) != 2 {
			t.Errorf("Expected 2 individual checks, got %d", len(overallStatus.Checks))
		}

		// Test health check with failure
		failingCheck := &HealthCheck{
			Name:        "database",
			Description: "Database connectivity check",
			CheckFunc: func(ctx context.Context) HealthStatus {
				return HealthStatus{
					Status:    HealthStatusUnhealthy,
					Message:   "Database connection failed",
					Timestamp: time.Now(),
					Error:     "connection timeout",
				}
			},
			Timeout:  2 * time.Second,
			Critical: true,
		}

		checker.RegisterCheck(failingCheck)

		failedStatus := checker.CheckHealth(ctx)
		if failedStatus.Status != HealthStatusUnhealthy {
			t.Error("Overall status should be unhealthy when critical check fails")
		}

		if len(failedStatus.Checks) != 3 {
			t.Errorf("Expected 3 individual checks, got %d", len(failedStatus.Checks))
		}
	})

	t.Run("Performance monitoring and profiling", func(t *testing.T) {
		// Test performance monitor creation
		monitor := NewPerformanceMonitor()
		if monitor == nil {
			t.Error("Should create performance monitor")
		}

		// Test operation timing
		timer := monitor.StartTimer("matrix_multiplication")
		time.Sleep(10 * time.Millisecond) // Simulate work
		duration := timer.Stop()

		if duration < 10*time.Millisecond {
			t.Error("Timer should measure at least 10ms")
		}

		// Test memory tracking
		monitor.RecordMemoryAllocation("arrays", 1024*1024) // 1MB
		monitor.RecordMemoryAllocation("arrays", 512*1024)  // 512KB

		memStats := monitor.GetMemoryStats("arrays")
		if memStats.TotalAllocated < 1024*1024+512*1024 {
			t.Error("Should track total memory allocations")
		}

		if memStats.AllocationCount != 2 {
			t.Errorf("Expected 2 allocations, got %d", memStats.AllocationCount)
		}

		// Test operation profiling
		for i := 0; i < 100; i++ {
			timer := monitor.StartTimer("array_operation")
			time.Sleep(time.Duration(i) * time.Microsecond) // Variable duration
			timer.Stop()
		}

		profile := monitor.GetOperationProfile("array_operation")
		if profile.Count != 100 {
			t.Errorf("Expected 100 operations, got %d", profile.Count)
		}

		if profile.AverageDuration == 0 {
			t.Error("Should calculate average duration")
		}

		if profile.MaxDuration < profile.MinDuration {
			t.Error("Max duration should be >= min duration")
		}

		// Test concurrent monitoring
		var wg sync.WaitGroup
		concurrentOps := 50

		for i := 0; i < concurrentOps; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				timer := monitor.StartTimer("concurrent_op")
				time.Sleep(time.Millisecond)
				timer.Stop()
			}(i)
		}

		wg.Wait()

		concurrentProfile := monitor.GetOperationProfile("concurrent_op")
		if concurrentProfile.Count != int64(concurrentOps) {
			t.Errorf("Expected %d concurrent operations, got %d", concurrentOps, concurrentProfile.Count)
		}
	})

	t.Run("Distributed tracing and spans", func(t *testing.T) {
		// Test tracer creation
		tracer := NewTracer("gonp-service")
		if tracer == nil {
			t.Error("Should create tracer")
		}

		// Test span creation and lifecycle
		ctx := context.Background()
		parentSpan := tracer.StartSpan(ctx, "matrix_operations")

		if parentSpan.TraceID() == "" {
			t.Error("Span should have trace ID")
		}

		if parentSpan.SpanID() == "" {
			t.Error("Span should have span ID")
		}

		// Test nested spans
		childCtx := tracer.ContextWithSpan(ctx, parentSpan)
		childSpan := tracer.StartSpan(childCtx, "matrix_multiply")

		if childSpan.ParentSpanID() != parentSpan.SpanID() {
			t.Error("Child span should reference parent span")
		}

		if childSpan.TraceID() != parentSpan.TraceID() {
			t.Error("Child span should share parent's trace ID")
		}

		// Test span attributes and events
		childSpan.SetAttribute("matrix.size", "1000x1000")
		childSpan.SetAttribute("algorithm", "strassen")
		childSpan.AddEvent("computation_started", map[string]interface{}{
			"timestamp": time.Now().UnixNano(),
			"threads":   8,
		})

		attrs := childSpan.Attributes()
		if attrs["matrix.size"] != "1000x1000" {
			t.Error("Should store span attributes")
		}

		events := childSpan.Events()
		if len(events) != 1 {
			t.Errorf("Expected 1 event, got %d", len(events))
		}

		// Test span status
		childSpan.SetStatus(SpanStatusOK, "Matrix multiplication completed successfully")
		childSpan.Finish()

		if childSpan.Duration() <= 0 {
			t.Error("Finished span should have duration")
		}

		// Test error handling
		errorSpan := tracer.StartSpan(ctx, "failing_operation")
		errorSpan.RecordError("division by zero", map[string]interface{}{
			"error_code": "MATH_ERROR",
			"severity":   "high",
		})
		errorSpan.SetStatus(SpanStatusError, "Operation failed due to division by zero")
		errorSpan.Finish()

		if errorSpan.Status().Code != SpanStatusError {
			t.Error("Error span should have error status")
		}

		parentSpan.Finish()
	})

	t.Run("Alerting and notification system", func(t *testing.T) {
		// Test alert manager creation
		alertManager := NewAlertManager()
		if alertManager == nil {
			t.Error("Should create alert manager")
		}

		// Test alert rule definition
		memoryRule := &AlertRule{
			Name:        "high_memory_usage",
			Description: "Memory usage exceeds threshold",
			Condition: func(metrics map[string]float64) bool {
				memUsage, exists := metrics["memory_usage_percent"]
				return exists && memUsage > 80.0
			},
			Severity:    AlertSeverityWarning,
			Duration:    30 * time.Second,
			Cooldown:    5 * time.Minute,
			Labels:      map[string]string{"component": "memory", "environment": "production"},
			Annotations: map[string]string{"summary": "High memory usage detected"},
		}

		alertManager.RegisterRule(memoryRule)

		errorRateRule := &AlertRule{
			Name:        "high_error_rate",
			Description: "Error rate exceeds acceptable threshold",
			Condition: func(metrics map[string]float64) bool {
				errorRate, exists := metrics["error_rate_percent"]
				return exists && errorRate > 5.0
			},
			Severity: AlertSeverityCritical,
			Duration: 10 * time.Second,
			Cooldown: 2 * time.Minute,
		}

		alertManager.RegisterRule(errorRateRule)

		// Test alert evaluation
		testMetrics := map[string]float64{
			"memory_usage_percent": 85.0, // Should trigger alert
			"error_rate_percent":   2.0,  // Should not trigger alert
			"cpu_usage_percent":    45.0,
		}

		alerts := alertManager.EvaluateRules(testMetrics)
		if len(alerts) != 1 {
			t.Errorf("Expected 1 alert, got %d", len(alerts))
		}

		if alerts[0].RuleName != "high_memory_usage" {
			t.Error("Should trigger memory usage alert")
		}

		if alerts[0].Severity != AlertSeverityWarning {
			t.Error("Alert should have warning severity")
		}

		// Test notification handlers
		var notificationsSent []Notification
		emailHandler := func(notification Notification) error {
			notificationsSent = append(notificationsSent, notification)
			return nil
		}

		alertManager.AddNotificationHandler("email", emailHandler)

		// Trigger notifications
		for _, alert := range alerts {
			alertManager.SendNotifications(alert)
		}

		if len(notificationsSent) != 1 {
			t.Errorf("Expected 1 notification sent, got %d", len(notificationsSent))
		}

		notification := notificationsSent[0]
		if notification.AlertName != "high_memory_usage" {
			t.Error("Notification should contain alert name")
		}

		if len(notification.Labels) == 0 {
			t.Error("Notification should contain labels")
		}
	})

	t.Run("Telemetry and OpenTelemetry integration", func(t *testing.T) {
		// Test telemetry provider creation
		provider := NewTelemetryProvider(TelemetryConfig{
			ServiceName:    "gonp-test",
			ServiceVersion: "1.0.0",
			Environment:    "test",
			SampleRate:     1.0, // Sample all traces for testing
		})
		if provider == nil {
			t.Error("Should create telemetry provider")
		}

		// Test resource attributes
		resource := provider.Resource()
		if resource.ServiceName() != "gonp-test" {
			t.Error("Should set service name in resource")
		}

		if resource.ServiceVersion() != "1.0.0" {
			t.Error("Should set service version in resource")
		}

		// Test meter provider
		meter := provider.Meter("gonp/array")
		if meter == nil {
			t.Error("Should create meter")
		}

		// Test OpenTelemetry metrics
		operationCounter := meter.NewInt64Counter("gonp_array_operations_total",
			"Total number of array operations")
		operationCounter.Add(context.Background(), 1, map[string]interface{}{
			"operation": "create",
			"type":      "float64",
		})

		memoryGauge := meter.NewFloat64Gauge("gonp_array_memory_bytes",
			"Current array memory usage")
		memoryGauge.Record(context.Background(), 1048576, map[string]interface{}{
			"pool": "default",
		})

		// Test trace provider
		tracer := provider.Tracer("gonp/math")
		ctx := context.Background()

		span := tracer.StartSpan(ctx, "vector_addition")
		span.SetAttribute("vector.length", 1000)
		span.SetAttribute("vector.dtype", "float64")

		// Simulate nested operation
		childCtx := tracer.ContextWithSpan(ctx, span)
		childSpan := tracer.StartSpan(childCtx, "simd_add")
		childSpan.SetAttribute("simd.instruction_set", "AVX2")
		childSpan.Finish()

		span.Finish()

		// Test span export
		spans := provider.GetExportedSpans()
		if len(spans) < 2 {
			t.Errorf("Expected at least 2 exported spans, got %d", len(spans))
		}

		// Test baggage propagation
		baggage := map[string]string{
			"user_id":    "12345",
			"session_id": "abcdef",
		}

		ctxWithBaggage := provider.ContextWithBaggage(ctx, baggage)
		spanWithBaggage := tracer.StartSpan(ctxWithBaggage, "authenticated_operation")

		retrievedBaggage := provider.BaggageFromContext(ctxWithBaggage)
		if retrievedBaggage["user_id"] != "12345" {
			t.Error("Should propagate baggage through context")
		}

		spanWithBaggage.Finish()
	})
}

// TestMonitoringPerformance tests monitoring system performance and overhead
func TestMonitoringPerformance(t *testing.T) {

	t.Run("Metrics collection performance", func(t *testing.T) {
		registry := NewMetricsRegistry()
		counter := registry.NewCounter("perf_test_counter", "Performance test counter")

		// Test high-frequency counter updates
		iterations := 1000000
		start := time.Now()

		for i := 0; i < iterations; i++ {
			counter.Inc()
		}

		duration := time.Since(start)
		opsPerSecond := float64(iterations) / duration.Seconds()

		// Should handle at least 100k ops/second
		if opsPerSecond < 100000 {
			t.Errorf("Counter performance too low: %.0f ops/sec", opsPerSecond)
		}

		t.Logf("Counter performance: %.0f ops/sec", opsPerSecond)
	})

	t.Run("Concurrent metrics collection", func(t *testing.T) {
		registry := NewMetricsRegistry()
		counter := registry.NewCounter("concurrent_counter", "Concurrent test counter")
		gauge := registry.NewGauge("concurrent_gauge", "Concurrent test gauge")

		numWorkers := 100
		operationsPerWorker := 1000
		var wg sync.WaitGroup

		start := time.Now()

		// Launch concurrent workers
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()

				for j := 0; j < operationsPerWorker; j++ {
					counter.Inc()
					gauge.Set(float64(workerID*operationsPerWorker + j))
				}
			}(i)
		}

		wg.Wait()
		duration := time.Since(start)

		expectedCount := float64(numWorkers * operationsPerWorker)
		actualCount := counter.Value()

		if actualCount != expectedCount {
			t.Errorf("Expected counter value %f, got %f", expectedCount, actualCount)
		}

		totalOps := expectedCount * 2 // counter + gauge operations
		opsPerSecond := totalOps / duration.Seconds()

		t.Logf("Concurrent performance: %.0f ops/sec with %d workers", opsPerSecond, numWorkers)
	})

	t.Run("Memory overhead measurement", func(t *testing.T) {
		// Measure baseline memory
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		// Create monitoring infrastructure
		registry := NewMetricsRegistry()
		monitor := NewPerformanceMonitor()
		tracer := NewTracer("overhead-test")

		// Create various metrics
		for i := 0; i < 1000; i++ {
			registry.NewCounter(fmt.Sprintf("counter_%d", i), "Test counter")
			registry.NewGauge(fmt.Sprintf("gauge_%d", i), "Test gauge")
		}

		// Measure memory after setup
		runtime.GC()
		runtime.ReadMemStats(&m2)

		memoryOverhead := int64(m2.Alloc - m1.Alloc)
		t.Logf("Memory overhead for 2000 metrics: %d bytes (%.2f MB)",
			memoryOverhead, float64(memoryOverhead)/1024/1024)

		// Should be reasonable overhead (less than 10MB for 2000 metrics)
		if memoryOverhead > 10*1024*1024 {
			t.Errorf("Memory overhead too high: %d bytes", memoryOverhead)
		}

		// Prevent unused variable warnings
		_ = monitor
		_ = tracer
	})
}

// TestMonitoringIntegration tests integration with existing systems
func TestMonitoringIntegration(t *testing.T) {

	t.Run("Memory pool monitoring integration", func(t *testing.T) {
		// Create monitored memory pool
		poolConfig := MemoryPoolConfig{
			MaxPoolSize:      10 * 1024 * 1024,
			MonitoringMode:   true,
			LeakDetection:    true,
			ProfilingEnabled: true,
		}

		pool, err := NewProductionMemoryPool(poolConfig)
		if err != nil {
			t.Fatalf("Failed to create memory pool: %v", err)
		}
		defer pool.Shutdown()

		// Create monitoring integration
		monitor := NewMemoryPoolMonitor(pool)
		if monitor == nil {
			t.Error("Should create memory pool monitor")
		}

		// Test metrics collection
		metrics := monitor.CollectMetrics()
		if len(metrics) == 0 {
			t.Error("Should collect memory pool metrics")
		}

		expectedMetrics := []string{
			"gonp_memory_pool_active_allocations",
			"gonp_memory_pool_total_allocations",
			"gonp_memory_pool_pool_size_bytes",
			"gonp_memory_pool_efficiency_ratio",
			"gonp_memory_pool_pressure_events_total",
		}

		for _, expected := range expectedMetrics {
			if _, exists := metrics[expected]; !exists {
				t.Errorf("Missing expected metric: %s", expected)
			}
		}

		// Perform some allocations and verify metrics
		var blocks []MemoryBlock
		for i := 0; i < 10; i++ {
			block, err := pool.Allocate(1024)
			if err != nil {
				t.Errorf("Allocation %d failed: %v", i, err)
				continue
			}
			blocks = append(blocks, block)
		}

		updatedMetrics := monitor.CollectMetrics()
		activeAllocs := updatedMetrics["gonp_memory_pool_active_allocations"]
		if activeAllocs != 10 {
			t.Errorf("Expected 10 active allocations, got %f", activeAllocs)
		}

		// Clean up
		for _, block := range blocks {
			pool.Deallocate(block)
		}
	})

	t.Run("Error tracking integration", func(t *testing.T) {
		// Create error monitor
		errorMonitor := NewErrorMonitor()
		if errorMonitor == nil {
			t.Error("Should create error monitor")
		}

		// Create structured errors and track them
		category := NewErrorCategory("TEST", "Test errors")
		errorType := NewErrorType(category, "TEST_ERROR", "Test error type")

		for i := 0; i < 5; i++ {
			err := NewStructuredError(errorType, SeverityError, fmt.Sprintf("Test error %d", i))
			err.WithContext("iteration", i)
			errorMonitor.RecordError(err)
		}

		// Test error metrics
		metrics := errorMonitor.GetMetrics()
		if metrics.TotalErrors != 5 {
			t.Errorf("Expected 5 total errors, got %d", metrics.TotalErrors)
		}

		if len(metrics.ErrorsByCategory) == 0 {
			t.Error("Should have errors by category")
		}

		if count, exists := metrics.ErrorsByCategory["TEST"]; !exists || count != 5 {
			t.Errorf("Expected 5 TEST category errors, got %d", count)
		}

		// Test error rate calculation
		errorRate := errorMonitor.GetErrorRate(time.Minute)
		if errorRate != 5.0 {
			t.Errorf("Expected error rate 5.0/min, got %f", errorRate)
		}
	})
}
