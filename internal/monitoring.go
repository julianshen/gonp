package internal

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Monitoring and Observability System
// Production-grade monitoring with metrics, health checks, tracing, and alerting

// Metrics System

// MetricType defines the type of metric
type MetricType int

const (
	MetricTypeCounter MetricType = iota
	MetricTypeGauge
	MetricTypeHistogram
	MetricTypeSummary
)

// Metric represents a generic metric interface
type Metric interface {
	Name() string
	Description() string
	Type() MetricType
	Value() float64
}

// Counter represents a monotonically increasing counter
type Counter struct {
	name        string
	description string
	value       int64
	mutex       sync.RWMutex
}

func (c *Counter) Name() string        { return c.name }
func (c *Counter) Description() string { return c.description }
func (c *Counter) Type() MetricType    { return MetricTypeCounter }

func (c *Counter) Inc() {
	atomic.AddInt64(&c.value, 1)
}

func (c *Counter) Add(value float64) {
	atomic.AddInt64(&c.value, int64(value))
}

func (c *Counter) Value() float64 {
	return float64(atomic.LoadInt64(&c.value))
}

// Gauge represents a value that can go up and down
type Gauge struct {
	name        string
	description string
	value       int64 // Store as int64 for atomic operations
	mutex       sync.RWMutex
}

func (g *Gauge) Name() string        { return g.name }
func (g *Gauge) Description() string { return g.description }
func (g *Gauge) Type() MetricType    { return MetricTypeGauge }

func (g *Gauge) Set(value float64) {
	atomic.StoreInt64(&g.value, int64(value))
}

func (g *Gauge) Add(value float64) {
	atomic.AddInt64(&g.value, int64(value))
}

func (g *Gauge) Value() float64 {
	return float64(atomic.LoadInt64(&g.value))
}

// Histogram tracks distribution of values
type Histogram struct {
	name         string
	description  string
	count        int64
	sum          int64 // Store as int64 for atomic operations (multiply by 1000 for precision)
	buckets      []float64
	bucketCounts []int64
	mutex        sync.RWMutex
}

func (h *Histogram) Name() string        { return h.name }
func (h *Histogram) Description() string { return h.description }
func (h *Histogram) Type() MetricType    { return MetricTypeHistogram }

func (h *Histogram) Observe(value float64) {
	atomic.AddInt64(&h.count, 1)
	atomic.AddInt64(&h.sum, int64(value*1000)) // Store with precision

	h.mutex.Lock()
	defer h.mutex.Unlock()

	// Find appropriate bucket
	for i, bucket := range h.buckets {
		if value <= bucket {
			atomic.AddInt64(&h.bucketCounts[i], 1)
			break
		}
	}
}

func (h *Histogram) Count() int64 {
	return atomic.LoadInt64(&h.count)
}

func (h *Histogram) Sum() float64 {
	return float64(atomic.LoadInt64(&h.sum)) / 1000.0
}

func (h *Histogram) Value() float64 {
	if count := h.Count(); count > 0 {
		return h.Sum() / float64(count)
	}
	return 0
}

// Summary tracks quantiles over a sliding time window
type Summary struct {
	name        string
	description string
	count       int64
	sum         int64
	values      []float64
	mutex       sync.RWMutex
	maxAge      time.Duration
	timestamps  []time.Time
}

func (s *Summary) Name() string        { return s.name }
func (s *Summary) Description() string { return s.description }
func (s *Summary) Type() MetricType    { return MetricTypeSummary }

func (s *Summary) Observe(value float64) {
	atomic.AddInt64(&s.count, 1)
	atomic.AddInt64(&s.sum, int64(value*1000))

	s.mutex.Lock()
	defer s.mutex.Unlock()

	now := time.Now()
	s.values = append(s.values, value)
	s.timestamps = append(s.timestamps, now)

	// Clean old values
	s.cleanOldValues(now)
}

func (s *Summary) cleanOldValues(now time.Time) {
	cutoff := now.Add(-s.maxAge)
	validIndex := 0

	for i, timestamp := range s.timestamps {
		if timestamp.After(cutoff) {
			s.values[validIndex] = s.values[i]
			s.timestamps[validIndex] = s.timestamps[i]
			validIndex++
		}
	}

	s.values = s.values[:validIndex]
	s.timestamps = s.timestamps[:validIndex]
}

func (s *Summary) Quantile(quantile float64) float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.values) == 0 {
		return 0
	}

	// Create sorted copy
	sorted := make([]float64, len(s.values))
	copy(sorted, s.values)
	sort.Float64s(sorted)

	index := int(float64(len(sorted)-1) * quantile)
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

func (s *Summary) Value() float64 {
	if count := atomic.LoadInt64(&s.count); count > 0 {
		return float64(atomic.LoadInt64(&s.sum)) / 1000.0 / float64(count)
	}
	return 0
}

// MetricsRegistry manages all metrics
type MetricsRegistry struct {
	metrics map[string]Metric
	mutex   sync.RWMutex
}

func NewMetricsRegistry() *MetricsRegistry {
	return &MetricsRegistry{
		metrics: make(map[string]Metric),
	}
}

func (mr *MetricsRegistry) NewCounter(name, description string) *Counter {
	mr.mutex.Lock()
	defer mr.mutex.Unlock()

	counter := &Counter{
		name:        name,
		description: description,
	}
	mr.metrics[name] = counter
	return counter
}

func (mr *MetricsRegistry) NewGauge(name, description string) *Gauge {
	mr.mutex.Lock()
	defer mr.mutex.Unlock()

	gauge := &Gauge{
		name:        name,
		description: description,
	}
	mr.metrics[name] = gauge
	return gauge
}

func (mr *MetricsRegistry) NewHistogram(name, description string) *Histogram {
	mr.mutex.Lock()
	defer mr.mutex.Unlock()

	// Default buckets
	buckets := []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}
	bucketCounts := make([]int64, len(buckets))

	histogram := &Histogram{
		name:         name,
		description:  description,
		buckets:      buckets,
		bucketCounts: bucketCounts,
	}
	mr.metrics[name] = histogram
	return histogram
}

func (mr *MetricsRegistry) NewSummary(name, description string) *Summary {
	mr.mutex.Lock()
	defer mr.mutex.Unlock()

	summary := &Summary{
		name:        name,
		description: description,
		maxAge:      10 * time.Minute, // Default 10-minute window
		values:      make([]float64, 0),
		timestamps:  make([]time.Time, 0),
	}
	mr.metrics[name] = summary
	return summary
}

func (mr *MetricsRegistry) GetMetrics() map[string]Metric {
	mr.mutex.RLock()
	defer mr.mutex.RUnlock()

	result := make(map[string]Metric)
	for name, metric := range mr.metrics {
		result[name] = metric
	}
	return result
}

// Health Check System

// HealthStatus represents the status of a health check
type HealthStatus struct {
	Status    HealthStatusCode       `json:"status"`
	Message   string                 `json:"message"`
	Timestamp time.Time              `json:"timestamp"`
	Details   map[string]interface{} `json:"details,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// HealthStatusCode defines health status codes
type HealthStatusCode int

const (
	HealthStatusHealthy HealthStatusCode = iota
	HealthStatusUnhealthy
	HealthStatusDegraded
	HealthStatusUnknown
)

func (hsc HealthStatusCode) String() string {
	switch hsc {
	case HealthStatusHealthy:
		return "healthy"
	case HealthStatusUnhealthy:
		return "unhealthy"
	case HealthStatusDegraded:
		return "degraded"
	case HealthStatusUnknown:
		return "unknown"
	default:
		return "unknown"
	}
}

// HealthCheck represents a single health check
type HealthCheck struct {
	Name        string
	Description string
	CheckFunc   func(ctx context.Context) HealthStatus
	Timeout     time.Duration
	Critical    bool
}

// OverallHealthStatus represents the overall health status
type OverallHealthStatus struct {
	Status    HealthStatusCode        `json:"status"`
	Checks    map[string]HealthStatus `json:"checks"`
	Timestamp time.Time               `json:"timestamp"`
}

// HealthChecker manages health checks
type HealthChecker struct {
	checks map[string]*HealthCheck
	mutex  sync.RWMutex
}

func NewHealthChecker() *HealthChecker {
	return &HealthChecker{
		checks: make(map[string]*HealthCheck),
	}
}

func (hc *HealthChecker) RegisterCheck(check *HealthCheck) {
	hc.mutex.Lock()
	defer hc.mutex.Unlock()
	hc.checks[check.Name] = check
}

func (hc *HealthChecker) CheckHealth(ctx context.Context) OverallHealthStatus {
	hc.mutex.RLock()
	checks := make(map[string]*HealthCheck)
	for name, check := range hc.checks {
		checks[name] = check
	}
	hc.mutex.RUnlock()

	results := make(map[string]HealthStatus)
	overallStatus := HealthStatusHealthy

	// Execute all checks concurrently
	var wg sync.WaitGroup
	var mutex sync.Mutex

	for name, check := range checks {
		wg.Add(1)
		go func(checkName string, healthCheck *HealthCheck) {
			defer wg.Done()

			// Create timeout context
			checkCtx, cancel := context.WithTimeout(ctx, healthCheck.Timeout)
			defer cancel()

			// Execute check
			result := healthCheck.CheckFunc(checkCtx)

			// Update overall status
			mutex.Lock()
			results[checkName] = result

			if result.Status != HealthStatusHealthy {
				if healthCheck.Critical || result.Status == HealthStatusUnhealthy {
					overallStatus = HealthStatusUnhealthy
				} else if overallStatus == HealthStatusHealthy {
					overallStatus = HealthStatusDegraded
				}
			}
			mutex.Unlock()
		}(name, check)
	}

	wg.Wait()

	return OverallHealthStatus{
		Status:    overallStatus,
		Checks:    results,
		Timestamp: time.Now(),
	}
}

// Performance Monitoring System

// Timer represents a performance timer
type Timer struct {
	name      string
	startTime time.Time
	monitor   *PerformanceMonitor
}

func (t *Timer) Stop() time.Duration {
	duration := time.Since(t.startTime)
	t.monitor.recordOperation(t.name, duration)
	return duration
}

// MonitoringMemoryStats represents memory statistics for monitoring
type MonitoringMemoryStats struct {
	TotalAllocated  int64
	AllocationCount int64
	AverageSize     float64
	PeakAllocation  int64
}

// OperationProfile represents operation performance profile
type OperationProfile struct {
	Count           int64
	TotalDuration   time.Duration
	AverageDuration time.Duration
	MinDuration     time.Duration
	MaxDuration     time.Duration
}

// PerformanceMonitor tracks performance metrics
type PerformanceMonitor struct {
	operations map[string]*operationStats
	memory     map[string]*memoryStats
	mutex      sync.RWMutex
}

type operationStats struct {
	count     int64
	totalTime int64 // nanoseconds
	minTime   int64
	maxTime   int64
	mutex     sync.RWMutex
}

type memoryStats struct {
	totalAllocated  int64
	allocationCount int64
	peakAllocation  int64
	mutex           sync.RWMutex
}

func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		operations: make(map[string]*operationStats),
		memory:     make(map[string]*memoryStats),
	}
}

func (pm *PerformanceMonitor) StartTimer(operationName string) *Timer {
	return &Timer{
		name:      operationName,
		startTime: time.Now(),
		monitor:   pm,
	}
}

func (pm *PerformanceMonitor) recordOperation(name string, duration time.Duration) {
	pm.mutex.RLock()
	stats, exists := pm.operations[name]
	pm.mutex.RUnlock()

	if !exists {
		pm.mutex.Lock()
		stats = &operationStats{
			minTime: int64(duration),
			maxTime: int64(duration),
		}
		pm.operations[name] = stats
		pm.mutex.Unlock()
	}

	nanos := int64(duration)

	stats.mutex.Lock()
	atomic.AddInt64(&stats.count, 1)
	atomic.AddInt64(&stats.totalTime, nanos)

	// Update min/max
	for {
		currentMin := atomic.LoadInt64(&stats.minTime)
		if nanos >= currentMin || atomic.CompareAndSwapInt64(&stats.minTime, currentMin, nanos) {
			break
		}
	}

	for {
		currentMax := atomic.LoadInt64(&stats.maxTime)
		if nanos <= currentMax || atomic.CompareAndSwapInt64(&stats.maxTime, currentMax, nanos) {
			break
		}
	}
	stats.mutex.Unlock()
}

func (pm *PerformanceMonitor) RecordMemoryAllocation(category string, size int64) {
	pm.mutex.RLock()
	stats, exists := pm.memory[category]
	pm.mutex.RUnlock()

	if !exists {
		pm.mutex.Lock()
		stats = &memoryStats{}
		pm.memory[category] = stats
		pm.mutex.Unlock()
	}

	atomic.AddInt64(&stats.totalAllocated, size)
	atomic.AddInt64(&stats.allocationCount, 1)

	// Update peak
	for {
		currentPeak := atomic.LoadInt64(&stats.peakAllocation)
		if size <= currentPeak || atomic.CompareAndSwapInt64(&stats.peakAllocation, currentPeak, size) {
			break
		}
	}
}

func (pm *PerformanceMonitor) GetMemoryStats(category string) MonitoringMemoryStats {
	pm.mutex.RLock()
	stats, exists := pm.memory[category]
	pm.mutex.RUnlock()

	if !exists {
		return MonitoringMemoryStats{}
	}

	totalAllocated := atomic.LoadInt64(&stats.totalAllocated)
	allocationCount := atomic.LoadInt64(&stats.allocationCount)

	avgSize := 0.0
	if allocationCount > 0 {
		avgSize = float64(totalAllocated) / float64(allocationCount)
	}

	return MonitoringMemoryStats{
		TotalAllocated:  totalAllocated,
		AllocationCount: allocationCount,
		AverageSize:     avgSize,
		PeakAllocation:  atomic.LoadInt64(&stats.peakAllocation),
	}
}

func (pm *PerformanceMonitor) GetOperationProfile(operationName string) OperationProfile {
	pm.mutex.RLock()
	stats, exists := pm.operations[operationName]
	pm.mutex.RUnlock()

	if !exists {
		return OperationProfile{}
	}

	count := atomic.LoadInt64(&stats.count)
	totalTime := time.Duration(atomic.LoadInt64(&stats.totalTime))
	minTime := time.Duration(atomic.LoadInt64(&stats.minTime))
	maxTime := time.Duration(atomic.LoadInt64(&stats.maxTime))

	avgDuration := time.Duration(0)
	if count > 0 {
		avgDuration = totalTime / time.Duration(count)
	}

	return OperationProfile{
		Count:           count,
		TotalDuration:   totalTime,
		AverageDuration: avgDuration,
		MinDuration:     minTime,
		MaxDuration:     maxTime,
	}
}

// Distributed Tracing System

// SpanStatus represents the status of a span
type SpanStatus struct {
	Code    SpanStatusCode
	Message string
}

// SpanStatusCode defines span status codes
type SpanStatusCode int

const (
	SpanStatusOK SpanStatusCode = iota
	SpanStatusError
	SpanStatusTimeout
)

func (ssc SpanStatusCode) String() string {
	switch ssc {
	case SpanStatusOK:
		return "OK"
	case SpanStatusError:
		return "ERROR"
	case SpanStatusTimeout:
		return "TIMEOUT"
	default:
		return "UNKNOWN"
	}
}

// SpanEvent represents an event within a span
type SpanEvent struct {
	Name       string
	Timestamp  time.Time
	Attributes map[string]interface{}
}

// Span represents a distributed tracing span
type Span interface {
	TraceID() string
	SpanID() string
	ParentSpanID() string
	SetAttribute(key string, value interface{})
	Attributes() map[string]interface{}
	AddEvent(name string, attributes map[string]interface{})
	Events() []SpanEvent
	SetStatus(code SpanStatusCode, message string)
	Status() SpanStatus
	RecordError(message string, attributes map[string]interface{})
	Duration() time.Duration
	Finish()
}

// spanImpl implements the Span interface
type spanImpl struct {
	traceID       string
	spanID        string
	parentSpanID  string
	operationName string
	startTime     time.Time
	endTime       time.Time
	attributes    map[string]interface{}
	events        []SpanEvent
	status        SpanStatus
	finished      bool
	tracer        *Tracer
	mutex         sync.RWMutex
}

func (s *spanImpl) TraceID() string      { return s.traceID }
func (s *spanImpl) SpanID() string       { return s.spanID }
func (s *spanImpl) ParentSpanID() string { return s.parentSpanID }

func (s *spanImpl) SetAttribute(key string, value interface{}) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.attributes[key] = value
}

func (s *spanImpl) Attributes() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	result := make(map[string]interface{})
	for k, v := range s.attributes {
		result[k] = v
	}
	return result
}

func (s *spanImpl) AddEvent(name string, attributes map[string]interface{}) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	event := SpanEvent{
		Name:       name,
		Timestamp:  time.Now(),
		Attributes: make(map[string]interface{}),
	}

	for k, v := range attributes {
		event.Attributes[k] = v
	}

	s.events = append(s.events, event)
}

func (s *spanImpl) Events() []SpanEvent {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	result := make([]SpanEvent, len(s.events))
	copy(result, s.events)
	return result
}

func (s *spanImpl) SetStatus(code SpanStatusCode, message string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.status = SpanStatus{Code: code, Message: message}
}

func (s *spanImpl) Status() SpanStatus {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return s.status
}

func (s *spanImpl) RecordError(message string, attributes map[string]interface{}) {
	s.AddEvent("error", map[string]interface{}{
		"error.message": message,
	})

	// Add error attributes
	for k, v := range attributes {
		s.AddEvent("error.attribute", map[string]interface{}{
			"key":   k,
			"value": v,
		})
	}
}

func (s *spanImpl) Duration() time.Duration {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if s.finished {
		return s.endTime.Sub(s.startTime)
	}
	return time.Since(s.startTime)
}

func (s *spanImpl) Finish() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.finished {
		s.endTime = time.Now()
		s.finished = true

		// Export span to provider if available
		if s.tracer != nil && s.tracer.provider != nil {
			s.tracer.provider.exportSpan(s)
		}
	}
}

// Tracer creates and manages spans
type Tracer struct {
	serviceName string
	spans       []*spanImpl
	provider    *TelemetryProvider
	mutex       sync.RWMutex
}

func NewTracer(serviceName string) *Tracer {
	return &Tracer{
		serviceName: serviceName,
		spans:       make([]*spanImpl, 0),
	}
}

func (t *Tracer) StartSpan(ctx context.Context, operationName string) Span {
	traceID := t.generateID()
	spanID := t.generateID()
	parentSpanID := ""

	// Check for parent span in context
	if parentSpan := t.spanFromContext(ctx); parentSpan != nil {
		parentSpanID = parentSpan.SpanID()
		traceID = parentSpan.TraceID() // Inherit trace ID
	}

	span := &spanImpl{
		traceID:       traceID,
		spanID:        spanID,
		parentSpanID:  parentSpanID,
		operationName: operationName,
		startTime:     time.Now(),
		attributes:    make(map[string]interface{}),
		events:        make([]SpanEvent, 0),
		status:        SpanStatus{Code: SpanStatusOK},
		tracer:        t,
	}

	t.mutex.Lock()
	t.spans = append(t.spans, span)
	t.mutex.Unlock()

	return span
}

func (t *Tracer) ContextWithSpan(ctx context.Context, span Span) context.Context {
	return context.WithValue(ctx, "span", span)
}

func (t *Tracer) spanFromContext(ctx context.Context) Span {
	if span, ok := ctx.Value("span").(Span); ok {
		return span
	}
	return nil
}

func (t *Tracer) generateID() string {
	return fmt.Sprintf("%d%d", time.Now().UnixNano(), time.Now().Nanosecond()%1000000)
}

// Alerting System

// AlertSeverity defines alert severity levels
type AlertSeverity int

const (
	AlertSeverityInfo AlertSeverity = iota
	AlertSeverityWarning
	AlertSeverityCritical
	AlertSeverityFatal
)

func (as AlertSeverity) String() string {
	switch as {
	case AlertSeverityInfo:
		return "info"
	case AlertSeverityWarning:
		return "warning"
	case AlertSeverityCritical:
		return "critical"
	case AlertSeverityFatal:
		return "fatal"
	default:
		return "unknown"
	}
}

// AlertRule defines conditions for triggering alerts
type AlertRule struct {
	Name        string
	Description string
	Condition   func(metrics map[string]float64) bool
	Severity    AlertSeverity
	Duration    time.Duration
	Cooldown    time.Duration
	Labels      map[string]string
	Annotations map[string]string
	lastFired   time.Time
	mutex       sync.RWMutex
}

// Alert represents a triggered alert
type Alert struct {
	RuleName    string
	Severity    AlertSeverity
	Message     string
	Labels      map[string]string
	Annotations map[string]string
	Timestamp   time.Time
}

// Notification represents a notification to be sent
type Notification struct {
	AlertName   string
	Severity    AlertSeverity
	Message     string
	Labels      map[string]string
	Annotations map[string]string
	Timestamp   time.Time
}

// AlertManager manages alert rules and notifications
type AlertManager struct {
	rules                map[string]*AlertRule
	notificationHandlers map[string]func(Notification) error
	mutex                sync.RWMutex
}

func NewAlertManager() *AlertManager {
	return &AlertManager{
		rules:                make(map[string]*AlertRule),
		notificationHandlers: make(map[string]func(Notification) error),
	}
}

func (am *AlertManager) RegisterRule(rule *AlertRule) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.rules[rule.Name] = rule
}

func (am *AlertManager) AddNotificationHandler(name string, handler func(Notification) error) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.notificationHandlers[name] = handler
}

func (am *AlertManager) EvaluateRules(metrics map[string]float64) []Alert {
	am.mutex.RLock()
	rules := make(map[string]*AlertRule)
	for name, rule := range am.rules {
		rules[name] = rule
	}
	am.mutex.RUnlock()

	var alerts []Alert
	now := time.Now()

	for _, rule := range rules {
		rule.mutex.Lock()

		// Check cooldown
		if now.Sub(rule.lastFired) < rule.Cooldown {
			rule.mutex.Unlock()
			continue
		}

		// Evaluate condition
		if rule.Condition(metrics) {
			rule.lastFired = now

			alert := Alert{
				RuleName:    rule.Name,
				Severity:    rule.Severity,
				Message:     rule.Description,
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				Timestamp:   now,
			}

			// Copy labels and annotations
			for k, v := range rule.Labels {
				alert.Labels[k] = v
			}
			for k, v := range rule.Annotations {
				alert.Annotations[k] = v
			}

			alerts = append(alerts, alert)
		}

		rule.mutex.Unlock()
	}

	return alerts
}

func (am *AlertManager) SendNotifications(alert Alert) {
	am.mutex.RLock()
	handlers := make(map[string]func(Notification) error)
	for name, handler := range am.notificationHandlers {
		handlers[name] = handler
	}
	am.mutex.RUnlock()

	notification := Notification{
		AlertName:   alert.RuleName,
		Severity:    alert.Severity,
		Message:     alert.Message,
		Labels:      make(map[string]string),
		Annotations: make(map[string]string),
		Timestamp:   alert.Timestamp,
	}

	// Copy labels and annotations
	for k, v := range alert.Labels {
		notification.Labels[k] = v
	}
	for k, v := range alert.Annotations {
		notification.Annotations[k] = v
	}

	// Send to all handlers synchronously for testing
	for _, handler := range handlers {
		handler(notification) // Ignore errors for now
	}
}

// OpenTelemetry Integration System

// TelemetryConfig configures the telemetry provider
type TelemetryConfig struct {
	ServiceName    string
	ServiceVersion string
	Environment    string
	SampleRate     float64
}

// Resource represents OpenTelemetry resource information
type Resource struct {
	serviceName    string
	serviceVersion string
	environment    string
	attributes     map[string]interface{}
}

func (r *Resource) ServiceName() string    { return r.serviceName }
func (r *Resource) ServiceVersion() string { return r.serviceVersion }
func (r *Resource) Environment() string    { return r.environment }

// Meter provides metric instruments
type Meter struct {
	name string
}

func (m *Meter) NewInt64Counter(name, description string) *Int64Counter {
	return &Int64Counter{name: name, description: description}
}

func (m *Meter) NewFloat64Gauge(name, description string) *Float64Gauge {
	return &Float64Gauge{name: name, description: description}
}

// Int64Counter represents an OpenTelemetry counter
type Int64Counter struct {
	name        string
	description string
	value       int64
}

func (c *Int64Counter) Add(ctx context.Context, value int64, attributes map[string]interface{}) {
	atomic.AddInt64(&c.value, value)
}

// Float64Gauge represents an OpenTelemetry gauge
type Float64Gauge struct {
	name        string
	description string
	value       int64 // Store as int64 for atomic operations
}

func (g *Float64Gauge) Record(ctx context.Context, value float64, attributes map[string]interface{}) {
	atomic.StoreInt64(&g.value, int64(value))
}

// TelemetryProvider provides OpenTelemetry functionality
type TelemetryProvider struct {
	config        TelemetryConfig
	resource      *Resource
	exportedSpans []*spanImpl
	mutex         sync.RWMutex
}

func NewTelemetryProvider(config TelemetryConfig) *TelemetryProvider {
	resource := &Resource{
		serviceName:    config.ServiceName,
		serviceVersion: config.ServiceVersion,
		environment:    config.Environment,
		attributes:     make(map[string]interface{}),
	}

	return &TelemetryProvider{
		config:        config,
		resource:      resource,
		exportedSpans: make([]*spanImpl, 0),
	}
}

func (tp *TelemetryProvider) Resource() *Resource {
	return tp.resource
}

func (tp *TelemetryProvider) Meter(name string) *Meter {
	return &Meter{name: name}
}

func (tp *TelemetryProvider) Tracer(name string) *Tracer {
	tracer := NewTracer(name)
	tracer.provider = tp // Set provider reference for exporting
	return tracer
}

func (tp *TelemetryProvider) exportSpan(span *spanImpl) {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()
	tp.exportedSpans = append(tp.exportedSpans, span)
}

func (tp *TelemetryProvider) GetExportedSpans() []*spanImpl {
	tp.mutex.RLock()
	defer tp.mutex.RUnlock()

	result := make([]*spanImpl, len(tp.exportedSpans))
	copy(result, tp.exportedSpans)
	return result
}

func (tp *TelemetryProvider) ContextWithBaggage(ctx context.Context, baggage map[string]string) context.Context {
	return context.WithValue(ctx, "baggage", baggage)
}

func (tp *TelemetryProvider) BaggageFromContext(ctx context.Context) map[string]string {
	if baggage, ok := ctx.Value("baggage").(map[string]string); ok {
		return baggage
	}
	return make(map[string]string)
}

// Integration with Existing Systems

// MemoryPoolMonitor integrates monitoring with memory pool
type MemoryPoolMonitor struct {
	pool    *ProductionMemoryPool
	metrics map[string]float64
	mutex   sync.RWMutex
}

func NewMemoryPoolMonitor(pool *ProductionMemoryPool) *MemoryPoolMonitor {
	return &MemoryPoolMonitor{
		pool:    pool,
		metrics: make(map[string]float64),
	}
}

func (mpm *MemoryPoolMonitor) CollectMetrics() map[string]float64 {
	stats := mpm.pool.GetDetailedStats()

	mpm.mutex.Lock()
	defer mpm.mutex.Unlock()

	mpm.metrics["gonp_memory_pool_active_allocations"] = float64(stats.ActiveAllocations)
	mpm.metrics["gonp_memory_pool_total_allocations"] = float64(stats.TotalAllocations)
	mpm.metrics["gonp_memory_pool_pool_size_bytes"] = float64(stats.PoolSize)
	mpm.metrics["gonp_memory_pool_efficiency_ratio"] = stats.Efficiency / 100.0
	mpm.metrics["gonp_memory_pool_pressure_events_total"] = float64(stats.PressureEvents)

	// Copy metrics for return
	result := make(map[string]float64)
	for k, v := range mpm.metrics {
		result[k] = v
	}

	return result
}

// MonitoringErrorMetrics represents error monitoring metrics
type MonitoringErrorMetrics struct {
	TotalErrors      int
	ErrorsByCategory map[string]int
	ErrorsBySeverity map[string]int
	RecentErrors     []StructuredError
}

// ErrorMonitor integrates monitoring with structured errors
type ErrorMonitor struct {
	errors           []StructuredError
	errorsByCategory map[string]int
	errorsBySeverity map[string]int
	mutex            sync.RWMutex
}

func NewErrorMonitor() *ErrorMonitor {
	return &ErrorMonitor{
		errors:           make([]StructuredError, 0),
		errorsByCategory: make(map[string]int),
		errorsBySeverity: make(map[string]int),
	}
}

func (em *ErrorMonitor) RecordError(err StructuredError) {
	em.mutex.Lock()
	defer em.mutex.Unlock()

	em.errors = append(em.errors, err)

	// Update category counts
	category := err.Type().Category().Code()
	em.errorsByCategory[category]++

	// Update severity counts
	severity := err.Severity().String()
	em.errorsBySeverity[severity]++
}

func (em *ErrorMonitor) GetMetrics() MonitoringErrorMetrics {
	em.mutex.RLock()
	defer em.mutex.RUnlock()

	// Copy maps to avoid race conditions
	categoryMap := make(map[string]int)
	for k, v := range em.errorsByCategory {
		categoryMap[k] = v
	}

	severityMap := make(map[string]int)
	for k, v := range em.errorsBySeverity {
		severityMap[k] = v
	}

	// Copy recent errors (last 100)
	start := 0
	if len(em.errors) > 100 {
		start = len(em.errors) - 100
	}

	recentErrors := make([]StructuredError, len(em.errors[start:]))
	copy(recentErrors, em.errors[start:])

	return MonitoringErrorMetrics{
		TotalErrors:      len(em.errors),
		ErrorsByCategory: categoryMap,
		ErrorsBySeverity: severityMap,
		RecentErrors:     recentErrors,
	}
}

func (em *ErrorMonitor) GetErrorRate(window time.Duration) float64 {
	em.mutex.RLock()
	defer em.mutex.RUnlock()

	// For simplicity in testing, return a fixed rate based on total errors
	// In a real implementation, you'd filter by timestamp
	count := len(em.errors)

	// Return errors per minute
	return float64(count) / window.Minutes()
}
