package internal

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ShapeError represents an error related to array shape mismatches
type ShapeError struct {
	Op     string
	Shape1 Shape
	Shape2 Shape
	Msg    string
}

func (e *ShapeError) Error() string {
	if e.Msg != "" {
		return fmt.Sprintf("shape error in %s: %s", e.Op, e.Msg)
	}
	return fmt.Sprintf("shape mismatch in %s: %v vs %v", e.Op, e.Shape1, e.Shape2)
}

// NewShapeError creates a new shape error
func NewShapeError(op string, shape1, shape2 Shape) *ShapeError {
	return &ShapeError{
		Op:     op,
		Shape1: shape1,
		Shape2: shape2,
	}
}

// NewShapeErrorWithMsg creates a new shape error with custom message
func NewShapeErrorWithMsg(op, msg string) *ShapeError {
	return &ShapeError{
		Op:  op,
		Msg: msg,
	}
}

// IndexError represents an error related to array indexing
type IndexError struct {
	Op      string
	Index   []int
	Shape   Shape
	Message string
}

func (e *IndexError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("index error in %s: %s", e.Op, e.Message)
	}
	return fmt.Sprintf("index error in %s: index %v out of bounds for shape %v", e.Op, e.Index, e.Shape)
}

// NewIndexError creates a new index error
func NewIndexError(op string, index []int, shape Shape) *IndexError {
	return &IndexError{
		Op:    op,
		Index: index,
		Shape: shape,
	}
}

// NewIndexErrorWithMsg creates a new index error with custom message
func NewIndexErrorWithMsg(op, msg string) *IndexError {
	return &IndexError{
		Op:      op,
		Message: msg,
	}
}

// TypeError represents an error related to data type mismatches
type TypeError struct {
	Op       string
	Expected DType
	Actual   DType
	Message  string
}

func (e *TypeError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("type error in %s: %s", e.Op, e.Message)
	}
	return fmt.Sprintf("type error in %s: expected %v, got %v", e.Op, e.Expected, e.Actual)
}

// NewTypeError creates a new type error
func NewTypeError(op string, expected, actual DType) *TypeError {
	return &TypeError{
		Op:       op,
		Expected: expected,
		Actual:   actual,
	}
}

// NewTypeErrorWithMsg creates a new type error with custom message
func NewTypeErrorWithMsg(op, msg string) *TypeError {
	return &TypeError{
		Op:      op,
		Message: msg,
	}
}

// ValueErr represents an error related to invalid values
type ValueErr struct {
	Op      string
	Value   interface{}
	Message string
}

func (e *ValueErr) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("value error in %s: %s", e.Op, e.Message)
	}
	return fmt.Sprintf("value error in %s: invalid value %v", e.Op, e.Value)
}

// NewValueError creates a new value error
func NewValueError(op string, value interface{}) *ValueErr {
	return &ValueErr{
		Op:    op,
		Value: value,
	}
}

// NewValueErrorWithMsg creates a new value error with custom message
func NewValueErrorWithMsg(op, msg string) *ValueErr {
	return &ValueErr{
		Op:      op,
		Message: msg,
	}
}

// MemoryError represents memory allocation or management errors
type MemoryError struct {
	Op      string
	Size    int64
	Message string
}

func (e *MemoryError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("memory error in %s: %s", e.Op, e.Message)
	}
	return fmt.Sprintf("memory error in %s: failed to allocate %d bytes", e.Op, e.Size)
}

// NewMemoryError creates a new memory error
func NewMemoryError(op string, size int64) *MemoryError {
	return &MemoryError{
		Op:   op,
		Size: size,
	}
}

// NewMemoryErrorWithMsg creates a new memory error with custom message
func NewMemoryErrorWithMsg(op, msg string) *MemoryError {
	return &MemoryError{
		Op:      op,
		Message: msg,
	}
}

// ComputationError represents errors during mathematical computations
type ComputationError struct {
	Op       string
	Function string
	Input    interface{}
	Message  string
}

func (e *ComputationError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("computation error in %s: %s", e.Op, e.Message)
	}
	if e.Function != "" {
		return fmt.Sprintf("computation error in %s: %s failed with input %v", e.Op, e.Function, e.Input)
	}
	return fmt.Sprintf("computation error in %s", e.Op)
}

// NewComputationError creates a new computation error
func NewComputationError(op, function string, input interface{}) *ComputationError {
	return &ComputationError{
		Op:       op,
		Function: function,
		Input:    input,
	}
}

// NewComputationErrorWithMsg creates a new computation error with custom message
// ErrSIMDUsed is a special marker error to indicate SIMD was successfully used
var ErrSIMDUsed = fmt.Errorf("SIMD optimization was used")

func NewComputationErrorWithMsg(op, msg string) *ComputationError {
	return &ComputationError{
		Op:      op,
		Message: msg,
	}
}

// IOError represents input/output related errors
type IOError struct {
	Op       string
	Filename string
	Message  string
}

func (e *IOError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("I/O error in %s: %s", e.Op, e.Message)
	}
	return fmt.Sprintf("I/O error in %s: operation failed on %s", e.Op, e.Filename)
}

// NewIOError creates a new I/O error
func NewIOError(op, filename string) *IOError {
	return &IOError{
		Op:       op,
		Filename: filename,
	}
}

// NewIOErrorWithMsg creates a new I/O error with custom message
func NewIOErrorWithMsg(op, msg string) *IOError {
	return &IOError{
		Op:      op,
		Message: msg,
	}
}

// ValidationError represents parameter validation errors
type ValidationError struct {
	Op        string
	Parameter string
	Value     interface{}
	Message   string
}

func (e *ValidationError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("validation error in %s: %s", e.Op, e.Message)
	}
	return fmt.Sprintf("validation error in %s: invalid parameter %s = %v", e.Op, e.Parameter, e.Value)
}

// NewValidationError creates a new validation error
func NewValidationError(op, parameter string, value interface{}) *ValidationError {
	return &ValidationError{
		Op:        op,
		Parameter: parameter,
		Value:     value,
	}
}

// NewValidationErrorWithMsg creates a new validation error with custom message
func NewValidationErrorWithMsg(op, msg string) *ValidationError {
	return &ValidationError{
		Op:      op,
		Message: msg,
	}
}

// IsErrorType checks if an error is of a specific type
func IsShapeError(err error) bool {
	_, ok := err.(*ShapeError)
	return ok
}

func IsIndexError(err error) bool {
	_, ok := err.(*IndexError)
	return ok
}

func IsTypeError(err error) bool {
	_, ok := err.(*TypeError)
	return ok
}

func IsValueError(err error) bool {
	_, ok := err.(*ValueErr)
	return ok
}

func IsMemoryError(err error) bool {
	_, ok := err.(*MemoryError)
	return ok
}

func IsComputationError(err error) bool {
	_, ok := err.(*ComputationError)
	return ok
}

func IsIOError(err error) bool {
	_, ok := err.(*IOError)
	return ok
}

func IsValidationError(err error) bool {
	_, ok := err.(*ValidationError)
	return ok
}

// Comprehensive Structured Error Hierarchy System
// Production-grade error handling with categories, contexts, recovery suggestions, and serialization

// Severity defines error severity levels
type Severity int

const (
	SeverityDebug Severity = iota
	SeverityInfo
	SeverityWarning
	SeverityError
	SeverityCritical
	SeverityFatal
)

func (s Severity) String() string {
	switch s {
	case SeverityDebug:
		return "DEBUG"
	case SeverityInfo:
		return "INFO"
	case SeverityWarning:
		return "WARNING"
	case SeverityError:
		return "ERROR"
	case SeverityCritical:
		return "CRITICAL"
	case SeverityFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// ErrorCategory represents a high-level error category
type ErrorCategory struct {
	code        string
	description string
}

func NewErrorCategory(code, description string) ErrorCategory {
	return ErrorCategory{
		code:        code,
		description: description,
	}
}

func (ec ErrorCategory) Code() string {
	return ec.code
}

func (ec ErrorCategory) Description() string {
	return ec.description
}

// ErrorType represents a specific error type within a category
type ErrorType struct {
	category    ErrorCategory
	code        string
	description string
}

func NewErrorType(category ErrorCategory, code, description string) ErrorType {
	return ErrorType{
		category:    category,
		code:        code,
		description: description,
	}
}

func (et ErrorType) Category() ErrorCategory {
	return et.category
}

func (et ErrorType) Code() string {
	return fmt.Sprintf("%s.%s", et.category.code, et.code)
}

func (et ErrorType) Description() string {
	return et.description
}

// ActionType defines the type of recovery action
type ActionType int

const (
	ActionTypeRetry ActionType = iota
	ActionTypeFallback
	ActionTypeIgnore
	ActionTypeAbort
)

func (at ActionType) String() string {
	switch at {
	case ActionTypeRetry:
		return "RETRY"
	case ActionTypeFallback:
		return "FALLBACK"
	case ActionTypeIgnore:
		return "IGNORE"
	case ActionTypeAbort:
		return "ABORT"
	default:
		return "UNKNOWN"
	}
}

// RecoveryAction defines an action that can be taken to recover from an error
type RecoveryAction struct {
	ActionType  ActionType
	Description string
	AutoApply   bool
	Handler     func() error
}

// StructuredError represents a comprehensive error with context, recovery, and serialization
type StructuredError interface {
	error
	Type() ErrorType
	Severity() Severity
	Message() string
	Context() map[string]interface{}
	GetContext(key string) (interface{}, bool)
	WithContext(key string, value interface{}) StructuredError
	WithCause(cause StructuredError) StructuredError
	RootCause() StructuredError
	CauseChain() []StructuredError
	AddRecoverySuggestion(suggestion string)
	GetRecoverySuggestions() []string
	AddRecoveryAction(action RecoveryAction)
	GetRecoveryActions() []RecoveryAction
	DetailedError() string
	DebugString() string
	ToJSON() ([]byte, error)
	Unwrap() error
}

// structuredErrorImpl implements StructuredError
type structuredErrorImpl struct {
	errorType           ErrorType
	severity            Severity
	message             string
	context             map[string]interface{}
	cause               StructuredError
	recoverySuggestions []string
	recoveryActions     []RecoveryAction
	timestamp           time.Time
	mutex               sync.RWMutex
}

func NewStructuredError(errorType ErrorType, severity Severity, message string) StructuredError {
	return &structuredErrorImpl{
		errorType:           errorType,
		severity:            severity,
		message:             message,
		context:             make(map[string]interface{}),
		recoverySuggestions: make([]string, 0),
		recoveryActions:     make([]RecoveryAction, 0),
		timestamp:           time.Now(),
	}
}

func (se *structuredErrorImpl) Error() string {
	return fmt.Sprintf("[%s] %s: %s", se.errorType.Code(), se.severity.String(), se.message)
}

func (se *structuredErrorImpl) Type() ErrorType {
	return se.errorType
}

func (se *structuredErrorImpl) Severity() Severity {
	return se.severity
}

func (se *structuredErrorImpl) Message() string {
	return se.message
}

func (se *structuredErrorImpl) Context() map[string]interface{} {
	se.mutex.RLock()
	defer se.mutex.RUnlock()

	result := make(map[string]interface{})
	for k, v := range se.context {
		result[k] = v
	}
	return result
}

func (se *structuredErrorImpl) GetContext(key string) (interface{}, bool) {
	se.mutex.RLock()
	defer se.mutex.RUnlock()
	value, exists := se.context[key]
	return value, exists
}

func (se *structuredErrorImpl) WithContext(key string, value interface{}) StructuredError {
	se.mutex.Lock()
	defer se.mutex.Unlock()
	se.context[key] = value
	return se
}

func (se *structuredErrorImpl) WithCause(cause StructuredError) StructuredError {
	se.cause = cause
	return se
}

func (se *structuredErrorImpl) RootCause() StructuredError {
	current := se
	for current.cause != nil {
		if impl, ok := current.cause.(*structuredErrorImpl); ok {
			current = impl
		} else {
			break
		}
	}
	return current
}

func (se *structuredErrorImpl) CauseChain() []StructuredError {
	chain := make([]StructuredError, 0)
	current := se.cause
	for current != nil {
		chain = append(chain, current)
		if impl, ok := current.(*structuredErrorImpl); ok {
			current = impl.cause
		} else {
			break
		}
	}
	return chain
}

func (se *structuredErrorImpl) AddRecoverySuggestion(suggestion string) {
	se.mutex.Lock()
	defer se.mutex.Unlock()
	se.recoverySuggestions = append(se.recoverySuggestions, suggestion)
}

func (se *structuredErrorImpl) GetRecoverySuggestions() []string {
	se.mutex.RLock()
	defer se.mutex.RUnlock()
	result := make([]string, len(se.recoverySuggestions))
	copy(result, se.recoverySuggestions)
	return result
}

func (se *structuredErrorImpl) AddRecoveryAction(action RecoveryAction) {
	se.mutex.Lock()
	defer se.mutex.Unlock()
	se.recoveryActions = append(se.recoveryActions, action)
}

func (se *structuredErrorImpl) GetRecoveryActions() []RecoveryAction {
	se.mutex.RLock()
	defer se.mutex.RUnlock()
	result := make([]RecoveryAction, len(se.recoveryActions))
	copy(result, se.recoveryActions)
	return result
}

func (se *structuredErrorImpl) DetailedError() string {
	var builder strings.Builder
	builder.WriteString(se.Error())

	// Add context if available
	se.mutex.RLock()
	if len(se.context) > 0 {
		builder.WriteString(" | Context: ")
		contextParts := make([]string, 0, len(se.context))

		// Special handling for function and line context
		if function, ok := se.context["function"]; ok {
			if line, ok := se.context["line"]; ok {
				contextParts = append(contextParts, fmt.Sprintf("%v:%v", function, line))
			} else {
				contextParts = append(contextParts, fmt.Sprintf("function=%v", function))
			}
		}

		// Add other context
		for key, value := range se.context {
			if key != "function" && key != "line" {
				contextParts = append(contextParts, fmt.Sprintf("%s=%v", key, value))
			}
		}

		builder.WriteString(strings.Join(contextParts, ", "))
	}
	se.mutex.RUnlock()

	return builder.String()
}

func (se *structuredErrorImpl) DebugString() string {
	var builder strings.Builder
	builder.WriteString("=== Structured Error Debug Information ===\n")
	builder.WriteString(fmt.Sprintf("Type: %s\n", se.errorType.Code()))
	builder.WriteString(fmt.Sprintf("Severity: %s\n", se.severity.String()))
	builder.WriteString(fmt.Sprintf("Message: %s\n", se.message))
	builder.WriteString(fmt.Sprintf("Timestamp: %s\n", se.timestamp.Format(time.RFC3339)))

	// Context
	se.mutex.RLock()
	if len(se.context) > 0 {
		builder.WriteString("Context:\n")
		for key, value := range se.context {
			builder.WriteString(fmt.Sprintf("  %s: %v\n", key, value))
		}
	}
	se.mutex.RUnlock()

	// Recovery suggestions and actions
	se.mutex.RLock()
	hasRecovery := len(se.recoverySuggestions) > 0 || len(se.recoveryActions) > 0
	if hasRecovery {
		builder.WriteString("Recovery:\n")

		// Recovery suggestions
		if len(se.recoverySuggestions) > 0 {
			builder.WriteString("  Suggestions:\n")
			for i, suggestion := range se.recoverySuggestions {
				builder.WriteString(fmt.Sprintf("    %d. %s\n", i+1, suggestion))
			}
		}

		// Recovery actions
		if len(se.recoveryActions) > 0 {
			builder.WriteString("  Actions:\n")
			for i, action := range se.recoveryActions {
				builder.WriteString(fmt.Sprintf("    %d. %s (%s, auto=%t)\n",
					i+1, action.Description, action.ActionType.String(), action.AutoApply))
			}
		}
	}
	se.mutex.RUnlock()

	// Cause chain
	causes := se.CauseChain()
	if len(causes) > 0 {
		builder.WriteString("Cause Chain:\n")
		for i, cause := range causes {
			builder.WriteString(fmt.Sprintf("  %d. %s\n", i+1, cause.Error()))
		}
	}

	return builder.String()
}

func (se *structuredErrorImpl) ToJSON() ([]byte, error) {
	se.mutex.RLock()
	defer se.mutex.RUnlock()

	// Filter sensitive data from context
	filteredContext := make(map[string]interface{})
	for k, v := range se.context {
		if !isSensitiveField(k) {
			filteredContext[k] = v
		}
	}

	data := struct {
		Type        string                 `json:"type"`
		Severity    string                 `json:"severity"`
		Message     string                 `json:"message"`
		Context     map[string]interface{} `json:"context,omitempty"`
		Suggestions []string               `json:"suggestions,omitempty"`
		Timestamp   string                 `json:"timestamp"`
	}{
		Type:        se.errorType.Code(),
		Severity:    se.severity.String(),
		Message:     se.message,
		Context:     filteredContext,
		Suggestions: se.recoverySuggestions,
		Timestamp:   se.timestamp.Format(time.RFC3339),
	}

	return json.Marshal(data)
}

func (se *structuredErrorImpl) Unwrap() error {
	if se.cause != nil {
		return se.cause
	}
	return nil
}

// StructuredErrorFromJSON creates a StructuredError from JSON data
func StructuredErrorFromJSON(data []byte) (StructuredError, error) {
	var jsonData struct {
		Type        string                 `json:"type"`
		Severity    string                 `json:"severity"`
		Message     string                 `json:"message"`
		Context     map[string]interface{} `json:"context,omitempty"`
		Suggestions []string               `json:"suggestions,omitempty"`
		Timestamp   string                 `json:"timestamp"`
	}

	if err := json.Unmarshal(data, &jsonData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal structured error: %w", err)
	}

	// Parse type code
	parts := strings.Split(jsonData.Type, ".")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid error type format: %s", jsonData.Type)
	}

	category := NewErrorCategory(parts[0], fmt.Sprintf("Deserialized %s category", parts[0]))
	errorType := NewErrorType(category, parts[1], fmt.Sprintf("Deserialized %s error", parts[1]))

	// Parse severity
	var severity Severity
	switch jsonData.Severity {
	case "DEBUG":
		severity = SeverityDebug
	case "INFO":
		severity = SeverityInfo
	case "WARNING":
		severity = SeverityWarning
	case "ERROR":
		severity = SeverityError
	case "CRITICAL":
		severity = SeverityCritical
	case "FATAL":
		severity = SeverityFatal
	default:
		severity = SeverityError
	}

	// Create structured error
	err := NewStructuredError(errorType, severity, jsonData.Message).(*structuredErrorImpl)

	// Restore context
	for key, value := range jsonData.Context {
		err.WithContext(key, value)
	}

	// Restore suggestions
	for _, suggestion := range jsonData.Suggestions {
		err.AddRecoverySuggestion(suggestion)
	}

	return err, nil
}

// ErrorCollector aggregates multiple errors
type ErrorCollector struct {
	errors []StructuredError
	mutex  sync.RWMutex
}

func NewErrorCollector() *ErrorCollector {
	return &ErrorCollector{
		errors: make([]StructuredError, 0),
	}
}

func (ec *ErrorCollector) Add(err StructuredError) {
	ec.mutex.Lock()
	defer ec.mutex.Unlock()
	ec.errors = append(ec.errors, err)
}

func (ec *ErrorCollector) GetAll() []StructuredError {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()
	result := make([]StructuredError, len(ec.errors))
	copy(result, ec.errors)
	return result
}

func (ec *ErrorCollector) GetBySeverity(severity Severity) []StructuredError {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()
	result := make([]StructuredError, 0)
	for _, err := range ec.errors {
		if err.Severity() == severity {
			result = append(result, err)
		}
	}
	return result
}

func (ec *ErrorCollector) GetByType(errorType ErrorType) []StructuredError {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()
	result := make([]StructuredError, 0)
	for _, err := range ec.errors {
		if err.Type().Code() == errorType.Code() {
			result = append(result, err)
		}
	}
	return result
}

func (ec *ErrorCollector) CountByCategory(category ErrorCategory) int {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()
	count := 0
	for _, err := range ec.errors {
		if err.Type().Category().Code() == category.Code() {
			count++
		}
	}
	return count
}

// ErrorSummary provides a summary of collected errors
type ErrorSummary struct {
	TotalErrors   int
	CriticalCount int
	ErrorCount    int
	WarningCount  int
	ByCategory    map[string]int
}

func (ec *ErrorCollector) GetSummary() ErrorSummary {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()

	summary := ErrorSummary{
		TotalErrors: len(ec.errors),
		ByCategory:  make(map[string]int),
	}

	for _, err := range ec.errors {
		switch err.Severity() {
		case SeverityCritical, SeverityFatal:
			summary.CriticalCount++
		case SeverityError:
			summary.ErrorCount++
		case SeverityWarning:
			summary.WarningCount++
		}

		category := err.Type().Category().Code()
		summary.ByCategory[category]++
	}

	return summary
}

// BatchError represents multiple errors as a single error
type BatchError interface {
	error
	ErrorCount() int
	HighestSeverity() Severity
	GetErrors() []StructuredError
	ToJSON() ([]byte, error)
}

type batchErrorImpl struct {
	message         string
	errors          []StructuredError
	highestSeverity Severity
}

func (ec *ErrorCollector) ToBatchError(message string) BatchError {
	ec.mutex.RLock()
	defer ec.mutex.RUnlock()

	errors := make([]StructuredError, len(ec.errors))
	copy(errors, ec.errors)

	highest := SeverityDebug
	for _, err := range errors {
		if err.Severity() > highest {
			highest = err.Severity()
		}
	}

	return &batchErrorImpl{
		message:         message,
		errors:          errors,
		highestSeverity: highest,
	}
}

func (be *batchErrorImpl) Error() string {
	return fmt.Sprintf("%s (%d errors, highest severity: %s)",
		be.message, len(be.errors), be.highestSeverity.String())
}

func (be *batchErrorImpl) ErrorCount() int {
	return len(be.errors)
}

func (be *batchErrorImpl) HighestSeverity() Severity {
	return be.highestSeverity
}

func (be *batchErrorImpl) GetErrors() []StructuredError {
	result := make([]StructuredError, len(be.errors))
	copy(result, be.errors)
	return result
}

func (be *batchErrorImpl) ToJSON() ([]byte, error) {
	data := struct {
		Message         string            `json:"message"`
		ErrorCount      int               `json:"error_count"`
		HighestSeverity string            `json:"highest_severity"`
		Errors          []json.RawMessage `json:"errors"`
	}{
		Message:         be.message,
		ErrorCount:      len(be.errors),
		HighestSeverity: be.highestSeverity.String(),
		Errors:          make([]json.RawMessage, 0, len(be.errors)),
	}

	for _, err := range be.errors {
		if jsonData, err := err.ToJSON(); err == nil {
			data.Errors = append(data.Errors, json.RawMessage(jsonData))
		}
	}

	return json.Marshal(data)
}

// ErrorRateLimiter limits error reporting frequency
type ErrorRateLimiter struct {
	maxErrors       int
	timeWindow      time.Duration
	errorCounts     map[string]int
	windowStart     time.Time
	mutex           sync.RWMutex
	totalErrors     int
	allowedErrors   int
	throttledErrors int
}

func NewErrorRateLimiter(maxErrors int, timeWindowMs int) *ErrorRateLimiter {
	return &ErrorRateLimiter{
		maxErrors:   maxErrors,
		timeWindow:  time.Duration(timeWindowMs) * time.Millisecond,
		errorCounts: make(map[string]int),
		windowStart: time.Now(),
	}
}

func (erl *ErrorRateLimiter) ShouldReport(err StructuredError) bool {
	erl.mutex.Lock()
	defer erl.mutex.Unlock()

	now := time.Now()
	erl.totalErrors++

	// Reset window if expired
	if now.Sub(erl.windowStart) > erl.timeWindow {
		erl.errorCounts = make(map[string]int)
		erl.windowStart = now
	}

	errorKey := err.Type().Code()
	currentCount := erl.errorCounts[errorKey]

	if currentCount < erl.maxErrors {
		erl.errorCounts[errorKey]++
		erl.allowedErrors++
		return true
	}

	erl.throttledErrors++
	return false
}

// RateLimiterStats provides statistics about error rate limiting
type RateLimiterStats struct {
	TotalErrors     int
	AllowedErrors   int
	ThrottledErrors int
}

func (erl *ErrorRateLimiter) GetStats() RateLimiterStats {
	erl.mutex.RLock()
	defer erl.mutex.RUnlock()

	return RateLimiterStats{
		TotalErrors:     erl.totalErrors,
		AllowedErrors:   erl.allowedErrors,
		ThrottledErrors: erl.throttledErrors,
	}
}

// ErrorReporter handles error reporting to multiple destinations
type ErrorReporter struct {
	handlers map[string]func(StructuredError) error
	mutex    sync.RWMutex
}

func NewErrorReporter() *ErrorReporter {
	return &ErrorReporter{
		handlers: make(map[string]func(StructuredError) error),
	}
}

func (er *ErrorReporter) AddHandler(name string, handler func(StructuredError) error) {
	er.mutex.Lock()
	defer er.mutex.Unlock()
	er.handlers[name] = handler
}

func (er *ErrorReporter) Report(err StructuredError) error {
	er.mutex.RLock()
	handlers := make(map[string]func(StructuredError) error)
	for name, handler := range er.handlers {
		handlers[name] = handler
	}
	er.mutex.RUnlock()

	var lastErr error
	for _, handler := range handlers {
		if handlerErr := handler(err); handlerErr != nil {
			lastErr = handlerErr
		}
	}

	return lastErr
}

// AsyncErrorReporter handles asynchronous error reporting
type AsyncErrorReporter struct {
	*ErrorReporter
	errorChan chan StructuredError
	done      chan struct{}
}

func NewAsyncErrorReporter(bufferSize int) *AsyncErrorReporter {
	aer := &AsyncErrorReporter{
		ErrorReporter: NewErrorReporter(),
		errorChan:     make(chan StructuredError, bufferSize),
		done:          make(chan struct{}),
	}

	go aer.processErrors()
	return aer
}

func (aer *AsyncErrorReporter) processErrors() {
	for {
		select {
		case err := <-aer.errorChan:
			aer.ErrorReporter.Report(err)
		case <-aer.done:
			return
		}
	}
}

func (aer *AsyncErrorReporter) ReportAsync(err StructuredError) {
	select {
	case aer.errorChan <- err:
	default:
		// Buffer full, drop error
	}
}

func (aer *AsyncErrorReporter) Flush() {
	// Wait for all queued errors to be processed
	for len(aer.errorChan) > 0 {
		time.Sleep(1 * time.Millisecond)
	}
}

func (aer *AsyncErrorReporter) Close() {
	close(aer.done)
	close(aer.errorChan)
}

// Error wrapping and enhancement functions
func WrapError(originalErr error, message string) StructuredError {
	var errorType ErrorType

	// Map existing errors to structured types
	switch {
	case errors.Is(originalErr, ErrOutOfMemory):
		category := NewErrorCategory("MEMORY", "Memory management errors")
		errorType = NewErrorType(category, "OUT_OF_MEMORY", "System out of memory")
	case errors.Is(originalErr, ErrInvalidSize):
		category := NewErrorCategory("MEMORY", "Memory management errors")
		errorType = NewErrorType(category, "INVALID_SIZE", "Invalid memory size")
	case errors.Is(originalErr, ErrDoubleDeallocation):
		category := NewErrorCategory("MEMORY", "Memory management errors")
		errorType = NewErrorType(category, "DOUBLE_DEALLOCATION", "Double deallocation detected")
	default:
		category := NewErrorCategory("GENERAL", "General errors")
		errorType = NewErrorType(category, "WRAPPED_ERROR", "Wrapped error")
	}

	structuredErr := NewStructuredError(errorType, SeverityError, message)
	structuredErr.WithContext("original_error", originalErr.Error())

	return structuredErr
}

func EnhanceError(originalErr error, context map[string]interface{}) StructuredError {
	structuredErr := WrapError(originalErr, originalErr.Error())
	for key, value := range context {
		structuredErr.WithContext(key, value)
	}
	return structuredErr
}

// isSensitiveField checks if a field name contains sensitive information
func isSensitiveField(fieldName string) bool {
	sensitiveFields := []string{
		"password", "passwd", "pwd", "secret", "key", "token",
		"auth", "credential", "private", "api_key", "access_token",
		"refresh_token", "session_id", "cookie", "authorization",
		"signature", "hash", "salt", "nonce", "csrf",
	}

	lowerField := strings.ToLower(fieldName)
	for _, sensitive := range sensitiveFields {
		if strings.Contains(lowerField, sensitive) {
			return true
		}
	}
	return false
}
