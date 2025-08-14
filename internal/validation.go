package internal

import (
	"fmt"
	"math"
)

// Validator provides parameter validation utilities
type Validator struct {
	context string // Context for error messages (e.g., function name)
}

// NewValidator creates a new validator with context
func NewValidator(context string) *Validator {
	return &Validator{context: context}
}

// ValidateNotNil checks if a pointer is not nil
func (v *Validator) ValidateNotNil(value interface{}, paramName string) error {
	if value == nil {
		return NewValidationError(v.context, paramName, value)
	}
	return nil
}

// ValidatePositive checks if a numeric value is positive
func (v *Validator) ValidatePositive(value float64, paramName string) error {
	if value <= 0 {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' must be positive, got %f", paramName, value))
	}
	return nil
}

// ValidateNonNegative checks if a numeric value is non-negative
func (v *Validator) ValidateNonNegative(value float64, paramName string) error {
	if value < 0 {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' must be non-negative, got %f", paramName, value))
	}
	return nil
}

// ValidateFinite checks if a float value is finite (not NaN or Inf)
func (v *Validator) ValidateFinite(value float64, paramName string) error {
	if math.IsNaN(value) {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' cannot be NaN", paramName))
	}
	if math.IsInf(value, 0) {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' cannot be infinite", paramName))
	}
	return nil
}

// ValidateRange checks if a value is within a specified range (inclusive)
func (v *Validator) ValidateRange(value, min, max float64, paramName string) error {
	if value < min || value > max {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' must be in range [%f, %f], got %f", paramName, min, max, value))
	}
	return nil
}

// ValidateIntRange checks if an integer value is within a specified range (inclusive)
func (v *Validator) ValidateIntRange(value, min, max int, paramName string) error {
	if value < min || value > max {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' must be in range [%d, %d], got %d", paramName, min, max, value))
	}
	return nil
}

// ValidateArrayNotEmpty checks if an array has at least one element
func (v *Validator) ValidateArrayNotEmpty(size int, paramName string) error {
	if size == 0 {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' cannot be empty array", paramName))
	}
	return nil
}

// ValidateShapeCompatible checks if two shapes are compatible for operations
func (v *Validator) ValidateShapeCompatible(shape1, shape2 Shape, operation string) error {
	if !shape1.Equal(shape2) {
		return NewShapeError(v.context, shape1, shape2)
	}
	return nil
}

// ValidateIndexBounds checks if an index is within array bounds
func (v *Validator) ValidateIndexBounds(index int, size int, paramName string) error {
	if index < 0 || index >= size {
		return NewIndexErrorWithMsg(v.context,
			fmt.Sprintf("index '%s' %d out of bounds for size %d", paramName, index, size))
	}
	return nil
}

// ValidateSliceBounds checks if slice bounds are valid
func (v *Validator) ValidateSliceBounds(start, end, size int) error {
	if start < 0 {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("slice start %d cannot be negative", start))
	}
	if end > size {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("slice end %d exceeds array size %d", end, size))
	}
	if start > end {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("slice start %d cannot be greater than end %d", start, end))
	}
	return nil
}

// ValidateAxis checks if an axis is valid for the given shape
func (v *Validator) ValidateAxis(axis int, shape Shape, paramName string) error {
	ndim := len(shape)
	if axis < -ndim || axis >= ndim {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("axis '%s' %d out of bounds for %d-dimensional array", paramName, axis, ndim))
	}
	return nil
}

// ValidateDTypeSupported checks if a data type is supported for an operation
func (v *Validator) ValidateDTypeSupported(dtype DType, supportedTypes []DType, operation string) error {
	for _, supported := range supportedTypes {
		if dtype == supported {
			return nil
		}
	}
	return NewValidationErrorWithMsg(v.context,
		fmt.Sprintf("operation '%s' does not support data type %v", operation, dtype))
}

// ValidateArraySameDType checks if arrays have the same data type
func (v *Validator) ValidateArraySameDType(dtype1, dtype2 DType, operation string) error {
	if dtype1 != dtype2 {
		return NewTypeError(v.context, dtype1, dtype2)
	}
	return nil
}

// ValidateQuantile checks if a quantile value is valid (0-1)
func (v *Validator) ValidateQuantile(q float64, paramName string) error {
	return v.ValidateRange(q, 0.0, 1.0, paramName)
}

// ValidateFilename checks if a filename is not empty and has valid characters
func (v *Validator) ValidateFilename(filename string, paramName string) error {
	if filename == "" {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' filename cannot be empty", paramName))
	}
	// Additional filename validation could be added here
	return nil
}

// ValidateStringNotEmpty checks if a string is not empty
func (v *Validator) ValidateStringNotEmpty(value string, paramName string) error {
	if value == "" {
		return NewValidationErrorWithMsg(v.context,
			fmt.Sprintf("parameter '%s' cannot be empty string", paramName))
	}
	return nil
}

// Batch validation functions

// ValidateAll runs multiple validation functions and returns the first error
func (v *Validator) ValidateAll(validations ...func() error) error {
	for _, validate := range validations {
		if err := validate(); err != nil {
			return err
		}
	}
	return nil
}

// ValidateAllCollectErrors runs multiple validation functions and collects all errors
func (v *Validator) ValidateAllCollectErrors(validations ...func() error) []error {
	var errors []error
	for _, validate := range validations {
		if err := validate(); err != nil {
			errors = append(errors, err)
		}
	}
	return errors
}

// Quick validation functions (without creating a Validator instance)

// QuickValidateNotNil is a convenience function for nil checking
func QuickValidateNotNil(value interface{}, context, paramName string) error {
	if value == nil {
		return NewValidationError(context, paramName, value)
	}
	return nil
}

// QuickValidatePositive is a convenience function for positive number checking
func QuickValidatePositive(value float64, context, paramName string) error {
	if value <= 0 {
		return NewValidationErrorWithMsg(context,
			fmt.Sprintf("parameter '%s' must be positive, got %f", paramName, value))
	}
	return nil
}

// QuickValidateIndexBounds is a convenience function for index validation
func QuickValidateIndexBounds(index, size int, context, paramName string) error {
	if index < 0 || index >= size {
		return NewIndexErrorWithMsg(context,
			fmt.Sprintf("index '%s' %d out of bounds for size %d", paramName, index, size))
	}
	return nil
}

// QuickValidateArrayNotEmpty is a convenience function for empty array checking
func QuickValidateArrayNotEmpty(size int, context, paramName string) error {
	if size == 0 {
		return NewValidationErrorWithMsg(context,
			fmt.Sprintf("parameter '%s' cannot be empty array", paramName))
	}
	return nil
}
