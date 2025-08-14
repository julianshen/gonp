package series

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/julianshen/gonp/array"
	"github.com/julianshen/gonp/internal"
)

// StringAccessor provides string methods for Series containing string data
type StringAccessor struct {
	series *Series
}

// Str returns a StringAccessor for string operations
func (s *Series) Str() *StringAccessor {
	return &StringAccessor{series: s}
}

// Basic string operations

// Lower converts all strings to lowercase
func (sa *StringAccessor) Lower() (*Series, error) {
	return sa.applyStringFunction(strings.ToLower)
}

// Upper converts all strings to uppercase
func (sa *StringAccessor) Upper() (*Series, error) {
	return sa.applyStringFunction(strings.ToUpper)
}

// Title converts strings to title case
func (sa *StringAccessor) Title() (*Series, error) {
	return sa.applyStringFunction(strings.Title)
}

// Capitalize capitalizes the first character of each string
func (sa *StringAccessor) Capitalize() (*Series, error) {
	capitalizeFunc := func(s string) string {
		if len(s) == 0 {
			return s
		}
		runes := []rune(s)
		runes[0] = unicode.ToUpper(runes[0])
		for i := 1; i < len(runes); i++ {
			runes[i] = unicode.ToLower(runes[i])
		}
		return string(runes)
	}
	return sa.applyStringFunction(capitalizeFunc)
}

// Strip removes leading and trailing whitespace
func (sa *StringAccessor) Strip() (*Series, error) {
	return sa.applyStringFunction(strings.TrimSpace)
}

// LStrip removes leading whitespace
func (sa *StringAccessor) LStrip() (*Series, error) {
	return sa.applyStringFunction(func(s string) string {
		return strings.TrimLeftFunc(s, unicode.IsSpace)
	})
}

// RStrip removes trailing whitespace
func (sa *StringAccessor) RStrip() (*Series, error) {
	return sa.applyStringFunction(func(s string) string {
		return strings.TrimRightFunc(s, unicode.IsSpace)
	})
}

// String manipulation

// Replace replaces occurrences of old string with new string
func (sa *StringAccessor) Replace(old, new string, n int) (*Series, error) {
	replaceFunc := func(s string) string {
		if n < 0 {
			return strings.ReplaceAll(s, old, new)
		}
		return strings.Replace(s, old, new, n)
	}
	return sa.applyStringFunction(replaceFunc)
}

// ReplaceAll replaces all occurrences of old string with new string
func (sa *StringAccessor) ReplaceAll(old, new string) (*Series, error) {
	return sa.Replace(old, new, -1)
}

// Slice extracts substring from start to end position
func (sa *StringAccessor) Slice(start, end int) (*Series, error) {
	sliceFunc := func(s string) string {
		runes := []rune(s)
		length := len(runes)

		// Handle negative indices
		if start < 0 {
			start = length + start
		}
		if end < 0 {
			end = length + end
		}

		// Bounds checking
		if start < 0 {
			start = 0
		}
		if end > length {
			end = length
		}
		if start >= end {
			return ""
		}

		return string(runes[start:end])
	}
	return sa.applyStringFunction(sliceFunc)
}

// SliceReplace replaces substring at specified positions
func (sa *StringAccessor) SliceReplace(start, end int, replacement string) (*Series, error) {
	sliceReplaceFunc := func(s string) string {
		runes := []rune(s)
		length := len(runes)

		// Handle negative indices
		if start < 0 {
			start = length + start
		}
		if end < 0 {
			end = length + end
		}

		// Bounds checking
		if start < 0 {
			start = 0
		}
		if end > length {
			end = length
		}
		if start >= end {
			start = end
		}

		before := string(runes[:start])
		after := string(runes[end:])
		return before + replacement + after
	}
	return sa.applyStringFunction(sliceReplaceFunc)
}

// Pad operations

// Pad pads strings to specified width
func (sa *StringAccessor) Pad(width int, side string, fillchar string) (*Series, error) {
	if len(fillchar) != 1 {
		return nil, fmt.Errorf("fillchar must be exactly one character")
	}

	padFunc := func(s string) string {
		currentLen := len([]rune(s))
		if currentLen >= width {
			return s
		}

		padLen := width - currentLen
		padding := strings.Repeat(fillchar, padLen)

		switch side {
		case "left":
			return padding + s
		case "right":
			return s + padding
		case "both":
			leftPad := padLen / 2
			rightPad := padLen - leftPad
			return strings.Repeat(fillchar, leftPad) + s + strings.Repeat(fillchar, rightPad)
		default:
			return s + padding // Default to right padding
		}
	}
	return sa.applyStringFunction(padFunc)
}

// LJust left-justifies strings in a field of given width
func (sa *StringAccessor) LJust(width int, fillchar string) (*Series, error) {
	return sa.Pad(width, "right", fillchar)
}

// RJust right-justifies strings in a field of given width
func (sa *StringAccessor) RJust(width int, fillchar string) (*Series, error) {
	return sa.Pad(width, "left", fillchar)
}

// Center centers strings in a field of given width
func (sa *StringAccessor) Center(width int, fillchar string) (*Series, error) {
	return sa.Pad(width, "both", fillchar)
}

// ZFill pads strings with zeros on the left
func (sa *StringAccessor) ZFill(width int) (*Series, error) {
	return sa.Pad(width, "left", "0")
}

// String tests (return boolean Series)

// Contains tests if each string contains the substring
func (sa *StringAccessor) Contains(substr string, regex bool) (*Series, error) {
	if regex {
		re, err := regexp.Compile(substr)
		if err != nil {
			return nil, fmt.Errorf("invalid regex pattern: %v", err)
		}
		return sa.applyStringTest(func(s string) bool {
			return re.MatchString(s)
		})
	}
	return sa.applyStringTest(func(s string) bool {
		return strings.Contains(s, substr)
	})
}

// StartsWith tests if each string starts with the prefix
func (sa *StringAccessor) StartsWith(prefix string) (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		return strings.HasPrefix(s, prefix)
	})
}

// EndsWith tests if each string ends with the suffix
func (sa *StringAccessor) EndsWith(suffix string) (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		return strings.HasSuffix(s, suffix)
	})
}

// IsAlpha tests if all characters in each string are alphabetic
func (sa *StringAccessor) IsAlpha() (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		if len(s) == 0 {
			return false
		}
		for _, r := range s {
			if !unicode.IsLetter(r) {
				return false
			}
		}
		return true
	})
}

// IsDigit tests if all characters in each string are digits
func (sa *StringAccessor) IsDigit() (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		if len(s) == 0 {
			return false
		}
		for _, r := range s {
			if !unicode.IsDigit(r) {
				return false
			}
		}
		return true
	})
}

// IsAlnum tests if all characters in each string are alphanumeric
func (sa *StringAccessor) IsAlnum() (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		if len(s) == 0 {
			return false
		}
		for _, r := range s {
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
				return false
			}
		}
		return true
	})
}

// IsLower tests if all characters in each string are lowercase
func (sa *StringAccessor) IsLower() (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		if len(s) == 0 {
			return false
		}
		hasLetter := false
		for _, r := range s {
			if unicode.IsLetter(r) {
				hasLetter = true
				if !unicode.IsLower(r) {
					return false
				}
			}
		}
		return hasLetter
	})
}

// IsUpper tests if all characters in each string are uppercase
func (sa *StringAccessor) IsUpper() (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		if len(s) == 0 {
			return false
		}
		hasLetter := false
		for _, r := range s {
			if unicode.IsLetter(r) {
				hasLetter = true
				if !unicode.IsUpper(r) {
					return false
				}
			}
		}
		return hasLetter
	})
}

// IsSpace tests if all characters in each string are whitespace
func (sa *StringAccessor) IsSpace() (*Series, error) {
	return sa.applyStringTest(func(s string) bool {
		if len(s) == 0 {
			return false
		}
		for _, r := range s {
			if !unicode.IsSpace(r) {
				return false
			}
		}
		return true
	})
}

// String information methods

// Len returns the length of each string
func (sa *StringAccessor) Len() (*Series, error) {
	result := array.Empty(internal.Shape{sa.series.Len()}, internal.Int64)

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var length int64
		if strVal, ok := val.(string); ok {
			length = int64(len([]rune(strVal))) // Count Unicode characters, not bytes
		} else {
			length = 0 // Non-string values have length 0
		}

		err := result.Set(length, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set length at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// Count counts occurrences of substring in each string
func (sa *StringAccessor) Count(substr string) (*Series, error) {
	result := array.Empty(internal.Shape{sa.series.Len()}, internal.Int64)

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var count int64
		if strVal, ok := val.(string); ok {
			count = int64(strings.Count(strVal, substr))
		} else {
			count = 0
		}

		err := result.Set(count, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set count at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// Find returns the lowest index where substring is found
func (sa *StringAccessor) Find(substr string) (*Series, error) {
	result := array.Empty(internal.Shape{sa.series.Len()}, internal.Int64)

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var index int64
		if strVal, ok := val.(string); ok {
			idx := strings.Index(strVal, substr)
			if idx == -1 {
				index = -1
			} else {
				// Convert byte index to rune index
				runeIndex := len([]rune(strVal[:idx]))
				index = int64(runeIndex)
			}
		} else {
			index = -1
		}

		err := result.Set(index, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set index at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// Split operations

// Split splits each string by delimiter and returns a Series of slices
func (sa *StringAccessor) Split(sep string, n int) (*Series, error) {
	result := array.Empty(internal.Shape{sa.series.Len()}, sa.series.DType()) // Using interface{} essentially

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var splitResult interface{}
		if strVal, ok := val.(string); ok {
			if n < 0 {
				splitResult = strings.Split(strVal, sep)
			} else {
				splitResult = strings.SplitN(strVal, sep, n)
			}
		} else {
			splitResult = []string{} // Empty slice for non-string values
		}

		err := result.Set(splitResult, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set split result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// Extract extracts capture groups from regex pattern
func (sa *StringAccessor) Extract(pattern string) (*Series, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid regex pattern: %v", err)
	}

	result := array.Empty(internal.Shape{sa.series.Len()}, sa.series.DType()) // Using interface{} essentially

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var extractResult interface{}
		if strVal, ok := val.(string); ok {
			matches := re.FindStringSubmatch(strVal)
			if len(matches) > 1 {
				// Return capture groups (excluding full match at index 0)
				extractResult = matches[1:]
			} else {
				extractResult = []string{}
			}
		} else {
			extractResult = []string{}
		}

		err := result.Set(extractResult, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set extract result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// Type conversion methods

// AsType converts string Series to specified type
func (sa *StringAccessor) AsType(dtype internal.DType) (*Series, error) {
	result := array.Empty(internal.Shape{sa.series.Len()}, dtype)

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var convertedVal interface{}
		var err error

		if strVal, ok := val.(string); ok {
			convertedVal, err = sa.convertStringToType(strVal, dtype)
			if err != nil {
				// Set appropriate NaN/zero value for conversion errors
				convertedVal = getNaNValue(dtype)
			}
		} else {
			convertedVal = getNaNValue(dtype)
		}

		err = result.Set(convertedVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set converted value at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// ToNumeric converts strings to numeric values
func (sa *StringAccessor) ToNumeric() (*Series, error) {
	return sa.AsType(internal.Float64)
}

// Helper methods

// applyStringFunction applies a function to each string in the Series
func (sa *StringAccessor) applyStringFunction(fn func(string) string) (*Series, error) {
	// Use the same dtype as the original series for string operations
	result := array.Empty(internal.Shape{sa.series.Len()}, sa.series.DType())

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var resultVal interface{}
		if strVal, ok := val.(string); ok {
			resultVal = fn(strVal)
		} else {
			resultVal = val // Keep non-string values as-is
		}

		err := result.Set(resultVal, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set string function result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// applyStringTest applies a boolean test function to each string in the Series
func (sa *StringAccessor) applyStringTest(fn func(string) bool) (*Series, error) {
	result := array.Empty(internal.Shape{sa.series.Len()}, internal.Bool)

	for i := 0; i < sa.series.Len(); i++ {
		val := sa.series.data.At(i)

		var testResult bool
		if strVal, ok := val.(string); ok {
			testResult = fn(strVal)
		} else {
			testResult = false // Non-string values fail all tests
		}

		err := result.Set(testResult, i)
		if err != nil {
			return nil, fmt.Errorf("failed to set string test result at index %d: %v", i, err)
		}
	}

	return NewSeries(result, sa.series.index.Copy(), sa.series.name)
}

// convertStringToType converts a string to the specified type
func (sa *StringAccessor) convertStringToType(s string, dtype internal.DType) (interface{}, error) {
	switch dtype {
	case internal.Float64:
		return strconv.ParseFloat(s, 64)
	case internal.Float32:
		f, err := strconv.ParseFloat(s, 32)
		return float32(f), err
	case internal.Int64:
		return strconv.ParseInt(s, 10, 64)
	case internal.Int32:
		i, err := strconv.ParseInt(s, 10, 32)
		return int32(i), err
	case internal.Int16:
		i, err := strconv.ParseInt(s, 10, 16)
		return int16(i), err
	case internal.Int8:
		i, err := strconv.ParseInt(s, 10, 8)
		return int8(i), err
	case internal.Uint64:
		return strconv.ParseUint(s, 10, 64)
	case internal.Uint32:
		u, err := strconv.ParseUint(s, 10, 32)
		return uint32(u), err
	case internal.Uint16:
		u, err := strconv.ParseUint(s, 10, 16)
		return uint16(u), err
	case internal.Uint8:
		u, err := strconv.ParseUint(s, 10, 8)
		return uint8(u), err
	case internal.Bool:
		return strconv.ParseBool(s)
	default:
		return s, nil // Return as string if unsupported type
	}
}

// getNaNValue function already exists in operations.go
