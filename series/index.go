package series

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"time"
)

// Index represents an index for labeling data in Series and DataFrame
type Index interface {
	// Len returns the number of elements in the index
	Len() int

	// Get returns the label at position i
	Get(i int) interface{}

	// Slice returns a new index with elements from start to end (exclusive)
	Slice(start, end int) Index

	// Loc finds the position of a label, returns position and whether found
	Loc(label interface{}) (int, bool)

	// Copy returns a deep copy of the index
	Copy() Index

	// Equal checks if this index is equal to another index
	Equal(other Index) bool

	// String returns a string representation of the index
	String() string

	// Type returns the underlying type of the index labels
	Type() reflect.Type

	// Values returns all index values as a slice
	Values() []interface{}

	// Append adds new values to the index and returns a new index
	Append(values ...interface{}) Index

	// Delete removes elements at specified positions and returns a new index
	Delete(positions ...int) Index

	// IsSorted returns true if the index is sorted
	IsSorted() bool

	// Sort returns a new sorted index and the permutation indices
	Sort() (Index, []int)
}

// RangeIndex represents a range-based integer index (like pandas RangeIndex)
type RangeIndex struct {
	start int
	stop  int
	step  int
	name  string
}

// NewRangeIndex creates a new RangeIndex
func NewRangeIndex(start, stop, step int) *RangeIndex {
	if step == 0 {
		panic("step cannot be zero")
	}
	return &RangeIndex{
		start: start,
		stop:  stop,
		step:  step,
	}
}

// NewDefaultRangeIndex creates a RangeIndex from 0 to length
func NewDefaultRangeIndex(length int) *RangeIndex {
	return NewRangeIndex(0, length, 1)
}

func (ri *RangeIndex) Len() int {
	if ri.step > 0 {
		if ri.start >= ri.stop {
			return 0
		}
		return (ri.stop - ri.start + ri.step - 1) / ri.step
	} else {
		if ri.start <= ri.stop {
			return 0
		}
		return (ri.start - ri.stop - ri.step - 1) / (-ri.step)
	}
}

func (ri *RangeIndex) Get(i int) interface{} {
	if i < 0 || i >= ri.Len() {
		panic(fmt.Sprintf("index %d out of bounds for RangeIndex of length %d", i, ri.Len()))
	}
	return ri.start + i*ri.step
}

func (ri *RangeIndex) Slice(start, end int) Index {
	length := ri.Len()
	if start < 0 {
		start = 0
	}
	if end > length {
		end = length
	}
	if start >= end {
		return NewRangeIndex(0, 0, 1)
	}

	newStart := ri.start + start*ri.step
	newStop := ri.start + end*ri.step
	return &RangeIndex{
		start: newStart,
		stop:  newStop,
		step:  ri.step,
		name:  ri.name,
	}
}

func (ri *RangeIndex) Loc(label interface{}) (int, bool) {
	val, ok := label.(int)
	if !ok {
		return -1, false
	}

	if ri.step > 0 {
		if val < ri.start || val >= ri.stop {
			return -1, false
		}
		if (val-ri.start)%ri.step != 0 {
			return -1, false
		}
		return (val - ri.start) / ri.step, true
	} else {
		if val > ri.start || val <= ri.stop {
			return -1, false
		}
		if (ri.start-val)%(-ri.step) != 0 {
			return -1, false
		}
		return (ri.start - val) / (-ri.step), true
	}
}

func (ri *RangeIndex) Copy() Index {
	return &RangeIndex{
		start: ri.start,
		stop:  ri.stop,
		step:  ri.step,
		name:  ri.name,
	}
}

func (ri *RangeIndex) Equal(other Index) bool {
	otherRi, ok := other.(*RangeIndex)
	if !ok {
		return false
	}
	return ri.start == otherRi.start && ri.stop == otherRi.stop && ri.step == otherRi.step
}

func (ri *RangeIndex) String() string {
	return fmt.Sprintf("RangeIndex(start=%d, stop=%d, step=%d)", ri.start, ri.stop, ri.step)
}

func (ri *RangeIndex) Type() reflect.Type {
	return reflect.TypeOf(int(0))
}

func (ri *RangeIndex) Values() []interface{} {
	length := ri.Len()
	values := make([]interface{}, length)
	for i := 0; i < length; i++ {
		values[i] = ri.Get(i)
	}
	return values
}

func (ri *RangeIndex) Append(values ...interface{}) Index {
	currentValues := ri.Values()
	allValues := append(currentValues, values...)
	return NewIndex(allValues)
}

func (ri *RangeIndex) Delete(positions ...int) Index {
	values := ri.Values()

	// Sort positions in descending order to delete from end to start
	sort.Sort(sort.Reverse(sort.IntSlice(positions)))

	for _, pos := range positions {
		if pos >= 0 && pos < len(values) {
			values = append(values[:pos], values[pos+1:]...)
		}
	}

	return NewIndex(values)
}

func (ri *RangeIndex) IsSorted() bool {
	return ri.step > 0 // Range index is sorted if step is positive
}

func (ri *RangeIndex) Sort() (Index, []int) {
	length := ri.Len()
	indices := make([]int, length)
	for i := range indices {
		indices[i] = i
	}

	if ri.step > 0 {
		// Already sorted
		return ri.Copy(), indices
	} else {
		// Reverse order
		for i := 0; i < length/2; i++ {
			indices[i], indices[length-1-i] = indices[length-1-i], indices[i]
		}
		newRi := NewRangeIndex(ri.stop+ri.step, ri.start+ri.step, -ri.step)
		return newRi, indices
	}
}

// Int64Index represents an index with int64 labels
type Int64Index struct {
	values []int64
	name   string
}

// NewInt64Index creates a new Int64Index
func NewInt64Index(values []int64) *Int64Index {
	return &Int64Index{values: append([]int64(nil), values...)}
}

func (ii *Int64Index) Len() int {
	return len(ii.values)
}

func (ii *Int64Index) Get(i int) interface{} {
	if i < 0 || i >= len(ii.values) {
		panic(fmt.Sprintf("index %d out of bounds for Int64Index of length %d", i, len(ii.values)))
	}
	return ii.values[i]
}

func (ii *Int64Index) Slice(start, end int) Index {
	if start < 0 {
		start = 0
	}
	if end > len(ii.values) {
		end = len(ii.values)
	}
	if start >= end {
		return NewInt64Index([]int64{})
	}
	return NewInt64Index(ii.values[start:end])
}

func (ii *Int64Index) Loc(label interface{}) (int, bool) {
	val, ok := label.(int64)
	if !ok {
		// Try converting from int
		if intVal, ok := label.(int); ok {
			val = int64(intVal)
		} else {
			return -1, false
		}
	}

	for i, v := range ii.values {
		if v == val {
			return i, true
		}
	}
	return -1, false
}

func (ii *Int64Index) Copy() Index {
	return NewInt64Index(ii.values)
}

func (ii *Int64Index) Equal(other Index) bool {
	otherIi, ok := other.(*Int64Index)
	if !ok {
		return false
	}
	if len(ii.values) != len(otherIi.values) {
		return false
	}
	for i, v := range ii.values {
		if v != otherIi.values[i] {
			return false
		}
	}
	return true
}

func (ii *Int64Index) String() string {
	return fmt.Sprintf("Int64Index(%v)", ii.values)
}

func (ii *Int64Index) Type() reflect.Type {
	return reflect.TypeOf(int64(0))
}

func (ii *Int64Index) Values() []interface{} {
	values := make([]interface{}, len(ii.values))
	for i, v := range ii.values {
		values[i] = v
	}
	return values
}

func (ii *Int64Index) Append(values ...interface{}) Index {
	newValues := append([]int64(nil), ii.values...)
	for _, v := range values {
		if val, ok := v.(int64); ok {
			newValues = append(newValues, val)
		} else if val, ok := v.(int); ok {
			newValues = append(newValues, int64(val))
		} else {
			panic(fmt.Sprintf("cannot append %T to Int64Index", v))
		}
	}
	return NewInt64Index(newValues)
}

func (ii *Int64Index) Delete(positions ...int) Index {
	newValues := append([]int64(nil), ii.values...)

	// Sort positions in descending order to delete from end to start
	sort.Sort(sort.Reverse(sort.IntSlice(positions)))

	for _, pos := range positions {
		if pos >= 0 && pos < len(newValues) {
			newValues = append(newValues[:pos], newValues[pos+1:]...)
		}
	}

	return NewInt64Index(newValues)
}

func (ii *Int64Index) IsSorted() bool {
	for i := 1; i < len(ii.values); i++ {
		if ii.values[i] < ii.values[i-1] {
			return false
		}
	}
	return true
}

func (ii *Int64Index) Sort() (Index, []int) {
	indices := make([]int, len(ii.values))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices based on values
	sort.Slice(indices, func(i, j int) bool {
		return ii.values[indices[i]] < ii.values[indices[j]]
	})

	// Create sorted values
	sortedValues := make([]int64, len(ii.values))
	for i, idx := range indices {
		sortedValues[i] = ii.values[idx]
	}

	return NewInt64Index(sortedValues), indices
}

// StringIndex represents an index with string labels
type StringIndex struct {
	values []string
	name   string
}

// NewStringIndex creates a new StringIndex
func NewStringIndex(values []string) *StringIndex {
	return &StringIndex{values: append([]string(nil), values...)}
}

func (si *StringIndex) Len() int {
	return len(si.values)
}

func (si *StringIndex) Get(i int) interface{} {
	if i < 0 || i >= len(si.values) {
		panic(fmt.Sprintf("index %d out of bounds for StringIndex of length %d", i, len(si.values)))
	}
	return si.values[i]
}

func (si *StringIndex) Slice(start, end int) Index {
	if start < 0 {
		start = 0
	}
	if end > len(si.values) {
		end = len(si.values)
	}
	if start >= end {
		return NewStringIndex([]string{})
	}
	return NewStringIndex(si.values[start:end])
}

func (si *StringIndex) Loc(label interface{}) (int, bool) {
	val, ok := label.(string)
	if !ok {
		return -1, false
	}

	for i, v := range si.values {
		if v == val {
			return i, true
		}
	}
	return -1, false
}

func (si *StringIndex) Copy() Index {
	return NewStringIndex(si.values)
}

func (si *StringIndex) Equal(other Index) bool {
	otherSi, ok := other.(*StringIndex)
	if !ok {
		return false
	}
	if len(si.values) != len(otherSi.values) {
		return false
	}
	for i, v := range si.values {
		if v != otherSi.values[i] {
			return false
		}
	}
	return true
}

func (si *StringIndex) String() string {
	return fmt.Sprintf("StringIndex(%v)", si.values)
}

func (si *StringIndex) Type() reflect.Type {
	return reflect.TypeOf("")
}

func (si *StringIndex) Values() []interface{} {
	values := make([]interface{}, len(si.values))
	for i, v := range si.values {
		values[i] = v
	}
	return values
}

func (si *StringIndex) Append(values ...interface{}) Index {
	newValues := append([]string(nil), si.values...)
	for _, v := range values {
		if val, ok := v.(string); ok {
			newValues = append(newValues, val)
		} else {
			newValues = append(newValues, fmt.Sprintf("%v", v))
		}
	}
	return NewStringIndex(newValues)
}

func (si *StringIndex) Delete(positions ...int) Index {
	newValues := append([]string(nil), si.values...)

	// Sort positions in descending order to delete from end to start
	sort.Sort(sort.Reverse(sort.IntSlice(positions)))

	for _, pos := range positions {
		if pos >= 0 && pos < len(newValues) {
			newValues = append(newValues[:pos], newValues[pos+1:]...)
		}
	}

	return NewStringIndex(newValues)
}

func (si *StringIndex) IsSorted() bool {
	for i := 1; i < len(si.values); i++ {
		if strings.Compare(si.values[i], si.values[i-1]) < 0 {
			return false
		}
	}
	return true
}

func (si *StringIndex) Sort() (Index, []int) {
	indices := make([]int, len(si.values))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices based on values
	sort.Slice(indices, func(i, j int) bool {
		return strings.Compare(si.values[indices[i]], si.values[indices[j]]) < 0
	})

	// Create sorted values
	sortedValues := make([]string, len(si.values))
	for i, idx := range indices {
		sortedValues[i] = si.values[idx]
	}

	return NewStringIndex(sortedValues), indices
}

// DateTimeIndex represents an index with time.Time labels
type DateTimeIndex struct {
	values []time.Time
	name   string
}

// NewDateTimeIndex creates a new DateTimeIndex
func NewDateTimeIndex(values []time.Time) *DateTimeIndex {
	return &DateTimeIndex{values: append([]time.Time(nil), values...)}
}

func (dti *DateTimeIndex) Len() int {
	return len(dti.values)
}

func (dti *DateTimeIndex) Get(i int) interface{} {
	if i < 0 || i >= len(dti.values) {
		panic(fmt.Sprintf("index %d out of bounds for DateTimeIndex of length %d", i, len(dti.values)))
	}
	return dti.values[i]
}

func (dti *DateTimeIndex) Slice(start, end int) Index {
	if start < 0 {
		start = 0
	}
	if end > len(dti.values) {
		end = len(dti.values)
	}
	if start >= end {
		return NewDateTimeIndex([]time.Time{})
	}
	return NewDateTimeIndex(dti.values[start:end])
}

func (dti *DateTimeIndex) Loc(label interface{}) (int, bool) {
	val, ok := label.(time.Time)
	if !ok {
		return -1, false
	}

	for i, v := range dti.values {
		if v.Equal(val) {
			return i, true
		}
	}
	return -1, false
}

func (dti *DateTimeIndex) Copy() Index {
	return NewDateTimeIndex(dti.values)
}

func (dti *DateTimeIndex) Equal(other Index) bool {
	otherDti, ok := other.(*DateTimeIndex)
	if !ok {
		return false
	}
	if len(dti.values) != len(otherDti.values) {
		return false
	}
	for i, v := range dti.values {
		if !v.Equal(otherDti.values[i]) {
			return false
		}
	}
	return true
}

func (dti *DateTimeIndex) String() string {
	return fmt.Sprintf("DateTimeIndex(%v)", dti.values)
}

func (dti *DateTimeIndex) Type() reflect.Type {
	return reflect.TypeOf(time.Time{})
}

func (dti *DateTimeIndex) Values() []interface{} {
	values := make([]interface{}, len(dti.values))
	for i, v := range dti.values {
		values[i] = v
	}
	return values
}

func (dti *DateTimeIndex) Append(values ...interface{}) Index {
	newValues := append([]time.Time(nil), dti.values...)
	for _, v := range values {
		if val, ok := v.(time.Time); ok {
			newValues = append(newValues, val)
		} else {
			panic(fmt.Sprintf("cannot append %T to DateTimeIndex", v))
		}
	}
	return NewDateTimeIndex(newValues)
}

func (dti *DateTimeIndex) Delete(positions ...int) Index {
	newValues := append([]time.Time(nil), dti.values...)

	// Sort positions in descending order to delete from end to start
	sort.Sort(sort.Reverse(sort.IntSlice(positions)))

	for _, pos := range positions {
		if pos >= 0 && pos < len(newValues) {
			newValues = append(newValues[:pos], newValues[pos+1:]...)
		}
	}

	return NewDateTimeIndex(newValues)
}

func (dti *DateTimeIndex) IsSorted() bool {
	for i := 1; i < len(dti.values); i++ {
		if dti.values[i].Before(dti.values[i-1]) {
			return false
		}
	}
	return true
}

func (dti *DateTimeIndex) Sort() (Index, []int) {
	indices := make([]int, len(dti.values))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices based on values
	sort.Slice(indices, func(i, j int) bool {
		return dti.values[indices[i]].Before(dti.values[indices[j]])
	})

	// Create sorted values
	sortedValues := make([]time.Time, len(dti.values))
	for i, idx := range indices {
		sortedValues[i] = dti.values[idx]
	}

	return NewDateTimeIndex(sortedValues), indices
}

// NewIndex creates an appropriate Index based on the input values
func NewIndex(values []interface{}) Index {
	if len(values) == 0 {
		return NewDefaultRangeIndex(0)
	}

	// Determine the type of the first non-nil value
	var sampleType reflect.Type
	for _, v := range values {
		if v != nil {
			sampleType = reflect.TypeOf(v)
			break
		}
	}

	if sampleType == nil {
		// All values are nil, default to string index
		strValues := make([]string, len(values))
		for i := range strValues {
			strValues[i] = ""
		}
		return NewStringIndex(strValues)
	}

	switch sampleType.Kind() {
	case reflect.Int, reflect.Int64:
		int64Values := make([]int64, len(values))
		for i, v := range values {
			if v == nil {
				int64Values[i] = 0 // or handle NaN equivalent
			} else if val, ok := v.(int); ok {
				int64Values[i] = int64(val)
			} else if val, ok := v.(int64); ok {
				int64Values[i] = val
			} else {
				// Fallback to string index for mixed types
				goto stringIndex
			}
		}
		return NewInt64Index(int64Values)

	case reflect.String:
		strValues := make([]string, len(values))
		for i, v := range values {
			if v == nil {
				strValues[i] = ""
			} else if val, ok := v.(string); ok {
				strValues[i] = val
			} else {
				// Fallback to string index for mixed types
				strValues[i] = fmt.Sprintf("%v", v)
			}
		}
		return NewStringIndex(strValues)

	default:
		if sampleType == reflect.TypeOf(time.Time{}) {
			timeValues := make([]time.Time, len(values))
			for i, v := range values {
				if v == nil {
					timeValues[i] = time.Time{} // zero time
				} else if val, ok := v.(time.Time); ok {
					timeValues[i] = val
				} else {
					// Fallback to string index for mixed types
					goto stringIndex
				}
			}
			return NewDateTimeIndex(timeValues)
		}
	}

stringIndex:
	// Default fallback: convert everything to strings
	strValues := make([]string, len(values))
	for i, v := range values {
		if v == nil {
			strValues[i] = ""
		} else {
			strValues[i] = fmt.Sprintf("%v", v)
		}
	}
	return NewStringIndex(strValues)
}
