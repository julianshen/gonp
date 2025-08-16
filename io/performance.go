package io

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// PerformanceOptions contains settings for optimizing I/O operations
type PerformanceOptions struct {
	// Buffer size for reading/writing operations
	BufferSize int

	// Number of worker goroutines for parallel processing
	Workers int

	// Chunk size for processing large files in chunks
	ChunkSize int

	// Enable memory-mapped file reading for large files
	UseMemoryMapping bool

	// Minimum file size to trigger memory mapping (in bytes)
	MemoryMapThreshold int64

	// Enable parallel processing for multi-core systems
	EnableParallel bool

	// Prefetch size for streaming operations
	PrefetchSize int
}

// DefaultPerformanceOptions returns optimized default settings
func DefaultPerformanceOptions() *PerformanceOptions {
	numCPU := runtime.NumCPU()

	return &PerformanceOptions{
		BufferSize:         64 * 1024,        // 64KB buffer
		Workers:            numCPU,           // One worker per CPU core
		ChunkSize:          10000,            // 10K rows per chunk
		UseMemoryMapping:   true,             // Enable for large files
		MemoryMapThreshold: 50 * 1024 * 1024, // 50MB threshold
		EnableParallel:     true,             // Enable parallel processing
		PrefetchSize:       1024 * 1024,      // 1MB prefetch
	}
}

// StreamingReader provides high-performance streaming file reading
type StreamingReader struct {
	file         *os.File
	reader       *bufio.Reader
	memoryMap    []byte
	currentPos   int64
	fileSize     int64
	options      *PerformanceOptions
	chunkChannel chan []byte
	errorChannel chan error
	ctx          context.Context
	cancel       context.CancelFunc
}

// NewStreamingReader creates a new high-performance file reader
func NewStreamingReader(filename string, options *PerformanceOptions) (*StreamingReader, error) {
	if options == nil {
		options = DefaultPerformanceOptions()
	}

	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}

	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to get file stats: %v", err)
	}

	fileSize := stat.Size()
	ctx, cancel := context.WithCancel(context.Background())

	sr := &StreamingReader{
		file:         file,
		fileSize:     fileSize,
		options:      options,
		chunkChannel: make(chan []byte, options.Workers),
		errorChannel: make(chan error, 1),
		ctx:          ctx,
		cancel:       cancel,
	}

	// Use memory mapping for large files
	if options.UseMemoryMapping && fileSize > options.MemoryMapThreshold {
		err = sr.initMemoryMapping()
		if err != nil {
			internal.DebugVerbose("Memory mapping failed, falling back to buffered reading: %v", err)
			sr.reader = bufio.NewReaderSize(file, options.BufferSize)
		}
	} else {
		sr.reader = bufio.NewReaderSize(file, options.BufferSize)
	}

	return sr, nil
}

// initMemoryMapping initializes memory-mapped file reading
func (sr *StreamingReader) initMemoryMapping() error {
	fd := int(sr.file.Fd())

	// Memory map the entire file
	data, err := syscall.Mmap(fd, 0, int(sr.fileSize), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return fmt.Errorf("mmap failed: %v", err)
	}

	sr.memoryMap = data
	internal.DebugVerbose("Memory-mapped file: %.2f MB", float64(sr.fileSize)/(1024*1024))
	return nil
}

// ReadChunk reads the next chunk of data
func (sr *StreamingReader) ReadChunk(size int) ([]byte, error) {
	if sr.memoryMap != nil {
		return sr.readChunkFromMemoryMap(size)
	}
	return sr.readChunkFromBuffer(size)
}

// readChunkFromMemoryMap reads from memory-mapped data
func (sr *StreamingReader) readChunkFromMemoryMap(size int) ([]byte, error) {
	if sr.currentPos >= sr.fileSize {
		return nil, io.EOF
	}

	endPos := sr.currentPos + int64(size)
	if endPos > sr.fileSize {
		endPos = sr.fileSize
	}

	chunk := make([]byte, endPos-sr.currentPos)
	copy(chunk, sr.memoryMap[sr.currentPos:endPos])
	sr.currentPos = endPos

	return chunk, nil
}

// readChunkFromBuffer reads from buffered reader
func (sr *StreamingReader) readChunkFromBuffer(size int) ([]byte, error) {
	chunk := make([]byte, size)
	n, err := sr.reader.Read(chunk)
	if err != nil && err != io.EOF {
		return nil, err
	}
	return chunk[:n], err
}

// StartStreaming begins streaming data processing
func (sr *StreamingReader) StartStreaming() {
	if !sr.options.EnableParallel {
		return
	}

	go func() {
		defer close(sr.chunkChannel)

		for {
			select {
			case <-sr.ctx.Done():
				return
			default:
				chunk, err := sr.ReadChunk(sr.options.ChunkSize)
				if err == io.EOF {
					return
				}
				if err != nil {
					sr.errorChannel <- err
					return
				}

				select {
				case sr.chunkChannel <- chunk:
				case <-sr.ctx.Done():
					return
				}
			}
		}
	}()
}

// GetChunkChannel returns the channel for receiving data chunks
func (sr *StreamingReader) GetChunkChannel() <-chan []byte {
	return sr.chunkChannel
}

// GetErrorChannel returns the channel for receiving errors
func (sr *StreamingReader) GetErrorChannel() <-chan error {
	return sr.errorChannel
}

// Close closes the reader and cleans up resources
func (sr *StreamingReader) Close() error {
	sr.cancel()

	if sr.memoryMap != nil {
		err := syscall.Munmap(sr.memoryMap)
		if err != nil {
			internal.DebugVerbose("Warning: failed to unmap memory: %v", err)
		}
	}

	return sr.file.Close()
}

// StreamingWriter provides high-performance streaming file writing
type StreamingWriter struct {
	file    *os.File
	writer  *bufio.Writer
	options *PerformanceOptions
}

// NewStreamingWriter creates a new high-performance file writer
func NewStreamingWriter(filename string, options *PerformanceOptions) (*StreamingWriter, error) {
	if options == nil {
		options = DefaultPerformanceOptions()
	}

	file, err := os.Create(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create file: %v", err)
	}

	writer := bufio.NewWriterSize(file, options.BufferSize)

	return &StreamingWriter{
		file:    file,
		writer:  writer,
		options: options,
	}, nil
}

// WriteChunk writes a chunk of data
func (sw *StreamingWriter) WriteChunk(data []byte) error {
	_, err := sw.writer.Write(data)
	return err
}

// Flush flushes any buffered data
func (sw *StreamingWriter) Flush() error {
	return sw.writer.Flush()
}

// Close closes the writer
func (sw *StreamingWriter) Close() error {
	err1 := sw.writer.Flush()
	err2 := sw.file.Close()

	if err1 != nil {
		return err1
	}
	return err2
}

// ParallelProcessor handles parallel processing of data chunks
type ParallelProcessor struct {
	workers    int
	chunkSize  int
	workerPool chan struct{}
	resultPool sync.Pool
}

// NewParallelProcessor creates a new parallel processor
func NewParallelProcessor(options *PerformanceOptions) *ParallelProcessor {
	if options == nil {
		options = DefaultPerformanceOptions()
	}

	pp := &ParallelProcessor{
		workers:    options.Workers,
		chunkSize:  options.ChunkSize,
		workerPool: make(chan struct{}, options.Workers),
	}

	// Initialize result pool for memory efficiency
	pp.resultPool.New = func() interface{} {
		return make([]interface{}, 0, options.ChunkSize)
	}

	// Fill worker pool
	for i := 0; i < options.Workers; i++ {
		pp.workerPool <- struct{}{}
	}

	return pp
}

// ProcessChunks processes data in parallel chunks
func (pp *ParallelProcessor) ProcessChunks(data [][]byte, processor func([]byte) ([]interface{}, error)) ([]interface{}, error) {
	numChunks := len(data)
	results := make([][]interface{}, numChunks)
	errors := make([]error, numChunks)

	var wg sync.WaitGroup

	for i, chunk := range data {
		wg.Add(1)
		go func(index int, chunkData []byte) {
			defer wg.Done()

			// Acquire worker
			<-pp.workerPool
			defer func() { pp.workerPool <- struct{}{} }()

			// Process chunk
			result, err := processor(chunkData)
			results[index] = result
			errors[index] = err
		}(i, chunk)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	// Combine results
	var combinedResults []interface{}
	for _, result := range results {
		combinedResults = append(combinedResults, result...)
	}

	return combinedResults, nil
}

// OptimizedCSVReader provides high-performance CSV reading
type OptimizedCSVReader struct {
	streamingReader *StreamingReader
	processor       *ParallelProcessor
	options         *PerformanceOptions
}

// NewOptimizedCSVReader creates a new optimized CSV reader
func NewOptimizedCSVReader(filename string, options *PerformanceOptions) (*OptimizedCSVReader, error) {
	if options == nil {
		options = DefaultPerformanceOptions()
	}

	streamingReader, err := NewStreamingReader(filename, options)
	if err != nil {
		return nil, err
	}

	processor := NewParallelProcessor(options)

	return &OptimizedCSVReader{
		streamingReader: streamingReader,
		processor:       processor,
		options:         options,
	}, nil
}

// ReadCSVOptimized reads a CSV file with performance optimizations
func (ocr *OptimizedCSVReader) ReadCSVOptimized() (*dataframe.DataFrame, error) {
	ctx := internal.StartProfiler("IO.ReadCSVOptimized")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	// For now, fallback to standard CSV reading with buffering optimizations
	// This provides a foundation for more advanced optimizations
	file, err := os.Open(ocr.streamingReader.file.Name())
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Use larger buffer for better performance
	reader := bufio.NewReaderSize(file, ocr.options.BufferSize)

	// Read and parse CSV with optimized buffering
	return readCSVWithOptimizedBuffer(reader)
}

// Close closes the optimized reader
func (ocr *OptimizedCSVReader) Close() error {
	return ocr.streamingReader.Close()
}

// readCSVWithOptimizedBuffer reads CSV with optimized buffering
func readCSVWithOptimizedBuffer(reader *bufio.Reader) (*dataframe.DataFrame, error) {
	// This is a simplified implementation that uses the existing CSV parser
	// but with optimized buffering. In a full implementation, this would
	// include custom CSV parsing optimized for performance.

	// Read header
	headerLine, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}

	// Parse header to get column names
	headerLine = headerLine[:len(headerLine)-1] // Remove newline
	columnNames := parseCSVLine(headerLine)

	// Initialize data containers
	numCols := len(columnNames)
	columnData := make([][]interface{}, numCols)
	for i := range columnData {
		columnData[i] = make([]interface{}, 0, 1000) // Pre-allocate for better performance
	}

	// Read data rows
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read data line: %v", err)
		}

		line = line[:len(line)-1] // Remove newline
		values := parseCSVLine(line)

		if len(values) != numCols {
			continue // Skip malformed rows
		}

		// Store values with type inference
		for i, val := range values {
			columnData[i] = append(columnData[i], inferValueType(val))
		}
	}

	// Create series for each column
	seriesList := make([]*series.Series, numCols)
	for i, colName := range columnNames {
		ser, err := series.FromValues(columnData[i], nil, colName)
		if err != nil {
			return nil, fmt.Errorf("failed to create series for column %s: %v", colName, err)
		}
		seriesList[i] = ser
	}

	// Create DataFrame
	return dataframe.FromSeries(seriesList)
}

// Helper functions

// parseCSVLine parses a CSV line into fields (simplified parser)
func parseCSVLine(line string) []string {
	var fields []string
	var currentField string
	inQuotes := false

	for i, char := range line {
		switch char {
		case '"':
			inQuotes = !inQuotes
		case ',':
			if !inQuotes {
				fields = append(fields, currentField)
				currentField = ""
			} else {
				currentField += string(char)
			}
		default:
			currentField += string(char)
		}

		// Handle end of line
		if i == len(line)-1 {
			fields = append(fields, currentField)
		}
	}

	return fields
}

// inferValueType infers the Go type for a string value
func inferValueType(value string) interface{} {
	if value == "" {
		return nil
	}

	// Try to parse as float64
	var f float64
	if n, err := fmt.Sscanf(value, "%f", &f); n == 1 && err == nil {
		// Check if it's actually an integer
		if f == float64(int64(f)) {
			return int64(f)
		}
		return f
	}

	// Try boolean
	if value == "true" || value == "True" || value == "TRUE" {
		return true
	}
	if value == "false" || value == "False" || value == "FALSE" {
		return false
	}

	// Default to string
	return value
}

// MemoryPool provides memory pooling for performance
type MemoryPool struct {
	pool sync.Pool
}

// NewMemoryPool creates a new memory pool
func NewMemoryPool(size int) *MemoryPool {
	return &MemoryPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, size)
			},
		},
	}
}

// Get gets a buffer from the pool
func (mp *MemoryPool) Get() []byte {
	return mp.pool.Get().([]byte)
}

// Put returns a buffer to the pool
func (mp *MemoryPool) Put(buf []byte) {
	// Reset the buffer
	buf = buf[:cap(buf)]
	for i := range buf {
		buf[i] = 0
	}
	mp.pool.Put(buf)
}

// GetFileSize returns the size of a file efficiently
func GetFileSize(filename string) (int64, error) {
	stat, err := os.Stat(filename)
	if err != nil {
		return 0, err
	}
	return stat.Size(), nil
}

// EstimateMemoryUsage estimates memory usage for loading a file
func EstimateMemoryUsage(filename string, estimateMultiplier float64) (int64, error) {
	fileSize, err := GetFileSize(filename)
	if err != nil {
		return 0, err
	}

	// Estimate memory usage as file size * multiplier
	// Typical multiplier: 2-4x for CSV (due to parsing overhead and type conversion)
	estimatedMemory := int64(float64(fileSize) * estimateMultiplier)

	return estimatedMemory, nil
}

// CheckAvailableMemory checks if enough memory is available
func CheckAvailableMemory(requiredMemory int64) bool {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Get system memory info (simplified approach)
	availableMemory := int64(m.Sys - m.HeapInuse)

	return availableMemory > requiredMemory
}

// PrefetchFile prefetches file content into system cache
// PrefetchFile is implemented per-OS to avoid cross-arch issues.

// ZeroCopyReader provides zero-copy reading capabilities
type ZeroCopyReader struct {
	data []byte
	pos  int
}

// NewZeroCopyReader creates a zero-copy reader from memory-mapped data
func NewZeroCopyReader(data []byte) *ZeroCopyReader {
	return &ZeroCopyReader{
		data: data,
		pos:  0,
	}
}

// ReadLine reads a line without copying data
func (zcr *ZeroCopyReader) ReadLine() ([]byte, error) {
	if zcr.pos >= len(zcr.data) {
		return nil, io.EOF
	}

	start := zcr.pos
	for zcr.pos < len(zcr.data) && zcr.data[zcr.pos] != '\n' {
		zcr.pos++
	}

	line := zcr.data[start:zcr.pos]
	zcr.pos++ // Skip newline

	return line, nil
}

// StringFromBytes converts bytes to string without copying (unsafe but fast)
func StringFromBytes(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}
