// Package gpu provides distributed computing with MPI-style communication
//
// This module implements distributed processing capabilities including
// message passing, collective operations, and parallel computations.
//
// Distributed Computing Features:
//   - MPI-style process initialization and management
//   - Point-to-point communication (blocking/non-blocking)
//   - Collective operations (broadcast, reduce, gather, etc.)
//   - Distributed parallel algorithms
//   - Fault tolerance and error handling
//   - Load balancing and work distribution

package gpu

import (
    "errors"
    "fmt"
    "sync"
    "time"
)

// MPICommunicator represents a distributed computing communicator
type MPICommunicator struct {
    rank     int                   // Process rank (0 to size-1)
    size     int                   // Total number of processes
    name     string                // Processor name
    channels map[int]chan *Message // Inbound channels keyed by source rank
    mutex    sync.RWMutex          // Protects channels access
    active   bool                  // Whether communicator is active
}

// Message represents a message in the distributed system
type Message struct {
	Data   interface{} // Message payload
	Source int         // Sender rank
	Tag    int         // Message tag
	Size   int         // Message size
}

// MessageStatus represents the status of a received message
type MessageStatus struct {
	Source int // Source process rank
	Tag    int // Message tag
	Size   int // Message size
}

// Request represents an asynchronous operation handle
type Request struct {
	message *Message
	done    chan error
	comm    *MPICommunicator
}

// ReduceOperation defines the type of reduction operation
type ReduceOperation int

const (
	ReduceSum ReduceOperation = iota
	ReduceMax
	ReduceMin
	ReduceProd
)

// WorkDistributor handles load balancing and work distribution
type WorkDistributor struct {
	comm *MPICommunicator
}

// TimeoutError represents a communication timeout
type TimeoutError struct {
	message string
}

func (e *TimeoutError) Error() string {
	return e.message
}

// Global communicator registry
var (
    // worlds holds isolated communicator groups keyed by world size.
    worlds   = make(map[int]*world)
    commMutex     sync.RWMutex
)

// world represents a simulated MPI world of a given size.
type world struct {
    size     int
    comms    map[int]*MPICommunicator // rank -> comm
    nextRank int                      // next rank to hand out
}

// InitializeMPI initializes the MPI-style distributed environment
func InitializeMPI(size int) (*MPICommunicator, error) {
    if size <= 0 {
        return nil, errors.New("size must be positive")
    }

    commMutex.Lock()
    defer commMutex.Unlock()

    // Get or build world for this size
    w, ok := worlds[size]
    if !ok {
        // Build a fresh world with shared channels
        w = &world{size: size, comms: make(map[int]*MPICommunicator)}
        // Create comms for each rank
        for r := 0; r < size; r++ {
            w.comms[r] = &MPICommunicator{
                rank:     r,
                size:     size,
                name:     fmt.Sprintf("processor-%d", r),
                channels: make(map[int]chan *Message), // inbound, keyed by source
                active:   false, // becomes active when handed out
            }
        }
        // Create shared channels for each directed pair (src -> dst)
        for src := 0; src < size; src++ {
            for dst := 0; dst < size; dst++ {
                if src == dst {
                    continue
                }
                ch := make(chan *Message, 100)
                // Attach to destination's inbound map under source key
                w.comms[dst].channels[src] = ch
            }
        }
        worlds[size] = w
    }

    // Hand out next rank for this world
    rank := w.nextRank
    w.nextRank = (w.nextRank + 1) % size
    comm := w.comms[rank]
    comm.active = true
    return comm, nil
}

// Rank returns the rank of the current process
func (c *MPICommunicator) Rank() int {
	return c.rank
}

// Size returns the total number of processes
func (c *MPICommunicator) Size() int {
	return c.size
}

// GetProcessorName returns the name of the processor
func (c *MPICommunicator) GetProcessorName() (string, error) {
	if !c.active {
		return "", errors.New("communicator not active")
	}
	return c.name, nil
}

// Send performs blocking send operation
func (c *MPICommunicator) Send(data interface{}, dest int, tag int) error {
    if !c.active {
        return errors.New("communicator not active")
    }

    if dest < 0 || dest >= c.size || dest == c.rank {
        return fmt.Errorf("invalid destination rank: %d", dest)
    }

    message := &Message{
        Data:   data,
        Source: c.rank,
        Tag:    tag,
        Size:   estimateMessageSize(data),
    }

    // Route to destination's inbound channel keyed by our rank
    commMutex.RLock()
    w := worlds[c.size]
    commMutex.RUnlock()
    if w == nil {
        return errors.New("world not initialized")
    }
    dstComm := w.comms[dest]
    if dstComm == nil {
        return fmt.Errorf("destination communicator %d not found", dest)
    }
    dstComm.mutex.RLock()
    ch, exists := dstComm.channels[c.rank]
    dstComm.mutex.RUnlock()
    if !exists {
        return fmt.Errorf("no channel to destination %d from source %d", dest, c.rank)
    }

    select {
    case ch <- message:
        return nil
    default:
        return errors.New("send buffer full")
    }
}

// Recv performs blocking receive operation
func (c *MPICommunicator) Recv(data interface{}, source int, tag int) error {
    if !c.active {
        return errors.New("communicator not active")
    }

    // Receive from our inbound channel from the specified source
    c.mutex.RLock()
    ch, exists := c.channels[source]
    c.mutex.RUnlock()

    if !exists {
        return fmt.Errorf("no channel from source %d", source)
    }

	message := <-ch
	if message.Tag != tag {
		return fmt.Errorf("tag mismatch: expected %d, got %d", tag, message.Tag)
	}

	return copyMessageData(message.Data, data)
}

// RecvWithTimeout performs receive with timeout
func (c *MPICommunicator) RecvWithTimeout(data interface{}, source int, tag int, timeout time.Duration) error {
	if !c.active {
		return errors.New("communicator not active")
	}

	c.mutex.RLock()
	ch, exists := c.channels[source]
	c.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("no channel from source %d", source)
	}

	select {
	case message := <-ch:
		if message.Tag != tag {
			return fmt.Errorf("tag mismatch: expected %d, got %d", tag, message.Tag)
		}
		return copyMessageData(message.Data, data)
	case <-time.After(timeout):
		return &TimeoutError{message: fmt.Sprintf("receive timeout after %v", timeout)}
	}
}

// ISend performs non-blocking send operation
func (c *MPICommunicator) ISend(data interface{}, dest int, tag int) (*Request, error) {
	if !c.active {
		return nil, errors.New("communicator not active")
	}

	request := &Request{
		message: &Message{
			Data:   data,
			Source: c.rank,
			Tag:    tag,
			Size:   estimateMessageSize(data),
		},
		done: make(chan error, 1),
		comm: c,
	}

	// Start asynchronous send
	go func() {
		err := c.Send(data, dest, tag)
		request.done <- err
	}()

	return request, nil
}

// IRecv performs non-blocking receive operation
func (c *MPICommunicator) IRecv(data interface{}, source int, tag int) (*Request, error) {
	if !c.active {
		return nil, errors.New("communicator not active")
	}

	request := &Request{
		done: make(chan error, 1),
		comm: c,
	}

	// Start asynchronous receive
	go func() {
		err := c.Recv(data, source, tag)
		request.done <- err
	}()

	return request, nil
}

// Wait waits for an asynchronous operation to complete
func (r *Request) Wait() error {
	return <-r.done
}

// Probe checks for incoming messages without receiving them
func (c *MPICommunicator) Probe(source int, tag int) (*MessageStatus, error) {
	if !c.active {
		return nil, errors.New("communicator not active")
	}

	c.mutex.RLock()
	ch, exists := c.channels[source]
	c.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no channel from source %d", source)
	}

	// Non-blocking check (simplified)
	select {
	case message := <-ch:
		// Put message back for actual receive
		ch <- message
		return &MessageStatus{
			Source: message.Source,
			Tag:    message.Tag,
			Size:   message.Size,
		}, nil
	default:
		return nil, errors.New("no message available")
	}
}

// Broadcast sends data from root to all processes
func (c *MPICommunicator) Broadcast(data interface{}, root int) error {
	if !c.active {
		return errors.New("communicator not active")
	}

    if c.rank == root {
        // Root sends to all other processes
        // If caller passed a pointer to slice (e.g., *[]float64), dereference for sending
        payload := data
        switch v := data.(type) {
        case *[]float64:
            payload = *v
        case *[][]float64:
            payload = *v
        }
        for i := 0; i < c.size; i++ {
            if i != root {
                err := c.Send(payload, i, 0) // Use tag 0 for broadcast
                if err != nil {
                    return fmt.Errorf("broadcast send to %d failed: %v", i, err)
                }
            }
        }
	} else {
		// Non-root processes receive from root
		err := c.Recv(data, root, 0)
		if err != nil {
			return fmt.Errorf("broadcast receive from %d failed: %v", root, err)
		}
	}

	return nil
}

// BroadcastWithTimeout broadcasts with timeout for fault tolerance
func (c *MPICommunicator) BroadcastWithTimeout(data interface{}, root int, timeout time.Duration) error {
	if !c.active {
		return errors.New("communicator not active")
	}

    if c.rank == root {
        // Root sends to all other processes with timeout handling
        errCh := make(chan error, c.size-1)

        // If caller passed a pointer to slice, dereference for sending
        payload := data
        switch v := data.(type) {
        case *[]float64:
            payload = *v
        case *[][]float64:
            payload = *v
        }
        for i := 0; i < c.size; i++ {
            if i != root {
                go func(dest int) {
                    err := c.Send(payload, dest, 0)
                    errCh <- err
                }(i)
            }
        }

		// Wait for all sends to complete or timeout
		for i := 0; i < c.size-1; i++ {
			select {
			case err := <-errCh:
				if err != nil {
					return err
				}
			case <-time.After(timeout):
				return &TimeoutError{message: "broadcast timeout"}
			}
		}
	} else {
		// Non-root processes receive from root with timeout
		return c.RecvWithTimeout(data, root, 0, timeout)
	}

	return nil
}

// Reduce performs reduction operation
func (c *MPICommunicator) Reduce(sendData interface{}, recvData interface{}, op ReduceOperation, root int) error {
	if !c.active {
		return errors.New("communicator not active")
	}

	if c.rank != root {
		// Non-root sends data to root
		return c.Send(sendData, root, 1) // Use tag 1 for reduce
	} else {
		// Root collects and reduces data
		values := [][]float64{}

		// Add root's own data
		if sendSlice, ok := sendData.([]float64); ok {
			values = append(values, sendSlice)
		} else {
			return errors.New("unsupported data type for reduce")
		}

		// Receive from all other processes
		for i := 0; i < c.size; i++ {
			if i != root {
				var received []float64
				err := c.Recv(&received, i, 1)
				if err != nil {
					return fmt.Errorf("reduce receive from %d failed: %v", i, err)
				}
				values = append(values, received)
			}
		}

		// Perform reduction
		result, err := performReduction(values, op)
		if err != nil {
			return err
		}

		return copyMessageData(result, recvData)
	}
}

// AllReduce performs reduction with result available to all processes
func (c *MPICommunicator) AllReduce(sendData interface{}, recvData interface{}, op ReduceOperation) error {
	if !c.active {
		return errors.New("communicator not active")
	}

	// First, gather all data to process 0
	if c.rank == 0 {
		values := [][]float64{}

		// Add process 0's data
		if sendSlice, ok := sendData.([]float64); ok {
			values = append(values, sendSlice)
		} else {
			return errors.New("unsupported data type for all-reduce")
		}

		// Receive from all other processes
		for i := 1; i < c.size; i++ {
			var received []float64
			err := c.Recv(&received, i, 2) // Use tag 2 for all-reduce
			if err != nil {
				return fmt.Errorf("all-reduce receive from %d failed: %v", i, err)
			}
			values = append(values, received)
		}

		// Perform reduction
		result, err := performReduction(values, op)
		if err != nil {
			return err
		}

		// Broadcast result to all processes
		err = c.Broadcast(result, 0)
		if err != nil {
			return err
		}

		return copyMessageData(result, recvData)
	} else {
		// Non-root processes send data to root
		err := c.Send(sendData, 0, 2)
		if err != nil {
			return err
		}

		// Then receive the broadcast result
		return c.Recv(recvData, 0, 0)
	}
}

// Gather collects data from all processes to root
func (c *MPICommunicator) Gather(sendData interface{}, recvData interface{}, root int) error {
	if !c.active {
		return errors.New("communicator not active")
	}

	if c.rank != root {
		// Non-root sends data to root
		return c.Send(sendData, root, 3) // Use tag 3 for gather
	} else {
		// Root collects all data
		gathered := [][]float64{}

		// Add root's own data first
		if sendSlice, ok := sendData.([]float64); ok {
			gathered = append(gathered, sendSlice)
		} else {
			return errors.New("unsupported data type for gather")
		}

		// Receive from all other processes
		for i := 0; i < c.size; i++ {
			if i != root {
				var received []float64
				err := c.Recv(&received, i, 3)
				if err != nil {
					return fmt.Errorf("gather receive from %d failed: %v", i, err)
				}
				gathered = append(gathered, received)
			}
		}

		return copyMessageData(gathered, recvData)
	}
}

// Finalize cleans up the communicator
func (c *MPICommunicator) Finalize() error {
    c.mutex.Lock()
    defer c.mutex.Unlock()

    c.active = false

    // Close all channels
    for _, ch := range c.channels {
        close(ch)
    }

    // If all communicators in this world are inactive, remove the world
    commMutex.Lock()
    w := worlds[c.size]
    if w != nil {
        allInactive := true
        for _, cm := range w.comms {
            if cm != nil && cm.active {
                allInactive = false
                break
            }
        }
        if allInactive {
            delete(worlds, c.size)
        }
    }
    commMutex.Unlock()

    return nil
}

// DistributedMatMul performs distributed matrix multiplication
func DistributedMatMul(comm *MPICommunicator, A, B []float64, M, N, K int) ([]float64, error) {
	rank := comm.Rank()
	size := comm.Size()

	// Distribute work by rows
	rowsPerProcess := M / size
	startRow := rank * rowsPerProcess
	endRow := startRow + rowsPerProcess
	if rank == size-1 {
		endRow = M // Last process takes remaining rows
	}

	var localA, localB []float64

	if rank == 0 {
		// Root distributes data
		localA = A[startRow*K : endRow*K]
		localB = B // All processes need full B matrix

		// Send A chunks to other processes
		for p := 1; p < size; p++ {
			pStartRow := p * rowsPerProcess
			pEndRow := pStartRow + rowsPerProcess
			if p == size-1 {
				pEndRow = M
			}

			chunk := A[pStartRow*K : pEndRow*K]
			err := comm.Send(chunk, p, 10)
			if err != nil {
				return nil, fmt.Errorf("failed to send A chunk to process %d: %v", p, err)
			}
		}

		// Broadcast B matrix
		err := comm.Broadcast(B, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast B matrix: %v", err)
		}
	} else {
		// Receive A chunk
		err := comm.Recv(&localA, 0, 10)
		if err != nil {
			return nil, fmt.Errorf("failed to receive A chunk: %v", err)
		}

		// Receive B matrix via broadcast
		err = comm.Recv(&localB, 0, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to receive B matrix: %v", err)
		}
	}

	// Compute local result
	localRows := endRow - startRow
	localResult := make([]float64, localRows*N)

	for i := 0; i < localRows; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += localA[i*K+k] * localB[k*N+j]
			}
			localResult[i*N+j] = sum
		}
	}

	// Gather results at root
	if rank == 0 {
		fullResult := make([]float64, M*N)

		// Copy local result
		copy(fullResult[startRow*N:endRow*N], localResult)

		// Receive results from other processes
		for p := 1; p < size; p++ {
			pStartRow := p * rowsPerProcess
			pEndRow := pStartRow + rowsPerProcess
			if p == size-1 {
				pEndRow = M
			}

			var received []float64
			err := comm.Recv(&received, p, 11)
			if err != nil {
				return nil, fmt.Errorf("failed to receive result from process %d: %v", p, err)
			}

			copy(fullResult[pStartRow*N:pEndRow*N], received)
		}

		return fullResult, nil
	} else {
		// Send local result to root
		err := comm.Send(localResult, 0, 11)
		if err != nil {
			return nil, fmt.Errorf("failed to send result to root: %v", err)
		}
		return nil, nil // Non-root processes return nil
	}
}

// NewWorkDistributor creates a new work distributor
func NewWorkDistributor(comm *MPICommunicator) *WorkDistributor {
	return &WorkDistributor{comm: comm}
}

// DistributeWork distributes work items among processes
func (wd *WorkDistributor) DistributeWork(workItems []int) ([]int, error) {
	rank := wd.comm.Rank()
	size := wd.comm.Size()

	itemsPerProcess := len(workItems) / size
	remainder := len(workItems) % size

	startIdx := rank * itemsPerProcess
	endIdx := startIdx + itemsPerProcess

	// Distribute remainder among first processes
	if rank < remainder {
		startIdx += rank
		endIdx += rank + 1
	} else {
		startIdx += remainder
		endIdx += remainder
	}

	if endIdx > len(workItems) {
		endIdx = len(workItems)
	}

	return workItems[startIdx:endIdx], nil
}

// IsTimeoutError checks if an error is a timeout error
func IsTimeoutError(err error) bool {
	_, ok := err.(*TimeoutError)
	return ok
}

// Utility functions

// estimateMessageSize estimates the size of a message payload
func estimateMessageSize(data interface{}) int {
	switch v := data.(type) {
	case []float64:
		return len(v) * 8 // 8 bytes per float64
	case []int:
		return len(v) * 8 // 8 bytes per int on 64-bit systems
	case string:
		return len(v)
	default:
		return 64 // Default estimate
	}
}

// copyMessageData copies message data to the destination
func copyMessageData(src, dest interface{}) error {
	switch srcSlice := src.(type) {
	case []float64:
		if destPtr, ok := dest.(*[]float64); ok {
			*destPtr = make([]float64, len(srcSlice))
			copy(*destPtr, srcSlice)
			return nil
		}
	case [][]float64:
		if destPtr, ok := dest.(*[][]float64); ok {
			*destPtr = make([][]float64, len(srcSlice))
			for i, slice := range srcSlice {
				(*destPtr)[i] = make([]float64, len(slice))
				copy((*destPtr)[i], slice)
			}
			return nil
		}
	}
	return errors.New("unsupported data type for copy")
}

// performReduction performs the specified reduction operation
func performReduction(values [][]float64, op ReduceOperation) ([]float64, error) {
	if len(values) == 0 {
		return nil, errors.New("no values to reduce")
	}

	// Assume all slices have the same length
	length := len(values[0])
	result := make([]float64, length)

	switch op {
	case ReduceSum:
		for i := 0; i < length; i++ {
			sum := 0.0
			for _, slice := range values {
				if i < len(slice) {
					sum += slice[i]
				}
			}
			result[i] = sum
		}
	case ReduceMax:
		for i := 0; i < length; i++ {
			max := values[0][i]
			for _, slice := range values {
				if i < len(slice) && slice[i] > max {
					max = slice[i]
				}
			}
			result[i] = max
		}
	case ReduceMin:
		for i := 0; i < length; i++ {
			min := values[0][i]
			for _, slice := range values {
				if i < len(slice) && slice[i] < min {
					min = slice[i]
				}
			}
			result[i] = min
		}
	case ReduceProd:
		for i := 0; i < length; i++ {
			prod := 1.0
			for _, slice := range values {
				if i < len(slice) {
					prod *= slice[i]
				}
			}
			result[i] = prod
		}
	default:
		return nil, errors.New("unsupported reduction operation")
	}

	return result, nil
}
