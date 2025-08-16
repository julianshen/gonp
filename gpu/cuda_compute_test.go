//go:build cuda
// +build cuda

package gpu

import (
    "math"
    "math/rand"
    "testing"
    "time"
)

func nearlyEqual(a, b, eps float64) bool {
    if a == b {
        return true
    }
    diff := math.Abs(a - b)
    if a == 0 || b == 0 || diff < math.SmallestNonzeroFloat64 {
        return diff < eps
    }
    return diff/(math.Abs(a)+math.Abs(b)) < eps
}

func TestCUDAExecuteVectorAddGPU(t *testing.T) {
    devices, err := DetectCUDADevices()
    if err != nil || len(devices) == 0 {
        t.Skipf("No CUDA device available: %v", err)
    }
    dev := devices[0]

    n := 1 << 14
    a := make([]float64, n)
    b := make([]float64, n)
    rand.Seed(42)
    for i := 0; i < n; i++ {
        a[i] = rand.NormFloat64()
        b[i] = rand.NormFloat64()
    }
    got, err := dev.ExecuteVectorAdd(a, b)
    if err != nil {
        t.Fatalf("ExecuteVectorAdd failed: %v", err)
    }
    if len(got) != n {
        t.Fatalf("unexpected length: got %d want %d", len(got), n)
    }
    for i := 0; i < n; i++ {
        want := a[i] + b[i]
        if !nearlyEqual(got[i], want, 1e-12) {
            t.Fatalf("mismatch at %d: got %g want %g", i, got[i], want)
        }
    }
}

func TestCUDAExecuteSumGPU(t *testing.T) {
    devices, err := DetectCUDADevices()
    if err != nil || len(devices) == 0 {
        t.Skipf("No CUDA device available: %v", err)
    }
    dev := devices[0]

    n := 1 << 16
    data := make([]float64, n)
    rand.Seed(time.Now().UnixNano())
    sum := 0.0
    for i := 0; i < n; i++ {
        v := rand.Float64() - 0.5
        data[i] = v
        sum += v
    }
    got, err := dev.ExecuteSum(data)
    if err != nil {
        t.Fatalf("ExecuteSum failed: %v", err)
    }
    if !nearlyEqual(got, sum, 1e-9*float64(n)) { // loose epsilon scaled by size
        t.Fatalf("sum mismatch: got %g want %g", got, sum)
    }
}

func TestCUDAExecuteMatMulGPU(t *testing.T) {
    devices, err := DetectCUDADevices()
    if err != nil || len(devices) == 0 {
        t.Skipf("No CUDA device available: %v", err)
    }
    dev := devices[0]

    M, N, K := 64, 48, 32
    A := make([]float64, M*K)
    B := make([]float64, K*N)
    rand.Seed(7)
    for i := range A { A[i] = rand.Float64() - 0.5 }
    for i := range B { B[i] = rand.Float64() - 0.5 }

    got, err := dev.ExecuteMatMul(A, B, M, N, K)
    if err != nil {
        t.Fatalf("ExecuteMatMul failed: %v", err)
    }
    if len(got) != M*N {
        t.Fatalf("unexpected length: got %d want %d", len(got), M*N)
    }
    // Verify a few random entries against CPU result
    for trial := 0; trial < 10; trial++ {
        i := rand.Intn(M)
        j := rand.Intn(N)
        want := 0.0
        for k := 0; k < K; k++ {
            want += A[i*K+k] * B[k*N+j]
        }
        gotVal := got[i*N+j]
        if !nearlyEqual(gotVal, want, 1e-9*float64(K)) {
            t.Fatalf("C[%d,%d] mismatch: got %g want %g", i, j, gotVal, want)
        }
    }
}

