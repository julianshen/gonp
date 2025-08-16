//go:build arm64 && !vet

package internal

import (
    "bufio"
    "os"
    "runtime"
    "strings"
)

// detectARMFeatures detects ARM64 SIMD capabilities
func detectARMFeatures(features *CPUFeatures) {
    // ARM64 has NEON (Advanced SIMD) by default per AArch64 spec
    features.HasNEON = true
    features.HasNEONFP = true
    features.HasASIMD = true
    features.HasFP = true

    // Detect optional extensions conservatively
    if runtime.GOOS == "linux" {
        // Parse /proc/cpuinfo Features
        flags := readLinuxCPUFlags()
        if flags["crc32"] {
            features.HasCRC32 = true
        }
        // Common crypto flags on ARM: aes, pmull, sha1, sha2
        if flags["aes"] || flags["sha1"] || flags["sha2"] || flags["pmull"] {
            features.HasCrypto = true
        }
    }

    DebugInfo("ARM64 Features detected: NEON=%v, NEONFP=%v, ASIMD=%v, FP=%v, CRC32=%v, Crypto=%v",
        features.HasNEON, features.HasNEONFP, features.HasASIMD, features.HasFP, features.HasCRC32, features.HasCrypto)
}

func readLinuxCPUFlags() map[string]bool {
    flags := map[string]bool{}
    f, err := os.Open("/proc/cpuinfo")
    if err != nil {
        return flags
    }
    defer f.Close()
    s := bufio.NewScanner(f)
    for s.Scan() {
        line := strings.ToLower(s.Text())
        if strings.HasPrefix(line, "features") || strings.HasPrefix(line, "flags") {
            // features : fp asimd evtstrm aes pmull sha1 sha2 crc32
            parts := strings.Split(line, ":")
            if len(parts) < 2 {
                continue
            }
            for _, tok := range strings.Fields(parts[1]) {
                flags[strings.TrimSpace(tok)] = true
            }
        }
    }
    return flags
}
