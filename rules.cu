#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

#define MAX_LEN 32
#define THREADS 512
// nvcc --shared -o rules.dll rules.cu -Xcompiler "/MD" -link -cudart static
// nvcc -shared -o librules.dll rules.cu

// XXHash64 constants
#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ int device_strncmp(const char* str1, const char* str2, int n) {
    for (int i = 0; i < n; i++) {
        if (str1[i] != str2[i]) {
            return str1[i] - str2[i];
        }
        if (str1[i] == '\0') {
            return 0; // Both strings ended
        }
    }
    return 0; // First n characters are equal
}


// AI optimized to reduce divergence it gave itself a 9.5/10 - I wonder where the last 0.5 went :)
__device__ bool binarySearchGPU(const uint64_t* array, int length, uint64_t target) {
    if (target == 0) return false;

    int left = 0;
    int right = length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        uint64_t midVal = array[mid];

        // Completely branchless
        int cmp = (midVal < target) ? -1 : (midVal > target) ? 1 : 0;
        left = (cmp == -1) ? mid + 1 : left;
        right = (cmp == 1) ? mid - 1 : right;
        if (cmp == 0) return true;
    }
    return false;
}

// Binary search on strings, also by AI but haven't bothered asking it to optimize for divergence.
__device__ bool binarySearchGPUFast(char* targets, uint8_t* targetLengths, int targetCount, const char* seek, uint8_t seekLength) {
    if (seek == nullptr || targets == nullptr || targetCount == 0 || seekLength == 0) {
        return false;
    }
    int left = 0;
    int right = targetCount - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        char* midTarget = &targets[mid * MAX_LEN];

        // Proper lexicographical comparison
        int cmp = 0;
        uint8_t compareLen = min(targetLengths[mid], seekLength);

        // Compare content first
        for (int i = 0; i < compareLen; i++) {
            if (midTarget[i] != seek[i]) {
                cmp = midTarget[i] - seek[i];
                break;
            }
        }

        // If content is equal up to min length, compare lengths
        if (cmp == 0) {
            cmp = targetLengths[mid] - seekLength;
        }

        if (cmp == 0) {
            return true;  // Exact match
        } else if (cmp < 0) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return false;
}

// 64-bit rotation function
__device__ uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

// XXHash round operation
__device__ uint64_t round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

// Merge round for combining accumulators
__device__ uint64_t mergeRound(uint64_t acc, uint64_t val) {
    val = round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

// Read 32-bit value from unaligned memory
__device__ uint32_t read32(const void* ptr) {
    const uint8_t* byte_ptr = (const uint8_t*)ptr;
    return (uint32_t)byte_ptr[0] | ((uint32_t)byte_ptr[1] << 8) |
           ((uint32_t)byte_ptr[2] << 16) | ((uint32_t)byte_ptr[3] << 24);
}

// Read 64-bit value from unaligned memory (little-endian)
__device__ uint64_t read64(const void* ptr) {
    const uint8_t* byte_ptr = (const uint8_t*)ptr;
    return (uint64_t)byte_ptr[0] | ((uint64_t)byte_ptr[1] << 8) |
           ((uint64_t)byte_ptr[2] << 16) | ((uint64_t)byte_ptr[3] << 24) |
           ((uint64_t)byte_ptr[4] << 32) | ((uint64_t)byte_ptr[5] << 40) |
           ((uint64_t)byte_ptr[6] << 48) | ((uint64_t)byte_ptr[7] << 56);
}

// XXHash64 computation for a single string
__device__ uint64_t xxhash64(const char* input, int len, uint64_t seed) {
    const char* p = input;
    const char* const end = p + len;
    uint64_t h64;

    if (len >= 32) {
        const char* const limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - PRIME64_1;

        // Process 32-byte chunks
        do {
            v1 = round(v1, read64(p)); p += 8;
            v2 = round(v2, read64(p)); p += 8;
            v3 = round(v3, read64(p)); p += 8;
            v4 = round(v4, read64(p)); p += 8;
        } while (p <= limit);

        // Combine accumulators
        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = mergeRound(h64, v1);
        h64 = mergeRound(h64, v2);
        h64 = mergeRound(h64, v3);
        h64 = mergeRound(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    // Process remaining bytes
    while (p + 8 <= end) {
        uint64_t k1 = read64(p);
        k1 *= PRIME64_2;
        k1 = rotl64(k1, 31);
        k1 *= PRIME64_1;
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    // Process last 4 bytes
    if (p + 4 <= end) {
        h64 ^= (uint64_t)read32(p) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    // Process remaining bytes
    while (p < end) {
        h64 ^= (*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    // Final avalanche
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

// CUDA kernel to compute hashes for all strings
extern "C" {
// 'l' Rule: converts all (ASCII) characters to lowercase.
__global__ void lowerCaseKernel(char *words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char *word = &words[idx * MAX_LEN];
        for (int i = 0; i < len; i++) {
            // Simple ASCII conversion: if character is A-Z, add 32.
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
    }
}

// 'u' Rule: converts all (ASCII) characters to uppercase.
__global__ void upperCaseKernel(char *words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char *word = &words[idx * MAX_LEN];
        for (int i = 0; i < len; i++) {
            // Simple ASCII conversion: if character is a-z, subtract 32.
            if (word[i] >= 'a' && word[i] <= 'z') {
                word[i] = word[i] - 32;
            }
        }
    }
}

// 'c' Rule: Capitalize first letter, lowercase rest
__global__ void capitalizeKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0) return;

        // Lowercase all characters
        for (int i = 0; i < len; ++i) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] += 32;
            }
        }
        // Uppercase first character
        if (word[0] >= 'a' && word[0] <= 'z') {
            word[0] -= 32;
        }
    }
}

// 'C' Rule: Lowercase first letter, uppercase rest
__global__ void invertCapitalizeKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0) return;

        // Uppercase all characters
        for (int i = 0; i < len; ++i) {
            if (word[i] >= 'a' && word[i] <= 'z') {
                word[i] -= 32;
            }
        }
        // Lowercase first character
        if (word[0] >= 'A' && word[0] <= 'Z') {
            word[0] += 32;
        }
    }
}

// 't' Rule: Toggle case of all characters
__global__ void toggleCaseKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        for (int i = 0; i < len; i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] += 32;
            } else if (word[i] >= 'a' && word[i] <= 'z') {
                word[i] -= 32;
            }
        }
    }
}

// 'q' Rule: Duplicate each character
__global__ void duplicateCharsKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (2 * len > MAX_LEN - 1) return;

        // Duplicate characters
        for (int i = len - 1; i >= 0; --i) {
            word[2 * i] = word[i];
            word[2 * i + 1] = word[i];
        }
        lengths[idx] = 2 * len;
        word[2 * len] = '\0';
    }
}

// 'r' Rule: Reverse Word
__global__ void reverseKernel(char *words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char *word = &words[idx * MAX_LEN];
        // Swap characters from beginning and end
        for (int i = 0; i < len / 2; i++) {
            char temp = word[i];
            word[i] = word[len - 1 - i];
            word[len - 1 - i] = temp;
        }
    }
}

// 'k' Rule: Swap first two characters
__global__ void swapFirstTwoKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len >= 2) {
            char temp = word[0];
            word[0] = word[1];
            word[1] = temp;
        }
    }
}

// 'K' Rule: Swap last two characters
__global__ void swapLastTwoKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len >= 2) {
            char temp = word[len-1];
            word[len-1] = word[len-2];
            word[len-2] = temp;
        }
    }
}

// 'd' Rule: Duplicate word
__global__ void duplicateWordKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len * 2 < MAX_LEN) {
            for (int i = 0; i < len; i++) {
                word[len + i] = word[i];
            }
            lengths[idx] = len * 2;
            word[len * 2] = '\0';
        }
    }
}

// 'f' Rule: Append reversed string
__global__ void reflectKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (2 * len > MAX_LEN - 1) return;

        // Append reversed characters
        for (int i = 0; i < len; ++i) {
            word[len + i] = word[len - 1 - i];
        }
        lengths[idx] = 2 * len;
        word[2 * len] = '\0';
    }
}

// '{' Rule: Rotate left (move first char to end)
__global__ void rotateLeftKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len <= 1) return;

        char first = word[0];
        for (int i = 0; i < len - 1; ++i) {
            word[i] = word[i + 1];
        }
        word[len - 1] = first;
    }
}

// '}' Rule: Rotate right (move last char to start)
__global__ void rotateRightKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len <= 1) return;

        char last = word[len - 1];
        for (int i = len - 1; i > 0; --i) {
            word[i] = word[i - 1];
        }
        word[0] = last;
    }
}

// '[' Rule: Delete first character
__global__ void deleteFirstKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len <= 0) return;
        for (int i = 0; i < len - 1; i++) {
            word[i] = word[i + 1];
        }
        lengths[idx] = len - 1;
        word[len - 1] = '\0';
    }
}

// ']' Rule: Delete last character
__global__ void deleteLastKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len <= 0) return;

        lengths[idx] = len - 1;
        word[len - 1] = '\0';
    }
}

// 'E' Rule: Title case (capitalize after spaces)
__global__ void titleCaseKernel(char* words, uint8_t *lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];

        // Lowercase all
        for (int i = 0; i < len; ++i) {
            if (word[i] >= 'A' && word[i] <= 'Z') word[i] += 32;
        }

        // Capitalize first and after spaces
        if (len > 0 && word[0] >= 'a' && word[0] <= 'z') {
            word[0] -= 32;
        }
        for (int i = 1; i < len; ++i) {
            if (word[i - 1] == ' ' && word[i] >= 'a' && word[i] <= 'z') {
                word[i] -= 32;
            }
        }
    }
}

// 'T' Rule: Toggle case at position
__global__ void togglePositionKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos < 0 || pos >= len) return;

        if (word[pos] >= 'a' && word[pos] <= 'z') {
            word[pos] -= 32;
        } else if (word[pos] >= 'A' && word[pos] <= 'Z') {
            word[pos] += 32;
        }
    }
}

// 'p' Rule: Repeat word N times
__global__ void repeatWordKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        int newLen = len * (count + 1);
        if (newLen >= MAX_LEN) newLen = MAX_LEN - 1;

        for (int i = 1; i <= count && (i * len) < newLen; i++) {
            for (int j = 0; j < len && (i * len + j) < newLen; j++) {
                word[i * len + j] = word[j];
            }
        }
        lengths[idx] = newLen;
        word[newLen] = '\0';
    }
}

// 'D' Rule: Delete character at position
__global__ void deletePositionKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos < 0 || pos >= len) return;

        for (int i = pos; i < len - 1; ++i) {
            word[i] = word[i + 1];
        }
        lengths[idx] = len - 1;
        word[len - 1] = '\0';
    }
}

// 'z' Rule: Prepend first character multiple times
__global__ void prependFirstCharKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0 || count <= 0) return;

        int newLen = len + count;
        if (newLen > MAX_LEN - 1) newLen = MAX_LEN - 1;
        char first = word[0];

        // Shift existing characters
        for (int i = newLen - 1; i >= count; --i) {
            word[i] = word[i - count];
        }
        // Prepend first character
        for (int i = 0; i < count; ++i) {
            word[i] = first;
        }
        lengths[idx] = newLen;
        word[newLen] = '\0';
    }
}

// 'Z' Rule: Append last character N times
__global__ void appendLastCharKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0) return;

        char last = word[len - 1];
        int newLen = len + count;
        if (newLen >= MAX_LEN) newLen = MAX_LEN - 1;

        for (int i = len; i < newLen; i++) {
            word[i] = last;
        }
        lengths[idx] = newLen;
        word[newLen] = '\0';
    }
}

// ''' Rule: Truncate at position
__global__ void truncateAtKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        if (pos < len) {
            lengths[idx] = pos;
            words[idx * MAX_LEN + pos] = '\0';
        }
    }
}

// 's' Rule: replaces all occurrences of oldChar with newChar.
__global__ void substitutionKernel(char *words, uint8_t *lengths, char oldChar, char newChar, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char *word = &words[idx * MAX_LEN];
        for (int i = 0; i < len; i++) {
            if (word[i] == oldChar) {
                word[i] = newChar;
            }
        }
    }
}

// 'S' Rule: replaces first occurrence of oldChar with newChar.
__global__ void substitutionFirstKernel(char *words, uint8_t *lengths, char oldChar, char newChar, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char *word = &words[idx * MAX_LEN];
        for (int i = 0; i < len; i++) {
            if (word[i] == oldChar) {
                word[i] = newChar;
                break;
            }
        }
    }
}

// '$' Rule: Appends a given character to the end of each word.
__global__ void appendKernel(char *words, uint8_t *lengths, char appendChar, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        if (len < MAX_LEN - 1) {
            words[idx * MAX_LEN + len] = appendChar;
            words[idx * MAX_LEN + len + 1] = '\0';
            lengths[idx] = len + 1;
        }
    }
}

// '^' Rule: Prepends a given character to the start of each word.
__global__ void prependKernel(char *words, uint8_t *lengths, char prefixChar, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        if (len < MAX_LEN - 1) {
            char *word = &words[idx * MAX_LEN];
            // Shift the word one character to the right (including the terminator)
            for (int i = len; i >= 0; i--) {
                word[i + 1] = word[i];
            }
            word[0] = prefixChar;
            lengths[idx] = len + 1;
        }
    }
}

// 'y' Rule: Prepend substring
__global__ void prependSubstrKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        int prependLen = min(count, len);
        int newLen = len + prependLen;

        if (newLen >= MAX_LEN) {
            newLen = MAX_LEN - 1;
            prependLen = newLen - len;
        }

        // Shift right
        for (int i = newLen - 1; i >= prependLen; i--) {
            word[i] = word[i - prependLen];
        }
        // Copy prefix
        for (int i = 0; i < prependLen; i++) {
            word[i] = word[i + prependLen];
        }
        lengths[idx] = newLen;
        word[newLen] = '\0';
    }
}

// 'Y' Rule: Append substring
__global__ void appendSubstrKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        int appendLen = min(count, len);
        int newLen = len + appendLen;

        if (newLen >= MAX_LEN) {
            newLen = MAX_LEN - 1;
            appendLen = newLen - len;
        }

        for (int i = 0; i < appendLen; i++) {
            word[len + i] = word[len - appendLen + i];
        }
        lengths[idx] = newLen;
        word[newLen] = '\0';
    }
}

// 'L' Rule: Bitwise shift left at position
__global__ void bitShiftLeftKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos >= 0 && pos < len) {
            word[pos] = word[pos] << 1;
        }
    }
}

// 'R' Rule: Bitwise shift right at position
__global__ void bitShiftRightKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos >= 0 && pos < len) {
            word[pos] = word[pos] >> 1;
        }
    }
}

// '-' Rule: Decrement character at position
__global__ void decrementCharKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos >= 0 && pos < len) {
            word[pos]--;
        }
    }
}

// '+' Rule: Increment character at position
__global__ void incrementCharKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos >= 0 && pos < len) {
            word[pos]++;
        }
    }
}

// '@' Rule: Delete all instances of character
__global__ void deleteAllCharKernel(char* words, uint8_t *lengths, char target, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        int newIndex = 0;
        for (int i = 0; i < len; i++) {
            if (word[i] != target) {
                word[newIndex++] = word[i];
            }
        }
        lengths[idx] = newIndex;
        word[newIndex] = '\0';
    }
}

// '.' Rule: Swap with next character
__global__ void swapNextKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos >= 0 && pos < len - 1) {
            char temp = word[pos];
            word[pos] = word[pos + 1];
            word[pos + 1] = temp;
        }
    }
}

// ',' Rule: Swap with next character
__global__ void swapLastKernel(char* words, uint8_t *lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (pos >= 0 && pos < len - 1) {
            char temp = word[pos];
            word[pos] = word[pos - 1];
            word[pos - 1] = temp;
        }
    }
}

// 'e' Rule: Title case with separator
__global__ void titleSeparatorKernel(char* words, uint8_t *lengths, char separator, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0) return;
        // Convert all characters to lowercase
        for (int i = 0; i < len; i++) {
            // Simple ASCII conversion: if character is A-Z, add 32.
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
        // Capitalize first character
        if (word[0] >= 'a' && word[0] <= 'z') {
            word[0] = word[0] - 32;
        }
        // Process trigger characters
        // -1 to prevent overflow in i+1 check and 1 to not double capitalize and gain small % performance
        // I can imagine this being undesired - feel free to report
        for (int i = 1; i < len - 1; ++i) {
            if (word[i] == separator) {
                word[i+1] = word[i+1] - 32;
            }
        }
    }
}

// 'i' Rule: Insert string at position
__global__ void insertKernel(char* words, uint8_t *lengths, int pos, char insert_char, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0 || pos > len) return;
        if (len + 1 >= MAX_LEN) return;

        // Shift characters right
        for (int i = len; i > pos; --i) {
            word[i] = word[i - 1];
        }
        word[pos] = insert_char;

        lengths[idx] = len + 1;
        word[len + 1] = '\0';
    }
}

// 'O' Rule: Omit Range M starting at pos N (ONM)
__global__ void OmitKernel(char* words, uint8_t *lengths, int pos, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len <= 0 || pos >= len) return;

        int delete_count = min(count, len - pos);
        for (int i = pos; i < len - delete_count; ++i) {
            word[i] = word[i + delete_count];
        }
        lengths[idx] = len - delete_count;
        word[len - delete_count] = '\0';
    }
}

// 'o' Rule: Overwrite at position
__global__ void overwriteKernel(char* words, uint8_t *lengths, int pos, char replace_char, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];

        // Check for empty string or invalid position
        if (len <= 0 || pos < 0 || pos >= len) return;
        // Overwrite single character
        word[pos] = replace_char;
    }
}

// '*' Rule: Swap any two
__global__ void swapAnyKernel(char* words, uint8_t *lengths, int pos, int replace_pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len == 0 || pos < 0 || replace_pos < 0 || pos >= len || replace_pos >= len) return;

        // Swap
        char temp = word[pos];
        word[pos] = word[replace_pos];
        word[replace_pos] = temp;
    }
}


// 'x' Rule: Delete all except a slice i.e. extract a substring
__global__ void extractKernel(char* words, uint8_t *lengths, int pos, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len <= 0 || pos >= len) return;
        int keep_count = min(count, len - pos);
        // First, shift the kept slice to the beginning
        for (int i = 0; i < keep_count; ++i) {
            word[i] = word[pos + i];
        }
        // Truncate after the kept portion
        lengths[idx] = keep_count;
        word[keep_count] = '\0';
    }
}

// '<' Rule: Reject words longer than or equal to count
__global__ void rejectLessKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        if (lengths[idx] >= count) {
            lengths[idx] = 0;
            words[idx * MAX_LEN] = '\0';
        }
    }
}

// '>' Rule: Reject words shorter than or equal to count
__global__ void rejectGreaterKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        if (lengths[idx] <= count) {
            lengths[idx] = 0;
            words[idx * MAX_LEN] = '\0';
        }
    }
}

// '_' Rule: Reject words with exact length
__global__ void rejectEqualKernel(char* words, uint8_t *lengths, int count, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        if (lengths[idx] == count) {
            lengths[idx] = 0;
            words[idx * MAX_LEN] = '\0';
        }
    }
}

// '!' Rule: Reject words containing specific character
__global__ void rejectContainKernel(char* words, uint8_t *lengths, char contain_char, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];

        for (int i = 0; i < len; ++i) {
            if (word[i] == contain_char) {
                lengths[idx] = 0;
                word[0] = '\0';
                return;
            }
        }
    }
}

// '/' Rule: Reject words not containing specific character
__global__ void rejectNotContainKernel(char* words, uint8_t *lengths, char contain_char, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        bool found = false;

        for (int i = 0; i < len; ++i) {
            if (word[i] == contain_char) {
                found = true;
                break;
            }
        }

        if (!found) {
            lengths[idx] = 0;
            word[0] = '\0';
        }
    }
}

// '3' Rule: Toggle case after nth instance of character
__global__ void toggleWithNSeparatorKernel(char* words, uint8_t *lengths, char separator_char, int separator_num, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        int count = 0;
        int toggle_pos = -1;

        // Find nth instance
        for (int i = 0; i < len; ++i) {
            if (word[i] == separator_char) {
                if (count == separator_num) {
                    toggle_pos = i + 1;
                    break;
                }
                count++;
            }
        }

        // Toggle next character if found and within bounds
        if (toggle_pos > 0 && toggle_pos < len) {
            if (word[toggle_pos] >= 'a' && word[toggle_pos] <= 'z') {
                word[toggle_pos] = word[toggle_pos] - 32;
            } else if (word[toggle_pos] >= 'A' && word[toggle_pos] <= 'Z') {
                word[toggle_pos] = word[toggle_pos] + 32;
            }
        }
    }
}


__global__ void countUniqueHashesKernel(uint64_t* hashes, int* flags, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t current = hashes[idx];
        int isUnique = 1;

        for (int i = 0; i < idx; i++) {
            if (hashes[i] == current) {
                isUnique = 0;
                break;
            }
        }
        flags[idx] = isUnique;
    }
}

//---------------------------------------------------------------------
// Host Wrappers to Launch Kernels
//---------------------------------------------------------------------
// l
DLL_EXPORT
void applyLowerCase(char *d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    lowerCaseKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// u
DLL_EXPORT
void applyUpperCase(char *d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    upperCaseKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// c
DLL_EXPORT
void applyCapitalize(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    capitalizeKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// C
DLL_EXPORT
void applyInvertCapitalize(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    invertCapitalizeKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// t
DLL_EXPORT
void applyToggleCase(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    toggleCaseKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// q
DLL_EXPORT
void applyDuplicateChars(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    duplicateCharsKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// r
DLL_EXPORT
void applyReverse(char *d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    reverseKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// k
DLL_EXPORT
void applySwapFirstTwo(char *d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapFirstTwoKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// K
DLL_EXPORT
void applySwapLastTwo(char *d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapLastTwoKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// d
DLL_EXPORT
void applyDuplicate(char *d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    duplicateWordKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// f
DLL_EXPORT
void applyReflect(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    reflectKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// {
DLL_EXPORT
void applyRotateLeft(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rotateLeftKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
 // }
DLL_EXPORT
void applyRotateRight(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rotateRightKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// [
DLL_EXPORT
void applyDeleteFirst(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deleteFirstKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// ]
DLL_EXPORT
void applyDeleteLast(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deleteLastKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// E
DLL_EXPORT
void applyTitleCase(char* d_words, uint8_t *d_lengths, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    titleCaseKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, numWords);
}
// T
DLL_EXPORT
void applyTogglePosition(char* d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    togglePositionKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// p
DLL_EXPORT
void applyRepeatWord(char* d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    repeatWordKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}
// D
DLL_EXPORT
void applyDeletePosition(char* d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deletePositionKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// z
DLL_EXPORT
void applyPrependFirstChar(char* d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependFirstCharKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}
// Z
DLL_EXPORT
void applyAppendLastChar(char* d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependFirstCharKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}
// '
DLL_EXPORT
void applyTruncateAt(char* d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    truncateAtKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// s
DLL_EXPORT
void applySubstitution(char *d_words, uint8_t *d_lengths, char oldChar, char newChar, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    substitutionKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, oldChar, newChar, numWords);
}
// S
DLL_EXPORT
void applySubstitutionFirst(char *d_words, uint8_t *d_lengths, char oldChar, char newChar, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    substitutionFirstKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, oldChar, newChar, numWords);
}
// $
DLL_EXPORT
void applyAppend(char *d_words, uint8_t *d_lengths, char appendChar, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    appendKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, appendChar, numWords);
}
// ^
DLL_EXPORT
void applyPrepend(char *d_words, uint8_t *d_lengths, char prefixChar, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, prefixChar, numWords);
}
// y
DLL_EXPORT
void applyPrependPrefixSubstr(char *d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependSubstrKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}
// Y
DLL_EXPORT
void applyAppendSuffixSubstr(char *d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    appendSubstrKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}
// L
DLL_EXPORT
void applyBitShiftLeft(char *d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    bitShiftLeftKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// R
DLL_EXPORT
void applyBitShiftRight(char *d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    bitShiftRightKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// -
DLL_EXPORT
void applyDecrementChar(char *d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    decrementCharKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// +
DLL_EXPORT
void applyIncrementChar(char *d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    incrementCharKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// @
DLL_EXPORT
void applyDeleteAllChar(char *d_words, uint8_t *d_lengths, char deleteChar, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deleteAllCharKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, deleteChar, numWords);
}
// .
DLL_EXPORT
void applySwapNext(char *d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapNextKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// ,
DLL_EXPORT
void applySwapLast(char *d_words, uint8_t *d_lengths, int pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapLastKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, numWords);
}
// e
DLL_EXPORT
void applyTitleSeparator(char *d_words, uint8_t *d_lengths, char separator, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    titleSeparatorKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, separator, numWords);
}
// i
DLL_EXPORT
void applyInsert(char* d_words, uint8_t *d_lengths, int pos, char insert_char, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    insertKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, insert_char, numWords);
}
// O
DLL_EXPORT
void applyOmit(char* d_words, uint8_t *d_lengths, int pos, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    OmitKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, count, numWords);
}
// o
DLL_EXPORT
void applyOverwrite(char* d_words, uint8_t *d_lengths, int pos, char replace_char, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    overwriteKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, replace_char, numWords);
}
// *
DLL_EXPORT
void applySwapAny(char* d_words, uint8_t *d_lengths, int pos, int replace_pos, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapAnyKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, replace_pos, numWords);
}
// x
DLL_EXPORT
void applyExtract(char* d_words, uint8_t *d_lengths, int pos, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    extractKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, pos, count, numWords);
}
// <
DLL_EXPORT
void applyRejectLess(char* d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rejectLessKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}

// >
DLL_EXPORT
void applyRejectGreater(char* d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rejectGreaterKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}

// _
DLL_EXPORT
void applyRejectEqual(char* d_words, uint8_t *d_lengths, int count, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rejectEqualKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, count, numWords);
}

// !
DLL_EXPORT
void applyRejectContain(char* d_words, uint8_t *d_lengths, char contain_char, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rejectContainKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, contain_char, numWords);
}

// /
DLL_EXPORT
void applyRejectNotContain(char* d_words, uint8_t *d_lengths, char contain_char, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rejectNotContainKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, contain_char, numWords);
}

// 3
DLL_EXPORT
void applyToggleWithNSeparator(char* d_words, uint8_t *d_lengths, char separator_char, int separator_num, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    toggleWithNSeparatorKernel<<<blocks, THREADS, 0, stream>>>(d_words, d_lengths, separator_char, separator_num, numWords);
}

// Allocate dictionary memory & dictionary lengths memory
DLL_EXPORT
void allocateDictionary(
    char **d_dict, uint8_t **d_dictLengths,
    int dictCount, cudaStream_t stream
) {
    CUDA_CHECK(cudaMallocAsync((void**)d_dict, dictCount * MAX_LEN * sizeof(char), stream));
    CUDA_CHECK(cudaMallocAsync((void**)d_dictLengths, dictCount * sizeof(uint8_t), stream));
    cudaStreamSynchronize(stream);
}

// Copy host memory to device
DLL_EXPORT
void pushDictionary(
    char *h_wordlist, uint8_t *h_wordlistLengths,
    char **d_wordlist, uint8_t **d_wordlistLengths,
    int wordlistCount, cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(*d_wordlist, h_wordlist, wordlistCount * MAX_LEN * sizeof(char), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(*d_wordlistLengths, h_wordlistLengths, wordlistCount * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    cudaStreamSynchronize(stream);
}

DLL_EXPORT
void overwriteDictionary(
    char **d_wordlist, uint8_t **d_wordlistLengths,
    char **d_overwrite, uint8_t **d_overwriteLengths,
    int wordlistCount,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(*d_overwrite, *d_wordlist, wordlistCount * MAX_LEN * sizeof(char), cudaMemcpyDeviceToDevice , stream));
    CUDA_CHECK(cudaMemcpyAsync(*d_overwriteLengths, *d_wordlistLengths, wordlistCount * sizeof(uint8_t), cudaMemcpyDeviceToDevice , stream));
}

DLL_EXPORT
void resetDictionary(
    char **d_wordlist, uint8_t **d_wordlistLengths,
    int wordlistCount,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemsetAsync(*d_wordlist, 0, wordlistCount * MAX_LEN * sizeof(char), stream));
    CUDA_CHECK(cudaMemsetAsync(*d_wordlistLengths, 0, wordlistCount * sizeof(uint8_t), stream));
}

// Copy device memory back to host
DLL_EXPORT
void pullDictionary(
    char **d_processed, int **d_processedLengths,
    char *h_processedDict, int *h_processedDictLengths,
    int numWords, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(h_processedDict, *d_processed, numWords * MAX_LEN * sizeof(char), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_processedDictLengths, *d_processedLengths, numWords * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

DLL_EXPORT
void deallocateDictionary(char *d_dict, uint8_t *d_dictLengths, cudaStream_t stream) {
    CUDA_CHECK(cudaFreeAsync(d_dict, stream));
    CUDA_CHECK(cudaFreeAsync(d_dictLengths, stream));
    cudaStreamSynchronize(stream);
}

// Allocate uint64's to store hashes
DLL_EXPORT
void allocateHashes(uint64_t **d_hashes, int hashCount, cudaStream_t stream) {
    CUDA_CHECK(cudaMallocAsync((void**)d_hashes, hashCount * sizeof(uint64_t), stream));
}

// Copy host memory to device
DLL_EXPORT
void pushHashes(uint64_t *h_hashes, uint64_t **d_hashes, int hashCount, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(*d_hashes, h_hashes, hashCount * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

// Push memory of A into B in overwriteHashes(A, B)
DLL_EXPORT
void overwriteHashes(
    uint64_t **d_hashes,
    uint64_t **d_toOverwrite,
    int hashCount,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(*d_toOverwrite, *d_hashes, hashCount * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
}

// Reset the hashes to 0
DLL_EXPORT
void resetHashes(
    uint64_t **d_hashes,
    int hashCount,
    cudaStream_t stream
) {
   CUDA_CHECK(cudaMemsetAsync(*d_hashes, 0, hashCount * sizeof(uint64_t), stream));
}

// Copy device memory back to host
DLL_EXPORT
void pullHashes(
    uint64_t **d_hashes,
    uint64_t *h_hashes,
    int hashCount,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(h_hashes, *d_hashes, hashCount * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

DLL_EXPORT
void deallocateHashes(uint64_t *d_hashes, cudaStream_t stream) {
    CUDA_CHECK(cudaFreeAsync(d_hashes, stream));
}

DLL_EXPORT
void initializeHitCount(uint64_t **d_hitCount, cudaStream_t stream) {
    CUDA_CHECK(cudaMallocAsync((void**)d_hitCount, sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(*d_hitCount, 0, sizeof(uint64_t), stream));
}

void pullHitCount(uint64_t *d_hitCount, uint64_t* h_hitCount, cudaStream_t stream) {
    cudaMemcpyAsync(h_hitCount, d_hitCount, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

DLL_EXPORT
void resetHitCount(uint64_t **d_hitCount, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(*d_hitCount, 0, sizeof(uint64_t), stream));
}
DLL_EXPORT
void deallocateHitCount(uint64_t *d_hitCount, cudaStream_t stream) {
    CUDA_CHECK(cudaFreeAsync(d_hitCount, stream));
}

// Kernel for hashing
__global__ void xxhashKernel(char* d_processed, int* d_processedLengths, uint64_t seed, uint64_t* hashes, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWords) return;

    char* input = &d_processed[idx * MAX_LEN];
    hashes[idx] = xxhash64(input, d_processedLengths[idx], seed);
}

// Kernel for hashing with hit detection
__global__ void xxhashWithHitsKernel(
    char* d_processed,
    uint8_t* d_processedLengths,
    const uint64_t* d_wordlistHashes,
    int wordlistCount,
    const uint64_t* d_targetHashes,
    int targetCount,
    uint64_t* hitCount,
    uint64_t* d_foundHashes,
    uint64_t seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wordlistCount) return;

    char* word = &d_processed[idx * MAX_LEN];
    const uint64_t hash = xxhash64(word, d_processedLengths[idx], seed);

    const bool inCompare = binarySearchGPU(d_targetHashes, targetCount, hash);
    if (inCompare) {
        uint64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(hitCount), 1ULL);
        d_foundHashes[pos] = hash;
    }
}

// Kernel with hit detection
__global__ void HitsKernel(
    char* d_processed,
    uint8_t* d_processedLengths,
    char* d_target,
    uint8_t* d_targetLengths,
    char* d_matching,
    uint8_t* d_matchingLengths,
    int wordlistCount,
    int targetCount,
    uint64_t* hitCount,
    bool storeHits
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wordlistCount) return;

    char* word = &d_processed[idx * MAX_LEN];
    uint8_t wordLength = d_processedLengths[idx];
    const bool inCompare = binarySearchGPUFast(d_target, d_targetLengths, targetCount, word, wordLength);
    if (inCompare) {
        //uint64_t i = atomicAdd(reinterpret_cast<unsigned long long*>(hitCount), 1ULL);
        if(storeHits) {
            for(int j = 0; j < wordLength; j++) {
                d_matching[idx * MAX_LEN + j] = word[j];
            }
            for(int j = wordLength; j < MAX_LEN; j++) {
                d_matching[idx * MAX_LEN + j] = '\0';
            }
            d_matchingLengths[idx] = wordLength;
        }
    }
}

// Host function with DLL_EXPORT
DLL_EXPORT
void computeCountFast(
    char *d_processed,
    uint8_t *d_processedLengths,
    char *d_target,
    uint8_t *d_targetLengths,
    char *d_matching,
    uint8_t *d_matchingLengths,
    int wordlistCount,
    int targetCount,
    uint64_t* d_hitCount,
    cudaStream_t stream,
    bool storeHits
) {
    int blocks = (wordlistCount + THREADS - 1) / THREADS;
    HitsKernel<<<blocks, THREADS, 0, stream>>>(
        d_processed,
        d_processedLengths,
        d_target,
        d_targetLengths,
        d_matching,
        d_matchingLengths,
        wordlistCount,
        targetCount,
        d_hitCount,
        storeHits
    );
    uint64_t hitCount;
    cudaMemcpyAsync(&hitCount, d_hitCount, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);

    auto exec_policy = thrust::cuda::par.on(stream);

    // Use Thrust for sorting with device lambdas
    thrust::device_vector<int> indices(wordlistCount);
    thrust::sequence(exec_policy, indices.begin(), indices.end());

    thrust::sort(exec_policy, indices.begin(), indices.end(),
        [=] __device__ (int a, int b) {
            const char* str_a = d_matching + a * MAX_LEN;
            const char* str_b = d_matching + b * MAX_LEN;
            uint8_t len_a = d_matchingLengths[a];
            uint8_t len_b = d_matchingLengths[b];

            uint8_t min_len = min(len_a, len_b);
            for (int i = 0; i < min_len; i++) {
                if (str_a[i] != str_b[i]) {
                    return str_a[i] < str_b[i];
                }
            }
            return len_a < len_b;
        }
    );

    // Use thrust::unique with device lambda
    auto new_end = thrust::unique(exec_policy, indices.begin(), indices.end(),
        [=] __device__ (int a, int b) {
            const char* str_a = d_matching + a * MAX_LEN;
            const char* str_b = d_matching + b * MAX_LEN;
            uint8_t len_a = d_matchingLengths[a];
            uint8_t len_b = d_matchingLengths[b];

            if (len_a != len_b) return false;
            for (int i = 0; i < len_a; i++) {
                if (str_a[i] != str_b[i]) return false;
            }
            return true;
        }
    );
    uint64_t unique_count = thrust::distance(indices.begin(), new_end);
    cudaMemcpyAsync(d_hitCount, &unique_count, sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
}

DLL_EXPORT
void getHitCount(uint64_t* d_hitCount, uint64_t* h_hitCount, cudaStream_t stream) {
    cudaMemcpyAsync(h_hitCount, d_hitCount, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

// Exported function to launch the hash kernel
DLL_EXPORT
void computeXXHashes(char *d_processed, int *d_processedLengths, uint64_t seed, uint64_t* d_hashes, int numWords, cudaStream_t stream) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    xxhashKernel<<<blocks, THREADS, 0, stream>>>(d_processed, d_processedLengths, seed, d_hashes, numWords);
    cudaStreamSynchronize(stream);
}

// Exported function to launch the kernel with hit count
DLL_EXPORT
uint64_t computeXXHashesWithCount(
    char *d_processed,
    uint8_t *d_processedLengths,
    const uint64_t* d_wordlistHashes,
    int wordlistCount,
    const uint64_t* d_targetHashes,
    int targetCount,
    uint64_t* d_hitCount,
    uint64_t* d_foundHashes,
    uint64_t seed,
    cudaStream_t stream
) {
    int blocks = (wordlistCount + THREADS - 1) / THREADS;
    xxhashWithHitsKernel<<<blocks, THREADS, 0, stream>>>(
        d_processed,
        d_processedLengths,
        d_wordlistHashes,
        wordlistCount,
        d_targetHashes,
        targetCount,
        d_hitCount,
        d_foundHashes,
        seed
    );

    uint64_t hitCount;
    pullHitCount(d_hitCount, &hitCount, stream);
    if(hitCount > 0) {
        auto exec_policy = thrust::cuda::par.on(stream);
        thrust::sort(exec_policy, d_foundHashes, d_foundHashes + hitCount);
        auto new_end = thrust::unique(exec_policy, d_foundHashes, d_foundHashes + hitCount);
        return new_end - d_foundHashes;
    }
    return hitCount;
}

DLL_EXPORT
void computeXXHashesWithHits(
    char *d_processed, uint8_t *d_processedLengths,
    const uint64_t* d_wordlistHashes, int wordlistCount,
    const uint64_t* d_targetHashes, int targetCount,
    uint64_t* d_hitCount,
    uint64_t* d_matchingHashes,
    uint64_t seed,
    cudaStream_t stream
) {
    int blocks = (wordlistCount + THREADS - 1) / THREADS;
    xxhashWithHitsKernel<<<blocks, THREADS, 0, stream>>>(
        d_processed, d_processedLengths,
        d_wordlistHashes, wordlistCount,
        d_targetHashes, targetCount,
        d_hitCount, d_matchingHashes,
        seed
    );

    auto exec_policy = thrust::cuda::par.on(stream);  // allocate to stream
    thrust::sort(exec_policy, d_matchingHashes, d_matchingHashes + wordlistCount);
    auto new_end = thrust::unique(exec_policy, d_matchingHashes, d_matchingHashes + wordlistCount);
    size_t unique_count = new_end - d_matchingHashes;
    pullHitCount(d_hitCount, &unique_count, stream);
}

} // extern "C"