#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define MAX_LEN 64
#define THREADS 1024
// nvcc --shared -o rules.dll rules.cu -Xcompiler "/MD" -link -cudart static
// nvcc -shared -o librules.dll rules.cu

// XXHash64 constants
#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL

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
// Lowercase kernel ("l") - converts all letters to lower case (ASCII only).
__global__ void lowerCaseKernel(char *words, int *lengths, int numWords) {
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

// Uppercase kernel ("u") - converts all letters to upper case (ASCII only).
__global__ void upperCaseKernel(char *words, int *lengths, int numWords) {
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

// Reverse kernel ("r") - reverses the characters of each word in place.
__global__ void reverseKernel(char *words, int *lengths, int numWords) {
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

// Append kernel ("$") - appends a given character to the end of each word.
__global__ void appendKernel(char *words, int *lengths, char appendChar, int numWords) {
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

// Prefix kernel ("^") - prepends a given character to each word.
__global__ void prependKernel(char *words, int *lengths, char prefixChar, int numWords) {
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

// Substitution kernel ("s") - replaces all occurrences of oldChar with newChar.
__global__ void substitutionKernel(char *words, int *lengths, char oldChar, char newChar, int numWords) {
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

// 'c' Rule: Capitalize first letter, lowercase rest
__global__ void capitalizeKernel(char* words, int* lengths, int numWords) {
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
__global__ void invertCapitalizeKernel(char* words, int* lengths, int numWords) {
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

// 'q' Rule: Duplicate each character
__global__ void duplicateCharsKernel(char* words, int* lengths, int numWords) {
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

// 'f' Rule: Append reversed string
__global__ void reflectKernel(char* words, int* lengths, int numWords) {
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
__global__ void rotateLeftKernel(char* words, int* lengths, int numWords) {
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
__global__ void rotateRightKernel(char* words, int* lengths, int numWords) {
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

// 'E' Rule: Title case (capitalize after spaces)
__global__ void titleCaseKernel(char* words, int* lengths, int numWords) {
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
__global__ void togglePositionKernel(char* words, int* lengths, int pos, int numWords) {
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

// 'D' Rule: Delete character at position
__global__ void deletePositionKernel(char* words, int* lengths, int pos, int numWords) {
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
__global__ void prependFirstCharKernel(char* words, int* lengths, int count, int numWords) {
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

// 't' Rule: Toggle case of all characters
__global__ void toggleCaseKernel(char* words, int* lengths, int numWords) {
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

// 'k' Rule: Swap first two characters
__global__ void swapFirstTwoKernel(char* words, int* lengths, int numWords) {
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
__global__ void swapLastTwoKernel(char* words, int* lengths, int numWords) {
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
__global__ void duplicateWordKernel(char* words, int* lengths, int numWords) {
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

// '[' Rule: Delete first character
__global__ void deleteFirstKernel(char* words, int* lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        if (len > 0) {
            for (int i = 0; i < len - 1; i++) {
                word[i] = word[i + 1];
            }
            lengths[idx] = len - 1;
            word[len - 1] = '\0';
        }
    }
}

// ']' Rule: Delete last character
__global__ void deleteLastKernel(char* words, int* lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        if (len > 0) {
            lengths[idx] = len - 1;
            words[idx * MAX_LEN + len - 1] = '\0';
        }
    }
}

// 'p' Rule: Repeat word N times
__global__ void repeatWordKernel(char* words, int* lengths, int count, int numWords) {
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

// 'Z' Rule: Append last character N times
__global__ void appendLastCharKernel(char* words, int* lengths, int count, int numWords) {
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
__global__ void truncateAtKernel(char* words, int* lengths, int pos, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        if (pos < len) {
            lengths[idx] = pos;
            words[idx * MAX_LEN + pos] = '\0';
        }
    }
}

// 'S' Rule: Replace first occurrence
__global__ void replaceFirstKernel(char* words, int* lengths, char oldChar, char newChar, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int len = lengths[idx];
        char* word = &words[idx * MAX_LEN];
        for (int i = 0; i < len; i++) {
            if (word[i] == oldChar) {
                word[i] = newChar;
                break;
            }
        }
    }
}

// 'y' Rule: Prepend substring
__global__ void prependSubstrKernel(char* words, int* lengths, int count, int numWords) {
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
__global__ void appendSubstrKernel(char* words, int* lengths, int count, int numWords) {
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
__global__ void bitShiftLeftKernel(char* words, int* lengths, int pos, int numWords) {
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
__global__ void bitShiftRightKernel(char* words, int* lengths, int pos, int numWords) {
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
__global__ void decrementCharKernel(char* words, int* lengths, int pos, int numWords) {
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
__global__ void incrementCharKernel(char* words, int* lengths, int pos, int numWords) {
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
__global__ void deleteAllCharKernel(char* words, int* lengths, char target, int numWords) {
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
__global__ void swapNextKernel(char* words, int* lengths, int pos, int numWords) {
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



//---------------------------------------------------------------------
// Host Wrappers to Launch Kernels
//---------------------------------------------------------------------
// l
__declspec(dllexport)
void applyLowerCase(char *d_words, int *d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    lowerCaseKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// u
__declspec(dllexport)
void applyUpperCase(char *d_words, int *d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    upperCaseKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// c
__declspec(dllexport)
void applyCapitalize(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    capitalizeKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// C
__declspec(dllexport)
void applyInvertCapitalize(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    invertCapitalizeKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// t
__declspec(dllexport)
void applyToggleCase(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    toggleCaseKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// q
__declspec(dllexport)
void applyDuplicateChars(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    duplicateCharsKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// r
__declspec(dllexport)
void applyReverse(char *d_words, int *d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    reverseKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// k
__declspec(dllexport)
void applySwapFirstTwo(char *d_words, int *d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapFirstTwoKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// K
__declspec(dllexport)
void applySwapLastTwo(char *d_words, int *d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapLastTwoKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// d
__declspec(dllexport)
void applyDuplicate(char *d_words, int *d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    duplicateWordKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// f
__declspec(dllexport)
void applyReflect(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    reflectKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// {
__declspec(dllexport)
void applyRotateLeft(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rotateLeftKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
 // }
__declspec(dllexport)
void applyRotateRight(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    rotateRightKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// [
__declspec(dllexport)
void applyDeleteFirst(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deleteFirstKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// ]
__declspec(dllexport)
void applyDeleteLast(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deleteLastKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// E
__declspec(dllexport)
void applyTitleCase(char* d_words, int* d_lengths, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    titleCaseKernel<<<blocks, THREADS>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();
}
// T
__declspec(dllexport)
void applyTogglePosition(char* d_words, int* d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    togglePositionKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}
// p
__declspec(dllexport)
void applyRepeatWord(char* d_words, int* d_lengths, int count, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    repeatWordKernel<<<blocks, THREADS>>>(d_words, d_lengths, count, numWords);
    cudaDeviceSynchronize();
}
// D
__declspec(dllexport)
void applyDeletePosition(char* d_words, int* d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deletePositionKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}

// z
__declspec(dllexport)
void applyPrependFirstChar(char* d_words, int* d_lengths, int count, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependFirstCharKernel<<<blocks, THREADS>>>(d_words, d_lengths, count, numWords);
    cudaDeviceSynchronize();
}

// Z
__declspec(dllexport)
void applyAppendLastChar(char* d_words, int* d_lengths, int count, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependFirstCharKernel<<<blocks, THREADS>>>(d_words, d_lengths, count, numWords);
    cudaDeviceSynchronize();
}
// '
__declspec(dllexport)
void applyTruncateAt(char* d_words, int* d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    truncateAtKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}
// s
__declspec(dllexport)
void applySubstitution(char *d_words, int *d_lengths, char oldChar, char newChar, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    substitutionKernel<<<blocks, THREADS>>>(d_words, d_lengths, oldChar, newChar, numWords);
    cudaDeviceSynchronize();
}
// S
__declspec(dllexport)
void applyReplaceFirst(char *d_words, int *d_lengths, char oldChar, char newChar, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    replaceFirstKernel<<<blocks, THREADS>>>(d_words, d_lengths, oldChar, newChar, numWords);
    cudaDeviceSynchronize();
}
// $
__declspec(dllexport)
void applyAppend(char *d_words, int *d_lengths, char appendChar, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    appendKernel<<<blocks, THREADS>>>(d_words, d_lengths, appendChar, numWords);
    cudaDeviceSynchronize();
}
// ^
__declspec(dllexport)
void applyPrepend(char *d_words, int *d_lengths, char prefixChar, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependKernel<<<blocks, THREADS>>>(d_words, d_lengths, prefixChar, numWords);
    cudaDeviceSynchronize();
}
// y
__declspec(dllexport)
void applyAppendSuffixSubstr(char *d_words, int *d_lengths, int count, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    appendSubstrKernel<<<blocks, THREADS>>>(d_words, d_lengths, count, numWords);
    cudaDeviceSynchronize();
}
// Y
__declspec(dllexport)
void applyPrependPrefixSubstr(char *d_words, int *d_lengths, int count, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    prependSubstrKernel<<<blocks, THREADS>>>(d_words, d_lengths, count, numWords);
    cudaDeviceSynchronize();
}
// L
__declspec(dllexport)
void applyBitShiftLeft(char *d_words, int *d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    bitShiftLeftKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}
// R
__declspec(dllexport)
void applyBitShiftRight(char *d_words, int *d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    bitShiftRightKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}
// -
__declspec(dllexport)
void applyDecrementChar(char *d_words, int *d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    decrementCharKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}
// +
__declspec(dllexport)
void applyIncrementChar(char *d_words, int *d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    incrementCharKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}
// @
__declspec(dllexport)
void applyDeleteAllChar(char *d_words, int *d_lengths, char deleteChar, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    deleteAllCharKernel<<<blocks, THREADS>>>(d_words, d_lengths, deleteChar, numWords);
    cudaDeviceSynchronize();
}
// .
__declspec(dllexport)
void applySwapNext(char *d_words, int *d_lengths, int pos, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    swapNextKernel<<<blocks, THREADS>>>(d_words, d_lengths, pos, numWords);
    cudaDeviceSynchronize();
}



// Existing helper functions (unchanged)
__declspec(dllexport)
void allocateOriginalDictMemoryOnGPU(
    char **d_originalDict, int **d_originalDictLengths,
    char *h_originalDict, int *h_originalDictLengths, int numWords
) {
    cudaMalloc((void**)d_originalDict, numWords * MAX_LEN * sizeof(char));
    cudaMalloc((void**)d_originalDictLengths, numWords * sizeof(int));

    cudaMemcpy(*d_originalDict, h_originalDict, numWords * MAX_LEN * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_originalDictLengths, h_originalDictLengths, numWords * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

__declspec(dllexport)
void allocateProcessedDictMemoryOnGPU(
    char **d_originalDict, int **d_originalDictLengths, uint64_t **d_hashes,
    char **d_processedDict, int **d_processedDictLengths, uint64_t *h_hashes, uint64_t numWords
) {
    cudaMalloc((void**)d_processedDict, numWords * MAX_LEN * sizeof(char));
    cudaMalloc((void**)d_processedDictLengths, numWords * sizeof(int));
    cudaMalloc((void**)d_hashes, numWords * sizeof(uint64_t));

    cudaMemcpy(*d_processedDict, d_originalDict, numWords * MAX_LEN * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_processedDictLengths, d_originalDictLengths, numWords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hashes, h_hashes, numWords * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

__declspec(dllexport)
void copyMemoryBackToHost(uint64_t* h_hashes, uint64_t *d_hashes, int numWords) {
    cudaMemcpy(h_hashes, d_hashes, numWords * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

__declspec(dllexport)
void freeOriginalMemoryOnGPU(int *d_originalDict, int *d_originalDictLengths) {
    cudaFree(d_originalDict);
    cudaFree(d_originalDictLengths);
    cudaDeviceSynchronize();
}

__declspec(dllexport)
void freeProcessedMemoryOnGPU(char *d_processedDict, int *d_processedDictLengths, uint64_t *d_hashes) {
    cudaFree(d_processedDict);
    cudaFree(d_processedDictLengths);
    cudaFree(d_hashes);
    cudaDeviceSynchronize();
}

__global__ void xxhashKernel(char* d_processedDict, int* d_processedDictLengths , uint64_t seed, uint64_t* hashes, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWords) return;

    char* input = &d_processedDict[idx * MAX_LEN];
    hashes[idx] = xxhash64(input, d_processedDictLengths[idx], seed);
}

// Exported function to launch the kernel

__declspec(dllexport)
void computeXXHashes(char *d_processedDict, int *d_processedDictLengths, uint64_t seed, uint64_t* d_hashes, int numWords) {
    int blocks = (numWords + THREADS - 1) / THREADS;
    xxhashKernel<<<blocks, THREADS>>>(d_processedDict, d_processedDictLengths , seed, d_hashes, numWords);
    cudaDeviceSynchronize();
}

} // extern "C"