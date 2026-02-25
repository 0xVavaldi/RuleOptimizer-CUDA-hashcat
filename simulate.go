package main

/*
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
#cgo windows LDFLAGS: -L. -lrules -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart -lcuda
#cgo !windows LDFLAGS: -L. -lrules -lcudart -lcuda
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

void allocateDictionary(char **d_wordlist, uint8_t **d_wordlistLengths, int wordlistCount, cudaStream_t stream);
void pushDictionary(char *h_wordlist, uint8_t *h_wordlistLengths, char **d_wordlist, uint8_t **d_wordlistLengths, int wordlistCount, cudaStream_t stream);
void overwriteDictionary(char **d_wordlist, uint8_t **d_wordlistLengths, char **d_overwrite, uint8_t **d_overwriteLengths, int wordlistCount, cudaStream_t stream);
void resetDictionary(char **d_wordlist, uint8_t **d_wordlistLengths, int wordlistCount, cudaStream_t stream);
void pullDictionary(char **d_wordlist, uint8_t **d_wordlistLengths, char *h_wordlist, uint8_t *h_wordlistLengths, int wordlistCount, cudaStream_t stream);
void deallocateDictionary(char *d_wordlist, uint8_t *d_wordlistLengths, cudaStream_t stream);

void allocateHashes(uint64_t **d_hashes, int hashCount, cudaStream_t stream);
void pushHashes(uint64_t *h_hashes, uint64_t **d_hashes, int hashCount, cudaStream_t stream);
void overwriteHashes(uint64_t **d_hashes, uint64_t **d_overwrite, int hashCount, cudaStream_t stream);
void pullHashes(uint64_t **d_hashes, uint64_t *h_hashes, int hashCount, cudaStream_t stream);
void deallocateHashes(uint64_t *d_hashes, cudaStream_t stream);

void initializeHitCount(uint64_t **d_hitCount, cudaStream_t stream);
void pullHitCount(uint64_t *d_hitCount, uint64_t* h_hitCount, cudaStream_t stream);
void resetHitCount(uint64_t **d_hitCount, cudaStream_t stream);
void deallocateHitCount(uint64_t *d_hitCount, cudaStream_t stream);

void computeXXHashes(char* d_words, int* d_lengths, uint64_t seed, uint64_t* d_hashes, int numWords);
void computeXXHashesWithHits(char *processedDict, uint8_t *processedLengths, uint64_t seed, const uint64_t *originalHashes, int originalCount, uint64_t *compareHashes, int compareCount, uint64_t *hitCount, uint64_t *matchingHashes, cudaStream_t stream);
uint64_t computeXXHashesWithCount(char *processedDict, uint8_t *processedLengths, const uint64_t *originalHashes, int originalCount, uint64_t *compareHashes, int compareCount, uint64_t *hitCount, uint64_t *matchingHashes, uint64_t seed, cudaStream_t stream);
void computeCountFast(char *d_processedDict, uint8_t *d_processedLengths, char *d_target, uint8_t *d_targetLengths, char *d_matching, uint8_t *d_matchingLengths, int wordlistCount, int targetCount, uint64_t *hitCount, cudaStream_t stream, bool storeHits);
*/
import "C"
import (
	"log"
	"runtime"
	"time"

	"github.com/schollz/progressbar/v3"
)

func generateSimulate(cli CLI) {
	log.Println("Loading Wordlist")
	wordlist, wordlistLengths, wordlistCount, _ := loadWordlist(cli.Simulate.Wordlist)
	wordlistHashes := loadHashedWordlist(cli.Simulate.Wordlist)
	log.Println("Loading Target")
	targetHashes := loadHashedWordlist(cli.Simulate.Target)
	targetHashes, _ = cleanWords(targetHashes, wordlistHashes)
	wordlistHashes = nil // clear hashes after removing founds.

	targetCount := len(targetHashes)
	if targetCount == 0 {
		log.Fatalf("No targets loaded, please check your input file and verify they are <%d characters", MaxLen)
	}
	log.Println("Loading Rules")
	rules := loadRulesFast(cli.Simulate.RuleFile)
	log.Printf("Loaded %d Rules", len(rules))

	// Convert to 2d byte array in chunks of size MaxLen
	wordlistBytes := make([]byte, (wordlistCount)*MaxLen)
	for i, word := range wordlist {
		copy(wordlistBytes[i*MaxLen:], word)
	}
	wordlist = nil

	// because it's single-threaded basically, it's hard to multi-process with dependencies unless you look ahead and then zap (but that is a tradeoff)

	deviceID := 0
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	CUDASetDevice(deviceID)
	stream := CUDAInitializeStream()
	gpuWordlist, gpuWordlistLengths := cudaInitializeDict(&wordlistBytes, &wordlistLengths, wordlistCount, stream)

	processedHashes := make([]uint64, wordlistCount)
	gpuProcessed, gpuProcessedLengths := cudaInitializeDict(&wordlistBytes, &wordlistLengths, wordlistCount, stream)
	gpuProcessedHashes := cudaInitializeHashes(&processedHashes, wordlistCount, stream)

	gpuFoundHashes := cudaInitializeHashes(&processedHashes, wordlistCount, stream)
	gpuTargetHashes := cudaInitializeHashes(&targetHashes, targetCount, stream)
	gpuHitCount := cudaInitializeHitCount(stream)

	// Clean up after we finish working
	defer func(
		gpuWordlist *C.char, gpuWordlistLengths *C.uint8_t,
		gpuProcessed *C.char, gpuProcessedLengths *C.uint8_t,
		gpuProcessedHashes *C.uint64_t,
		gpuFoundHashes *C.uint64_t,
		gpuTargetHashes *C.uint64_t,
		gpuHitCount *C.uint64_t,
		stream C.cudaStream_t,
	) {
		cudaDeinitializeDict(gpuWordlist, gpuWordlistLengths, stream)
		cudaDeinitializeDict(gpuProcessed, gpuProcessedLengths, stream)
		cudaDeinitializeHashes(gpuProcessedHashes, stream)
		cudaDeinitializeHashes(gpuFoundHashes, stream)
		cudaDeinitializeHashes(gpuTargetHashes, stream)
		cudaDeinitializeHitCount(gpuHitCount, stream)
		CUDADeinitializeStream(stream)
		runtime.UnlockOSThread()
	}(
		gpuWordlist, gpuWordlistLengths,
		gpuProcessed, gpuProcessedLengths,
		gpuProcessedHashes,
		gpuFoundHashes,
		gpuTargetHashes,
		gpuHitCount,
		stream,
	)

	log.Println("Starting simulation mode - processing rules in order")

	// Track total progress
	totalHits := uint64(0)
	processedRules := 0

	processBar := progressbar.NewOptions(len(rules)*wordlistCount,
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowDescriptionAtLineEnd(),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionThrottle(1000*time.Millisecond),
		progressbar.OptionShowElapsedTimeOnFinish(),
		progressbar.OptionSetWidth(25),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
	)
	// Process each rule sequentially without optimization
	for i := range rules {
		if totalHits == uint64(targetCount) {
			log.Println("All target words found, stopping simulation")
			break
		}
		//log.Printf("Processing rule %d/%d (remaining targets: %d)", i+1, len(rules), targetCount)

		// Process single rule with existing GPU resources
		rule := &rules[i]
		hits := CUDASingleRuleScore(
			&rule.RuleLine,
			gpuWordlist, gpuWordlistLengths,
			gpuProcessed, gpuProcessedLengths,
			gpuProcessedHashes, wordlistCount,
			gpuTargetHashes, targetCount,
			gpuHitCount,
			gpuFoundHashes,
			stream,
		)
		processBar.Add(wordlistCount)

		if hits > 0 {
			// Get the found hashes from GPU
			hashedWords := CUDAGetHashes(hits, gpuFoundHashes, stream)
			// Remove found words from target and calculate hits
			ruleHits := 0
			targetHashes, ruleHits = cleanWords(targetHashes, hashedWords)
			totalHits += uint64(ruleHits)
			//log.Printf("Rule %d found %d new words (total: %d)", i+1, ruleHits, totalHits)
			appendScoreToFile(cli.Simulate.OutputFile, uint64(ruleHits), rule)
		} else {
			appendScoreToFile(cli.Simulate.OutputFile, uint64(0), rule)
		}
		processedRules++
	}
	processBar.Finish()
	processBar.Close()
	println()
	log.Printf("Simulation completed: %d rules processed, %d total hits, %d targets remaining",
		processedRules, totalHits, targetCount)
}
