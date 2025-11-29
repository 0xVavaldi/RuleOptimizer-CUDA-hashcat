package main

/*
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
#cgo LDFLAGS: -L. -lrules -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart -lcuda
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
void resetHashes(uint64_t **d_hashes, int hashCount, cudaStream_t stream);
void resetHitCount(uint64_t **d_hitCount, cudaStream_t stream);
void deallocateHitCount(uint64_t *d_hitCount, cudaStream_t stream);

void computeXXHashes(char* d_words, int* d_lengths, uint64_t seed, uint64_t* d_hashes, int numWords);
void computeXXHashesWithHits(char *processedDict, uint8_t *processedLengths, uint64_t seed, const uint64_t *originalHashes, int originalCount, uint64_t *compareHashes, int compareCount, uint64_t *hitCount, uint64_t *matchingHashes, cudaStream_t stream);
uint64_t computeXXHashesWithCount(char *processedDict, uint8_t *processedLengths, const uint64_t *originalHashes, int originalCount, uint64_t *compareHashes, int compareCount, uint64_t *hitCount, uint64_t *matchingHashes, uint64_t seed, cudaStream_t stream);
void computeCountFast(char *d_processedDict, uint8_t *d_processedLengths, char *d_target, uint8_t *d_targetLengths, char *d_matching, uint8_t *d_matchingLengths, int wordlistCount, int targetCount, uint64_t *hitCount, cudaStream_t stream, bool storeHits);
*/
import "C"
import (
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
)

// d_ and gpu prefix define variables that exist on the GPU
// Count suffix defines the amount of elements in the array
// Lengths suffix defines the lengths of each element respectively
func generatePhase1(cli CLI) {
	//wordlistHashes := loadHashedWordlist(cli.Score.Wordlist)
	wordlist, wordlistLengths, wordlistCount, _ := loadWordlist(cli.Score.Wordlist)
	targetHashes := loadHashedWordlist(cli.Score.Target)
	targetCount := len(targetHashes)
	rules := loadRulesFast(cli.Score.RuleFile)

	totalMemoryEstimate := wordlistCount*4 + wordlistCount*1 + targetCount*4 + targetCount*1
	log.Printf("Expected memory usage is: %dGB", totalMemoryEstimate/1000000000)
	deviceCount := CUDAGetDeviceCount()
	log.Printf("Detected %d GPU devices. Initializing", deviceCount)

	// Convert to 2d byte array in chunks of size MaxLen
	wordlistBytes := make([]byte, (wordlistCount)*MaxLen)
	for i, word := range wordlist {
		copy(wordlistBytes[i*MaxLen:], word)
	}
	wordlist = nil

	processRuleFile(
		cli,
		rules,
		deviceCount,
		&wordlistBytes,
		&wordlistLengths,
		&targetHashes,
		wordlistCount,
		targetCount,
	)
}

func processRuleFile(
	cli CLI,
	rules []ruleObj,
	deviceCount int,
	wordlistBytes *[]byte,
	wordlistLengths *[]uint8,
	targetHashes *[]uint64,
	wordlistCount int,
	targetCount int,
) {
	ruleChan := make(chan *ruleObj, 5)
	var writerMutex sync.Mutex
	var wg sync.WaitGroup
	var wgResult sync.WaitGroup
	wg.Add(deviceCount)
	wgResult.Add(1)

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

	type result struct {
		hits uint64
		rule *ruleObj
	}
	resultChan := make(chan result) // Buffer size can be adjusted

	go func() {
		defer wgResult.Done()
		outputFile, err := os.OpenFile(cli.Score.OutputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Println("Error opening or creating file:", err)
			return
		}
		defer outputFile.Close()

		for res := range resultChan {
			if _, err = outputFile.WriteString(strconv.FormatUint(res.hits, 10) + "\t" + FormatAllRules(res.rule.RuleLine) + "\n"); err != nil {
				fmt.Println("Error writing to file:", err)
				continue
			}
		}
	}()

	for i := 0; i < deviceCount; i++ {
		go func() {
			runtime.LockOSThread()
			CUDASetDevice(i)
			stream := CUDAInitializeStream()
			gpuWordlist, gpuWordlistLengths := cudaInitializeDict(wordlistBytes, wordlistLengths, wordlistCount, stream)

			processedHashes := make([]uint64, wordlistCount)
			gpuProcessed, gpuProcessedLengths := cudaInitializeDict(wordlistBytes, wordlistLengths, wordlistCount, stream)
			gpuProcessedHashes := cudaInitializeHashes(&processedHashes, wordlistCount, stream)

			gpuFoundHashes := cudaInitializeHashes(&processedHashes, wordlistCount, stream)
			gpuTargetHashes := cudaInitializeHashes(targetHashes, targetCount, stream)
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
				wg.Done()
			}(
				gpuWordlist, gpuWordlistLengths,
				gpuProcessed, gpuProcessedLengths,
				gpuProcessedHashes,
				gpuFoundHashes,
				gpuTargetHashes,
				gpuHitCount,
				stream,
			)

			// Initialize processed variables for this worker
			// Process each rule received from the channel and binary search the results against the original wordlist
			// to extract new entries that don't already exist. wordlistHashes lives per gpu device
			for rule := range ruleChan {
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

				writerMutex.Lock()
				processBar.Add(wordlistCount)
				resultChan <- result{hits: hits, rule: rule}
				writerMutex.Unlock()
			}
		}()
	}

	// Send all rules to the workers
	for i := range rules {
		ruleChan <- &rules[i]
	}
	close(ruleChan)
	wg.Wait()
	close(resultChan)
	wgResult.Wait()
	processBar.Close()
	processBar.Finish()
	return
}

// ~10% improvement by using wordlist instead of hashes. Trading VRAM for speed.
func generatePhase1Fast(cli CLI) {
	wordlist, wordlistLengths, wordlistCount, _ := loadWordlist(cli.Score.Wordlist)
	target, targetLengths, targetCount, _ := loadWordlist(cli.Score.Target)
	target = removeAfromB(wordlist, target)

	log.Println("Loading Rules")
	log.Println("Estimated memory usage wordlist: ", ByteCountSI(wordlistCount*32+wordlistCount))
	log.Println("Estimated memory usage target: ", ByteCountSI(targetCount*32+targetCount))
	log.Println("Estimated memory usage buffer: ", ByteCountSI(wordlistCount*32+wordlistCount))
	log.Println("Estimated memory hits: ", ByteCountSI(wordlistCount*32))
	log.Println("Estimated Total Memory: ", ByteCountSI(wordlistCount*96+targetCount*32+wordlistCount*4+targetCount*4))
	rules := loadRulesFast(cli.Score.RuleFile)
	// Start processing
	// Start processing
	// Start processing
	wordlistBytes := make([]byte, wordlistCount*MaxLen)
	for i, word := range wordlist {
		copy(wordlistBytes[i*MaxLen:], word)
		wordlist[i] = ""
	}
	wordlist = nil

	targetBytes := make([]byte, targetCount*MaxLen)
	for i, word := range target {
		copy(targetBytes[i*MaxLen:], word)
		target[i] = ""
	}
	targetBytes = nil

	processRuleFileFast(
		cli,
		rules,
		&wordlistBytes,
		&wordlistLengths,
		&targetBytes,
		&targetLengths,
		wordlistCount,
		targetCount,
	)
}

func processRuleFileFast(
	cli CLI,
	rules []ruleObj,
	wordlistBytes *[]byte,
	wordlistLengths *[]uint8,
	targetBytes *[]byte,
	targetLengths *[]uint8,
	wordlistCount int,
	targetCount int,
) {
	deviceCount := CUDAGetDeviceCount()
	log.Printf("Detected %d GPU devices.", deviceCount)

	ruleChan := make(chan *ruleObj, 5)
	var writerMutex sync.Mutex
	var wg sync.WaitGroup
	var wgg sync.WaitGroup
	wg.Add(deviceCount)
	wgg.Add(1)

	totalMemoryEstimate := wordlistCount*MaxLen + wordlistCount*MaxLen + targetCount*MaxLen + targetCount*MaxLen
	fmt.Printf("Expected memory usage is: %dGB", totalMemoryEstimate/1000000000)

	processBar := progressbar.NewOptions(wordlistCount*len(rules),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowDescriptionAtLineEnd(),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionThrottle(1000*time.Millisecond),
		progressbar.OptionShowElapsedTimeOnFinish(),
		progressbar.OptionSetWidth(25),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
	)

	type result struct {
		hits uint64
		rule *ruleObj
	}
	resultChan := make(chan result) // Buffer size can be adjusted

	go func() {
		defer wgg.Done()
		outputFile, err := os.OpenFile(cli.Score.OutputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Println("Error opening or creating output score file:", err)
			return
		}
		defer outputFile.Close()

		for res := range resultChan {
			if _, err = outputFile.WriteString(strconv.FormatUint(res.hits, 10) + "\t" + FormatAllRules(res.rule.RuleLine) + "\n"); err != nil {
				fmt.Println("Error writing to output score file:", err)
				continue
			}
		}
	}()

	for i := 0; i < deviceCount; i++ {
		go func() {
			runtime.LockOSThread()
			CUDASetDevice(i)
			stream := CUDAInitializeStream()
			gpuWordlist, gpuWordlistLengths := cudaInitializeDict(wordlistBytes, wordlistLengths, wordlistCount, stream)
			gpuProcessed, gpuProcessedLengths := cudaInitializeDict(wordlistBytes, wordlistLengths, wordlistCount, stream)
			gpuTarget, gpuTargetLengths := cudaInitializeDict(targetBytes, targetLengths, targetCount, stream)
			gpuMatching, gpuMatchingLengths := cudaAllocateDict(wordlistCount, stream)
			gpuHitCount := cudaInitializeHitCount(stream)
			defer func(
				gpuWordlist *C.char, gpuWordlistLengths *C.uint8_t,
				gpuProcessed *C.char, gpuProcessedLengths *C.uint8_t,
				gpuTarget *C.char, gpuTargetLengths *C.uint8_t,
				gpuMatching *C.char, gpuMatchingLengths *C.uint8_t,
				gpuHitCount *C.uint64_t,
				stream C.cudaStream_t,
			) {
				cudaDeinitializeDict(gpuWordlist, gpuWordlistLengths, stream)
				cudaDeinitializeDict(gpuProcessed, gpuProcessedLengths, stream)
				cudaDeinitializeDict(gpuTarget, gpuTargetLengths, stream)
				cudaDeinitializeDict(gpuMatching, gpuMatchingLengths, stream)
				cudaDeinitializeHitCount(gpuHitCount, stream)
				CUDADeinitializeStream(stream)
				runtime.UnlockOSThread()
				wg.Done()
			}(
				gpuWordlist, gpuWordlistLengths,
				gpuProcessed, gpuProcessedLengths,
				gpuTarget, gpuTargetLengths,
				gpuMatching, gpuMatchingLengths,
				gpuHitCount,
				stream,
			)
			// Initialize processed variables for this worker
			// Process each rule received from the channel and binary search the results against the original wordlist
			// to extract new entries that don't already exist. wordlistHashes lives per gpu device.
			storeHits := true
			for rule := range ruleChan {
				hits := CUDASingleRuleScoreFast(
					&rule.RuleLine,
					gpuWordlist, gpuWordlistLengths,
					gpuProcessed, gpuProcessedLengths,
					gpuTarget, gpuTargetLengths,
					gpuMatching, gpuMatchingLengths,
					wordlistCount, targetCount,
					gpuHitCount,
					stream,
					storeHits,
				)

				writerMutex.Lock()
				processBar.Add(wordlistCount)
				resultChan <- result{hits: hits, rule: rule}
				writerMutex.Unlock()
			}
		}()
	}

	// Send all rules to the workers
	for i := range rules {
		ruleChan <- &rules[i]
	}
	close(ruleChan)
	wg.Wait()
	close(resultChan)
	wgg.Wait()
	processBar.Close()
	processBar.Finish()
	return
}
