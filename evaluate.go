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
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
)

func prepareCompare(wordlist, target string) ([]uint64, int, []uint64, int) {
	var startup sync.WaitGroup
	startup.Add(2)

	wordlistCount, _ := lineCounter(wordlist)
	originalHashes := make([]uint64, 0, wordlistCount)
	go func() {
		defer startup.Done()
		originalHashes = loadHashedWordlist(wordlist)
	}()

	var compareDictHashes []uint64
	var compareDictCount int
	if fileExists(stateHashFile) {
		var err error
		go func() {
			defer startup.Done()
			compareDictHashes, err = loadCompareDictState(stateHashFile)
			if err != nil {
				log.Printf("Error loading hash state: %v", err)
				return
			}
			compareDictCount = len(compareDictHashes)
		}()
	} else {
		go func() {
			defer startup.Done()
			compareDictCount, _ = lineCounter(target)
			compareDictHashes = make([]uint64, 0, compareDictCount)
			compareDictHashes = loadHashedWordlist(target)
		}()
	}

	startup.Wait()
	return originalHashes, wordlistCount, compareDictHashes, compareDictCount
}

func generatePhase2(cli CLI) {
	originalHashes, wordlistCount, compareDictHashes, compareDictCount := prepareCompare(cli.Evaluate.Wordlist, cli.Evaluate.Target)
	// End startup preparation
	wordlist, wordlistLengths, wordlistCount := loadWordlist(cli.Evaluate.Wordlist)

	log.Println("Loading Rule Scores")
	var rules []ruleObj
	if fileExists(stateFile) {
		fmt.Println("Loading from state file...")
		var err error
		rules = loadStateScores(stateFile)
		if err != nil {
			log.Printf("Error loading state: %v, loading from original file", err)
			rules = loadRuleScores(cli.Evaluate.ScoreFile)
		}
	} else {
		// Load from original file
		rules = loadRuleScores(cli.Evaluate.ScoreFile)
	}
	log.Printf("Loaded %d Rule Scores", len(rules))

	// Start processing
	// Start processing
	// Start processing. Don't assume they are sorted properly but ensure they are.
	deviceCount := CUDAGetDeviceCount()
	log.Printf("Detected %d GPU devices. Sorting rules (can take a minute+)", deviceCount)

	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Fitness > rules[j].Fitness
	})
	// Compute Rule

	wordlistBytes := make([]byte, (wordlistCount)*MaxLen)
	for i, word := range wordlist {
		copy(wordlistBytes[i*MaxLen:], word)
	}
	wordlist = nil

	lastBestFitness := rules[0].Fitness
	if fileExists(stateFile) {
		lastBestFitness = rules[len(rules)-1].Fitness
	}
	var bestRule *ruleObj
	var hashedWords []uint64

	saveEvery := 100
	processed := 0
	for lastBestFitness > 0 {
		// todo: find a way to reduce rules in size (memory usage)
		rules, bestRule, hashedWords = processRuleRound(
			rules,
			deviceCount,
			&wordlistBytes,
			&wordlistLengths,
			&originalHashes,
			&compareDictHashes,
			wordlistCount,
			compareDictCount,
			lastBestFitness,
		)
		log.Printf("New candidate %d with score: %d", bestRule.ID, bestRule.LastFitness)
		lastBestFitness = bestRule.LastFitness
		rules[bestRule.ID-1].LastFitness = 0 // ID starts at 1

		compareDictHashes = cleanWords(&compareDictHashes, &hashedWords)
		compareDictCountNew := len(compareDictHashes)

		print()
		log.Printf("%d new words found.", compareDictCount-compareDictCountNew)
		appendScoreToFile(cli.Evaluate.OutputFile, uint64(compareDictCount-compareDictCountNew), bestRule)
		compareDictCount = compareDictCountNew

		processed += 1
		if processed%saveEvery == 0 {
			// Sort rules by LastFitness in descending order
			err := saveState(stateFile, stateHashFile, &rules, compareDictHashes)
			if err != nil {
				log.Printf("Error saving state: %v", err)
				return // state file is corrupt, better to stop early.
			}
		}
	}
}

func processRuleRound(rules []ruleObj, deviceCount int, wordlistBytes *[]byte, wordlistLengths *[]uint8, wordlistHashes *[]uint64, targetHashes *[]uint64, wordlistCount int, targetCount int, lastFitness uint64) ([]ruleObj, *ruleObj, []uint64) {
	deviceCount = CUDAGetDeviceCount()
	ruleChan := make(chan *ruleObj, deviceCount)
	var writerMutex sync.Mutex
	var wg sync.WaitGroup
	wg.Add(deviceCount)

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

	currentBestFitnessRule := &rules[0]
	hashedWords := make([]uint64, lastFitness)

	for i := 0; i < deviceCount; i++ {
		go func() {
			runtime.LockOSThread()
			CUDASetDevice(i)
			stream := CUDAInitializeStream()
			gpuWordlist, gpuWordlistLengths := cudaInitializeDict(wordlistBytes, wordlistLengths, wordlistCount, stream)

			processedHashes := make([]uint64, wordlistCount)
			gpuProcessed, gpuProcessedLengths := cudaInitializeDict(wordlistBytes, wordlistLengths, wordlistCount, stream)
			gpuProcessedHashes := cudaInitializeHashes(&processedHashes, wordlistCount, stream)

			gpuFoundHashes := cudaInitializeHashes(&processedHashes, targetCount, stream)
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
			// to extract new entries that don't already exist. originalHashes lives per gpu device
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
				if len(rule.RuleLine) <= 5 { // experimental 2025-08-22 with the purpose of promoting shorter processedHashes in a decreasing linear algorithm
					hits *= uint64(10)
					hits /= uint64(5 + len(rule.RuleLine))
				}
				// Write results with existing mutex
				processBar.Add(wordlistCount)
				rule.LastFitness = hits
				if hits >= currentBestFitnessRule.LastFitness {
					// double validation with mutex to guarantee it's accurate
					writerMutex.Lock()
					if hits == currentBestFitnessRule.LastFitness {
						// If it's smaller in operations with the same hits, take that instead
						if len(rule.RuleLine) < len(currentBestFitnessRule.RuleLine) {
							currentBestFitnessRule = rule
							hashedWords = CUDAGetHashes(hits, gpuFoundHashes, stream)
						}
					} else if hits > currentBestFitnessRule.LastFitness {
						currentBestFitnessRule = rule
						hashedWords = CUDAGetHashes(hits, gpuFoundHashes, stream)
					}
					writerMutex.Unlock()
				}
			}
		}()
	}

	// Send all rules to the workers
	for i := range rules {
		if rules[i].LastFitness == 0 { // Must have the potential to be better
			processBar.Add(wordlistCount)
			continue
		}
		if lastFitness == 0 {
			ruleChan <- &rules[i]
			continue
		}

		if rules[i].LastFitness < 250000 && rules[i].LastFitness >= 100000 && rules[i].LastFitness < lastFitness-5000 && lastFitness >= 5000 { // if it's far off, skip it
			processBar.Add(wordlistCount)
			continue
		}
		if rules[i].LastFitness < 80000 && rules[i].LastFitness >= 50000 && rules[i].LastFitness < lastFitness-2000 && lastFitness >= 2000 { // if it's far off, skip it
			processBar.Add(wordlistCount)
			continue
		}
		if rules[i].LastFitness < 50000 && rules[i].LastFitness >= 10000 && rules[i].LastFitness < lastFitness-500 && lastFitness >= 500 { // if it's far off, skip it
			processBar.Add(wordlistCount)
			continue
		}
		if rules[i].LastFitness < 10000 && rules[i].LastFitness < lastFitness-100 && lastFitness >= 100 { // if it's far off, skip it
			processBar.Add(wordlistCount)
			continue
		}

		if rules[i].LastFitness < 3000 && rules[i].LastFitness < lastFitness-10 && lastFitness >= 10 { // if it's far off, skip it
			processBar.Add(wordlistCount)
			continue
		}
		if rules[i].LastFitness < currentBestFitnessRule.LastFitness { // Must have the potential to be better
			processBar.Add(wordlistCount)
			continue
		}
		if rules[i].Fitness < currentBestFitnessRule.LastFitness { // No potentials left
			break
		}
		ruleChan <- &rules[i]
	}

	close(ruleChan)
	wg.Wait()
	processBar.Close()
	processBar.Finish()
	return rules, currentBestFitnessRule, hashedWords
}
