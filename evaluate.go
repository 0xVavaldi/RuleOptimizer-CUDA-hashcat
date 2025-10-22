package main

import "C"
import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
)

func prepareCompare(wordlist, target string) ([]uint64, int, []uint64, int) {
	var startup sync.WaitGroup
	startup.Add(2)

	originalDictCount, _ := lineCounter(wordlist)
	originalHashes := make([]uint64, 0, originalDictCount)
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
	return originalHashes, originalDictCount, compareDictHashes, compareDictCount
}

func generatePhase2(cli CLI) {
	originalHashes, originalDictCount, compareDictHashes, compareDictCount := prepareCompare(cli.Evaluate.Wordlist, cli.Evaluate.Target)
	originalDict := make([]string, 0, originalDictCount)
	// End startup preparation

	file, err := os.Open(cli.Evaluate.Wordlist)
	if err != nil {
		log.Println("Error opening wordlist:", err)
		return
	}
	defer file.Close()

	log.Println("Loading Input Wordlist")
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		input := scanner.Text()
		if len(input) <= MaxLen {
			originalDict = append(originalDict, input)
		}
	}

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

	originalDictGPUArray := make([]byte, (originalDictCount)*MaxLen)
	originalDictGPUArrayLengths := make([]uint32, originalDictCount)
	for i, word := range originalDict {
		copy(originalDictGPUArray[i*MaxLen:], word)
		originalDictGPUArrayLengths[i] = uint32(len(word))
	}

	lastBestFitness := rules[0].Fitness
	if fileExists(stateFile) {
		lastBestFitness = rules[len(rules)-1].Fitness
	}
	var bestRule *ruleObj
	var hashedWords []uint64

	saveEvery := 100
	processed := 0
	for lastBestFitness > 0 {
		// todo: find a way to reduce rules in size
		rules, bestRule, hashedWords = processRuleRound(
			rules,
			deviceCount,
			&originalDictGPUArray,
			&originalDictGPUArrayLengths,
			&originalHashes,
			&compareDictHashes,
			originalDictCount,
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

func processRuleRound(rules []ruleObj, deviceCount int, originalDictGPUArray *[]byte, originalDictGPUArrayLengths *[]uint32, originalHashes *[]uint64, compareDictHashes *[]uint64, originalDictCount int, compareDictCount int, lastFitness uint64) ([]ruleObj, *ruleObj, []uint64) {
	deviceCount = CUDAGetDeviceCount()
	ruleChan := make(chan *ruleObj, deviceCount)
	var writerMutex sync.Mutex
	var wg sync.WaitGroup
	wg.Add(deviceCount)

	processBar := progressbar.NewOptions(len(rules)*originalDictCount,
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
			defer func() {
				runtime.UnlockOSThread()
				wg.Done()
			}()
			d_originalDict, d_originalDictLengths, d_originalHashes, d_compareHashes, stream := CUDAInitialize(originalDictGPUArray, originalDictGPUArrayLengths, originalHashes, originalDictCount, compareDictHashes, compareDictCount)
			func() {
				// Initialize processed variables for this worker
				hashes := make([]uint64, originalDictCount)
				d_hashes := CUDAInitializeHashHits(&hashes, originalDictCount, stream)
				hashTable := make([]bool, originalDictCount*1)
				d_hashTable := CUDAInitializeHashTable(&hashTable, originalDictCount*1, stream)
				d_processedDict, d_processedDictLengths, d_hitCount := CUDAInitializeProcessed(originalDictCount, stream)
				defer func(d_originalDict *C.char, d_originalDictLengths *C.int, d_originalHashes *C.uint64_t, d_compareHashes *C.uint64_t, d_processedDict *C.char, d_processedDictLengths *C.int, d_hitCount *C.uint64_t, d_hashes *C.uint64_t, d_hashTable *C.bool, stream C.cudaStream_t) {
					CUDADeinitialize(d_originalDict, d_originalDictLengths, stream)
					CUDADeinitializeHashes(d_originalHashes, d_compareHashes, stream)
					CUDADeinitializeHashHits(d_hashes, stream)
					CUDADeinitializeHashTable(d_hashTable, stream)
					CUDADeinitializeProcessed(d_processedDict, d_processedDictLengths, d_hitCount, stream)
					CUDADeinitializeStream(stream)
				}(d_originalDict, d_originalDictLengths, d_originalHashes, d_compareHashes, d_processedDict, d_processedDictLengths, d_hitCount, d_hashes, d_hashTable, stream)

				// Initialize processed variables for this worker
				// Process each rule received from the channel and binary search the results against the original wordlist
				// to extract new entries that don't already exist. originalHashes lives per gpu device
				for rule := range ruleChan {
					hits := CUDASingleRuleScore(
						&rule.RuleLine,
						d_originalDict, d_originalDictLengths,
						d_processedDict, d_processedDictLengths,
						d_originalHashes, originalDictCount,
						d_compareHashes, compareDictCount,
						d_hitCount, d_hashes,
						d_hashTable,
						stream,
					)
					if len(rule.RuleLine) <= 5 { // experimental 2025-08-22 with the purpose of promoting shorter hashes in a decreasing linear algorithm
						hits *= uint64(10)
						hits /= uint64(5 + len(rule.RuleLine))
					}
					// Write results with existing mutex
					processBar.Add(originalDictCount)
					rule.LastFitness = hits
					if hits >= currentBestFitnessRule.LastFitness {
						// double validation with mutex to guarantee it's accurate
						writerMutex.Lock()
						if hits == currentBestFitnessRule.LastFitness {
							// If it's smaller in operations with the same hits, take that instead
							if len(rule.RuleLine) < len(currentBestFitnessRule.RuleLine) {
								currentBestFitnessRule = rule
								hashedWords = CUDAGetHashes(hits, d_hashes, stream)
							}
						} else if hits > currentBestFitnessRule.LastFitness {
							currentBestFitnessRule = rule
							hashedWords = CUDAGetHashes(hits, d_hashes, stream)
						}
						writerMutex.Unlock()
					}
				}
			}()
		}()
	}

	// Send all rules to the workers
	for i := range rules {
		if rules[i].LastFitness == 0 { // Must have the potential to be better
			processBar.Add(originalDictCount)
			continue
		}
		if lastFitness == 0 {
			ruleChan <- &rules[i]
			continue
		}

		if rules[i].LastFitness < 250000 && rules[i].LastFitness >= 100000 && rules[i].LastFitness < lastFitness-5000 && lastFitness >= 5000 { // if it's far off, skip it
			processBar.Add(originalDictCount)
			continue
		}
		if rules[i].LastFitness < 80000 && rules[i].LastFitness >= 50000 && rules[i].LastFitness < lastFitness-2000 && lastFitness >= 2000 { // if it's far off, skip it
			processBar.Add(originalDictCount)
			continue
		}
		if rules[i].LastFitness < 50000 && rules[i].LastFitness >= 10000 && rules[i].LastFitness < lastFitness-500 && lastFitness >= 500 { // if it's far off, skip it
			processBar.Add(originalDictCount)
			continue
		}
		if rules[i].LastFitness < 10000 && rules[i].LastFitness < lastFitness-100 && lastFitness >= 100 { // if it's far off, skip it
			processBar.Add(originalDictCount)
			continue
		}

		if rules[i].LastFitness < 3000 && rules[i].LastFitness < lastFitness-10 && lastFitness >= 10 { // if it's far off, skip it
			processBar.Add(originalDictCount)
			continue
		}
		if rules[i].LastFitness < currentBestFitnessRule.LastFitness { // Must have the potential to be better
			processBar.Add(originalDictCount)
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
