package main

import "C"
import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
)

func generatePhase1(cli CLI) {
	originalDictName := cli.Score.Wordlist
	originalHashes := loadHashedWordlist(originalDictName)
	originalDictCount := len(originalHashes)
	originalDict := make([]string, 0, originalDictCount)

	compareDictName := cli.Score.Target
	compareDictHashes := loadHashedWordlist(compareDictName)
	compareDictCount := len(compareDictHashes)

	file, err := os.Open(originalDictName)
	if err != nil {
		log.Println("Error opening file:", err)
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
	originalDictCount = len(originalDict)

	log.Println("Loading Rules")
	rules := loadRulesFast(cli.Score.RuleFile)

	deviceCount := CUDAGetDeviceCount()
	log.Printf("Detected %d GPU devices. Initializing", deviceCount)

	// Start processing
	// Start processing
	// Start processing

	// Compute Rule
	originalDictGPUArray := make([]byte, (originalDictCount)*MaxLen)
	originalDictGPUArrayLengths := make([]uint32, originalDictCount)
	for i, word := range originalDict {
		copy(originalDictGPUArray[i*MaxLen:], word)
		originalDictGPUArrayLengths[i] = uint32(len(word))
	}

	processRuleFile(
		cli,
		rules,
		deviceCount,
		&originalDictGPUArray,
		&originalDictGPUArrayLengths,
		&originalHashes,
		&compareDictHashes,
		originalDictCount,
		compareDictCount,
	)
}

func processRuleFile(
	cli CLI, rules []ruleObj, deviceCount int, originalDictGPUArray *[]byte, originalDictGPUArrayLengths *[]uint32, originalHashes *[]uint64, compareDictHashes *[]uint64, originalDictCount int, compareDictCount int) {
	deviceCount = CUDAGetDeviceCount()
	ruleChan := make(chan *ruleObj, 2)
	var writerMutex sync.Mutex
	var wg sync.WaitGroup
	var wgg sync.WaitGroup
	wg.Add(deviceCount)
	wgg.Add(1)

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

	type result struct {
		hits uint64
		rule *ruleObj
	}
	resultChan := make(chan result) // Buffer size can be adjusted

	go func() {
		defer wgg.Done()
		outputFile, err := os.OpenFile(cli.Simulate.OutputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
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
					if len(rule.RuleLine) <= 5 { // experimental 2025-08-22
						hits *= uint64(10)
						hits /= uint64(5 + len(rule.RuleLine))
					}
					//hits := CountUnique(hashedWords)
					// Write results with existing mutex
					writerMutex.Lock()
					processBar.Add(originalDictCount)
					resultChan <- result{hits: hits, rule: rule}
					//appendScoreToFile(cli.OutputFile, hits, rule)
					writerMutex.Unlock()
				}
			}()
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
