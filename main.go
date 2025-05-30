package main

// compile code: nvcc -shared -Xcompiler -o librules.so rules.cu
// nvcc -shared -o librules.dll rules.cu
// nvcc -shared -o libxxhash.dll xxhash.cu

/*
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"
import (
	"bufio"
	"fmt"
	"github.com/alecthomas/kong"
	"github.com/schollz/progressbar/v3"
	"log"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
)

type CLI struct {
	Wordlist   string `arg:"" help:"Path to wordlist file (must fit in memory)"`
	Target     string `arg:"" help:"Path to target data file (must fit in memory)"`
	ScoreFile  string `short:"s" help:"Aggregated score file TSV."`
	RuleFile   string `short:"r" help:"Rule file to analyse."`
	OutputFile string `short:"o" help:"Score File to output results to."`
}

type ruleObj struct {
	ID           uint64
	Fitness      uint64
	LastFitness  uint64
	RuleLine     []Rule
	PreProcessed bool
	Hits         map[uint64]struct{}
	HitsMutex    sync.Mutex
}

type lineObj struct {
	ID   uint64
	line string
}

// Constants
const MaxLen = 32

func RemoveElements(original, toRemove []uint64) []uint64 {
	// Create a map for faster lookups
	removeMap := make(map[uint64]struct{})
	for _, v := range toRemove {
		removeMap[v] = struct{}{}
	}

	// Filter the original slice
	result := make([]uint64, 0, len(original))
	for _, v := range original {
		if _, exists := removeMap[v]; !exists {
			result = append(result, v)
		}
	}
	return result
}

func CountUnique(items []uint64) uint64 {
	unique := make(map[uint64]struct{})
	for _, item := range items {
		unique[item] = struct{}{}
	}
	return uint64(len(unique))
}

func appendScoreToFile(ruleFileName string, hits uint64, rule *ruleObj) {
	outputFile, err := os.OpenFile(ruleFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening or creating file:", err)
		return
	}

	if _, err = outputFile.WriteString(strconv.FormatUint(hits, 10) + "\t" + FormatAllRules(rule.RuleLine) + "\n"); err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	err = outputFile.Close()
	if err != nil {
		return
	}
}

func processRuleRound(
	rules []ruleObj, deviceCount int, originalDictGPUArray *[]byte, originalDictGPUArrayLengths *[]uint32, originalHashes *[]uint64, compareDictHashes *[]uint64, originalDictCount int, compareDictCount int) ([]ruleObj, *ruleObj) {
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
				hashTable := make([]bool, originalDictCount*100)
				d_hashTable := CUDAInitializeHashTable(&hashTable, originalDictCount*100, stream)
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
					// Write results with existing mutex
					processBar.Add(originalDictCount)
					rule.LastFitness = hits
					if hits >= currentBestFitnessRule.LastFitness {
						// double validation with mutex to guarantee it's accurate
						writerMutex.Lock()
						if hits == currentBestFitnessRule.LastFitness {
							// If it's smaller in operations with the same hits, take that instead
							if len(currentBestFitnessRule.RuleLine) > len(rule.RuleLine) {
								currentBestFitnessRule = rule
							}
						} else if hits > currentBestFitnessRule.LastFitness {
							currentBestFitnessRule = rule
						}
						writerMutex.Unlock()
					}
				}
			}()
		}()
	}

	// Send all rules to the workers
	for i := range rules {
		//if rules[i].LastFitness < lastBestFitness/2 { // if it's far off, skip it
		//	if rules[i].LastFitness > 2 {
		//		processBar.Add(originalDictCount)
		//		continue
		//	}
		//}
		if rules[i].LastFitness == 0 { // cannot get more cracks
			processBar.Add(originalDictCount)
			continue
		}
		if rules[i].LastFitness < currentBestFitnessRule.LastFitness { // Must have the potential to be better
			processBar.Add(originalDictCount)
			continue
		}
		if rules[i].Fitness < currentBestFitnessRule.LastFitness { // Must have the potential to be better
			break
		}
		ruleChan <- &rules[i]
	}
	close(ruleChan)
	wg.Wait()
	processBar.Close()
	processBar.Finish()
	return rules, currentBestFitnessRule
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
		outputFile, err := os.OpenFile(cli.OutputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
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
				hashTable := make([]bool, originalDictCount*100)
				d_hashTable := CUDAInitializeHashTable(&hashTable, originalDictCount*100, stream)
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

func main() {
	var cli CLI
	kong.Parse(&cli,
		kong.Name("CudaRuleScoreOptimizer"),
		kong.Description("An application that optimizes rule scores with set optimization theory based on performance."),
		kong.UsageOnError(),
	)

	if cli.OutputFile == "" {
		log.Println("Output file is not specified, quitting.")
		os.Exit(-1)
	}

	if _, err := os.Stat(cli.OutputFile); err == nil {
		log.Println("Output file exists, quitting.")
		os.Exit(-1)
	}

	if cli.ScoreFile != "" {
		generatePhase2(cli)
	}
	if cli.Wordlist != "" && cli.Target != "" && cli.RuleFile != "" {
		generatePhase1(cli)
	}
}

func generatePhase1(cli CLI) {
	originalDictName := cli.Wordlist
	originalHashes := loadHashedWordlist(originalDictName)
	originalDictCount := len(originalHashes)
	originalDict := make([]string, 0, originalDictCount)

	compareDictName := cli.Target
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
	rules := loadRulesFast(cli.RuleFile)

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

func generatePhase2(cli CLI) {
	originalDictName := cli.Wordlist
	originalDictCount, _ := lineCounter(originalDictName)
	originalDict := make([]string, 0, originalDictCount)

	originalHashes := loadHashedWordlist(originalDictName)
	compareDictName := cli.Target
	compareDictCount, _ := lineCounter(compareDictName)
	compareDictHashes := loadHashedWordlist(compareDictName)

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

	log.Println("Loading Rule Scores")
	rules := loadRuleScores(cli.ScoreFile)
	log.Printf("Loaded %d Rule Scores", len(rules))

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

	lastBestFitness := rules[0].Fitness + 1000
	var bestRule *ruleObj
	for lastBestFitness > 0 {
		rules, bestRule = processRuleRound(
			rules,
			deviceCount,
			&originalDictGPUArray,
			&originalDictGPUArrayLengths,
			&originalHashes,
			&compareDictHashes,
			originalDictCount,
			compareDictCount,
		)
		log.Printf("New candidate %d with score: %d", bestRule.ID, bestRule.LastFitness)
		lastBestFitness = bestRule.LastFitness
		rules[bestRule.ID-1].LastFitness = 0 // ID starts at 1

		func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			CUDASetDevice(0)
			d_originalDict, d_originalDictLengths, d_originalHashes, d_compareHashes, stream := CUDAInitialize(&originalDictGPUArray, &originalDictGPUArrayLengths, &originalHashes, originalDictCount, &compareDictHashes, compareDictCount)
			func() {
				// Initialize processed variables for this worker
				d_processedDict, d_processedDictLengths, d_hitCount := CUDAInitializeProcessed(originalDictCount, stream)

				hashes := make([]uint64, originalDictCount)
				d_hashes := CUDAInitializeHashHits(&hashes, originalDictCount, stream)
				hashTable := make([]bool, originalDictCount*100)
				d_hashTable := CUDAInitializeHashTable(&hashTable, originalDictCount*100, stream)
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
				hashedWords := CUDASingleRuleHashed(
					&bestRule.RuleLine,
					d_originalDict, d_originalDictLengths,
					d_processedDict, d_processedDictLengths,
					d_originalHashes, originalDictCount,
					d_compareHashes, compareDictCount,
					d_hitCount, d_hashes, d_hashTable,
					stream,
				)
				print("Cleaning words...")
				compareDictHashes = RemoveElements(compareDictHashes, hashedWords)
				compareDictCountNew := len(compareDictHashes)
				log.Printf("%d new words found.", compareDictCount-compareDictCountNew)
				appendScoreToFile(cli.OutputFile, uint64(compareDictCount-compareDictCountNew), bestRule)
				compareDictCount = compareDictCountNew
			}()
		}()
	}
}
