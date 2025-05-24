package main

// compile code: nvcc -shared -Xcompiler -o librules.so rules.cu
// nvcc -shared -o librules.dll rules.cu
// nvcc -shared -o libxxhash.dll xxhash.cu

/*
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"
import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/alecthomas/kong"
	"github.com/cespare/xxhash/v2"
	"github.com/schollz/progressbar/v3"
	"io"
	"log"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type CLI struct {
	Wordlist  string `arg:"" help:"Path to wordlist file (must fit in memory)"`
	Target    string `arg:"" help:"Path to target data file (must fit in memory)"`
	Rules     string `short:"r" help:"Rules to target"`
	ScoreFile string `short:"o" help:"Score File to output results to"`
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

func ProcessRules(ruleObject *ruleObj, ruleHits *map[uint64]struct{}, originalDict *map[string]struct{}, compareDict *map[uint64]struct{}) uint64 {
	numWorkers := runtime.NumCPU()

	var wg sync.WaitGroup
	var newFitness atomic.Uint64
	newFitness.Store(0)
	passwordChan := make(chan []string, numWorkers)
	workerInternal := func() {
		defer wg.Done()
		localHits := make(map[uint64]struct{})

		for passwordBuffer := range passwordChan {
			for _, passwordCopy := range passwordBuffer {
				passwordCopyCopy := passwordCopy
				for _, oneRule := range ruleObject.RuleLine {
					passwordCopyCopy = oneRule.Process(passwordCopyCopy)
				}
				if passwordCopy == passwordCopyCopy {
					continue
				}
				if _, exists1 := (*originalDict)[passwordCopyCopy]; exists1 { // do not save if it would be covered under the `:` rule. this check requires ~30GB of RAM
					continue
				}
				hash := xxhash.Sum64String(passwordCopyCopy)
				if _, exists1 := (*compareDict)[hash]; exists1 {
					if _, exists2 := (*ruleHits)[hash]; !exists2 { // rulefile.Hits
						localHits[hash] = struct{}{}
					}
				}
			}
		}

		// Save which xxhashes are new hits for future iterations for each rule line
		ruleObject.HitsMutex.Lock()
		localFitness := uint64(0)
		for hash := range localHits {
			if _, exists := ruleObject.Hits[hash]; !exists {
				ruleObject.Hits[hash] = struct{}{}
				localFitness++
			}
		}
		ruleObject.HitsMutex.Unlock()

		newFitness.Add(localFitness)
	}

	// Start workers
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go workerInternal()
	}

	pwBuffer := make([]string, 0, 100000)
	for password := range *originalDict {
		pwBuffer = append(pwBuffer, password)
		if len(pwBuffer)%100000 == 0 {
			passwordChan <- pwBuffer
			pwBuffer = make([]string, 0, 100000)
		}
	}
	if len(pwBuffer) > 0 {
		passwordChan <- pwBuffer
	}
	close(passwordChan)
	wg.Wait()
	ruleObject.HitsMutex.Lock()
	ruleObject.LastFitness = newFitness.Load()
	ruleObject.PreProcessed = true
	ruleObject.HitsMutex.Unlock()

	return ruleObject.LastFitness
}

func timer(name string) func() {
	start := time.Now()
	return func() {
		fmt.Printf("%s took %v\n", name, time.Since(start))
	}
}

func parseUint64(s string) uint64 {
	var result uint64
	fmt.Sscanf(s, "%d", &result)
	return result
}

func lineCounter(inputFile string) (int, error) {
	file, err := os.Open(inputFile)
	if err != nil {
		return 0, err
	}

	defer file.Close()
	buf := make([]byte, 32*1024)
	count := 0
	lineSep := []byte{'\n'}

	for {
		c, err := file.Read(buf)
		count += bytes.Count(buf[:c], lineSep)
		switch {
		case err == io.EOF:
			return count, nil
		case err != nil:
			return count, err
		}
	}
}

func loadRulesFast(inputFile string) []ruleObj {
	defer timer("loadRules")()
	ruleLines, _ := lineCounter(inputFile)
	ruleQueue := make(chan lineObj, 100)
	ruleOutput := make(chan ruleObj, ruleLines)
	threadCount := runtime.NumCPU()
	wg := sync.WaitGroup{}

	for i := 0; i < threadCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rawLineObj := range ruleQueue {
				ruleObject, _ := ConvertFromHashcat(rawLineObj.ID, rawLineObj.line)
				hits := make(map[uint64]struct{})
				ruleOutput <- ruleObj{rawLineObj.ID, 0, 0, ruleObject, false, hits, sync.Mutex{}}
			}
		}()
	}

	ruleBar := progressbar.NewOptions(ruleLines,
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowDescriptionAtLineEnd(),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionThrottle(500*time.Millisecond),
		progressbar.OptionShowElapsedTimeOnFinish(),
		progressbar.OptionSetWidth(25),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
	)
	file, err := os.Open(inputFile)
	if err != nil {
		log.Println("Error opening file:", err)
		return []ruleObj{}
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	ruleLineCounter := uint64(1)

	for scanner.Scan() {
		lineObject := new(lineObj)
		lineObject.ID = ruleLineCounter
		lineObject.line = scanner.Text()
		if len(lineObject.line) == 0 {
			continue
		}
		ruleQueue <- *lineObject
		ruleLineCounter++
		if ruleLineCounter%10000 == 0 {
			ruleBar.Add(10000)
		}
	}
	ruleBar.Add(int(ruleLineCounter))
	close(ruleQueue)
	go func() {
		wg.Wait()
		close(ruleOutput)
	}()

	// Step 1: Consume the channel into a slice
	var sortedRules []ruleObj
	for obj := range ruleOutput {
		sortedRules = append(sortedRules, obj)
	}

	// Step 2: Sort the slice by ID
	sort.Slice(sortedRules, func(i, j int) bool {
		return sortedRules[i].ID < sortedRules[j].ID
	})
	ruleBar.Finish()
	ruleBar.Close()
	println()
	return sortedRules
}

func loadHashedWordlist(inputFile string) []uint64 {
	defer timer("loadHashedWordlist")()
	wordlistLineCount, _ := lineCounter(inputFile)
	passwordQueue := make(chan []string, 100)
	resultQueue := make(chan []uint64, 100)
	threadCount := runtime.NumCPU()
	wg := sync.WaitGroup{}
	wgg := sync.WaitGroup{}

	for i := 0; i < threadCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for bufferQueue := range passwordQueue {
				var bufferResults []uint64
				for _, rawLine := range bufferQueue {
					bufferResults = append(bufferResults, xxhash.Sum64String(rawLine))
				}
				resultQueue <- bufferResults
			}
		}()
	}

	wordlistBar := progressbar.NewOptions(wordlistLineCount,
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowDescriptionAtLineEnd(),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionThrottle(500*time.Millisecond),
		progressbar.OptionShowElapsedTimeOnFinish(),
		progressbar.OptionSetWidth(25),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
	)
	file, err := os.Open(inputFile)
	if err != nil {
		log.Println("Error opening file:", err)
		return []uint64{}
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	ruleLineCounter := uint64(1)

	var resultSlice []uint64
	wgg.Add(1)
	go func() {
		defer wgg.Done()
		for obj := range resultQueue {
			resultSlice = append(resultSlice, obj...)
		}
	}()

	var buffer []string
	bufferSize := 10000
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}
		buffer = append(buffer, line)
		ruleLineCounter++
		if len(buffer) >= bufferSize {
			passwordQueue <- buffer
			buffer = make([]string, 0, bufferSize)
			wordlistBar.Add(bufferSize)
		}
	}
	if len(buffer) > 0 {
		passwordQueue <- buffer
		wordlistBar.Add(len(buffer))
	}
	wordlistBar.Finish()
	wordlistBar.Close()
	println()

	println("Finalizing Preparation")
	close(passwordQueue)
	wg.Wait()

	close(resultQueue)
	wgg.Wait()
	sort.Slice(resultSlice, func(i, j int) bool { return resultSlice[i] < resultSlice[j] })
	return resultSlice
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

func main() {
	var cli CLI
	kong.Parse(&cli,
		kong.Name("CudaRuleScorer"),
		kong.Description("An application that scores rules based on performance"),
		kong.UsageOnError(),
	)

	if _, err := os.Stat(cli.Rules + ".score"); err == nil {
		log.Println("Rules have already been scored, skipping.")
		os.Exit(-1)
	}

	//var wg sync.WaitGroup
	originalDictName := cli.Wordlist
	originalDictCount, _ := lineCounter(originalDictName)
	originalDict := make([]string, 0, originalDictCount)
	originalDictMap := make(map[string]struct{}, originalDictCount)

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
		originalDict = append(originalDict, scanner.Text())
		originalDictMap[scanner.Text()] = struct{}{}
	}

	log.Println("Loading Rules")
	ruleFileName := cli.Rules
	rules := loadRulesFast(ruleFileName)

	deviceCount := CUDAGetDeviceCount()
	var wg sync.WaitGroup
	wg.Add(deviceCount)
	log.Printf("Detected %d GPU devices. Initializing", deviceCount)

	// Start processing
	// Start processing
	// Start processing

	// Compute Rule
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

	originalDictGPUArray := make([]byte, (originalDictCount)*MaxLen)
	originalDictGPUArrayLengths := make([]uint32, originalDictCount)
	for i, word := range originalDict {
		copy(originalDictGPUArray[i*MaxLen:], word)
		originalDictGPUArrayLengths[i] = uint32(len(word))
	}

	var writerMutex sync.Mutex
	ruleChan := make(chan *ruleObj)

	for i := 0; i < deviceCount; i++ {
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			CUDASetDevice(i)
			d_originalDict, d_originalDictLengths, d_originalHashes, d_compareHashes, stream := CUDAInitialize(&originalDictGPUArray, &originalDictGPUArrayLengths, &originalHashes, originalDictCount, &compareDictHashes, compareDictCount, deviceCount)
			func() {
				defer wg.Done()
				// Initialize processed variables for this worker
				d_processedDict, d_processedDictLengths, d_hitCount := CUDAInitializeProcessed(originalDictCount, stream)
				defer func(deviceCount int, d_originalDict *C.char, d_originalDictLengths *C.int, stream C.cudaStream_t) {
					CUDADeinitialize(d_originalDict, d_originalDictLengths, stream)
				}(deviceCount, d_originalDict, d_originalDictLengths, stream)

				defer CUDADeinitializeProcessed(d_processedDict, d_processedDictLengths, d_hitCount, stream)
				// Initialize processed variables for this worker
				// Process each rule received from the channel and binary search the results against the original wordlist
				// to extract new entries that don't already exist. originalHashes lives per gpu device
				for rule := range ruleChan {
					hits := CUDASingleRule(
						&rule.RuleLine,
						d_originalDict, d_originalDictLengths,
						d_processedDict, d_processedDictLengths,
						d_originalHashes, originalDictCount,
						d_compareHashes, compareDictCount,
						d_hitCount,
						stream,
					)
					// Write results with existing mutex
					writerMutex.Lock()
					processBar.Add(originalDictCount)
					appendScoreToFile(ruleFileName+".score", hits, rule)
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
	log.Println("Cleaning up GPU Memory")
}
