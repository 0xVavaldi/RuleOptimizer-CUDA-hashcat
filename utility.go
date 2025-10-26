package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/schollz/progressbar/v3"
)

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

// Removing sortedArray (A) from targets (B). Return A-B
func cleanWords(sortedArray *[]uint64, targets *[]uint64) []uint64 {
	// Use two pointers technique for O(n+m) complexity
	i, j := 0, 0 // i for sortedArray, j for targets
	k := 0       // position to write in sortedArray

	for i < len(*sortedArray) && j < len(*targets) {
		if (*sortedArray)[i] < (*targets)[j] {
			// Keep this element
			(*sortedArray)[k] = (*sortedArray)[i]
			k++
			i++
		} else if (*sortedArray)[i] == (*targets)[j] {
			// Skip this element
			i++
			j++
		} else { // sortedArray[i] > targets[j]
			// Move to next element
			j++
		}
	}

	// Copy remaining elements
	for i < len(*sortedArray) {
		(*sortedArray)[k] = (*sortedArray)[i]
		k++
		i++
	}

	return (*sortedArray)[:k]
}

// Output Writer
func appendScoreToFile(ruleFileName string, hits uint64, rule *ruleObj) {
	outputFile, err := os.OpenFile(ruleFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening or creating output file:", err)
		return
	}

	if _, err = outputFile.WriteString(strconv.FormatUint(hits, 10) + "\t" + FormatAllRules(rule.RuleLine) + "\n"); err != nil {
		fmt.Println("Error writing to file output file:", err)
		return
	}

	err = outputFile.Close()
	if err != nil {
		return
	}
}

// Idea is to load a wordlist that's binary searchable later.
func loadWordlist(inputFile string) ([]string, []uint8, int) {
	defer timer("loadWordlist")()

	file, err := os.Open(inputFile)
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
		return nil, nil, 0
	}
	defer file.Close()

	type WordEntry struct {
		Word   string
		Length uint8
	}

	var entries []WordEntry
	sum := 0

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 || len(line) > MaxLen {
			continue
		}

		length := uint8(len(line))
		sum += int(length)

		entries = append(entries, WordEntry{
			Word:   line,
			Length: length,
		})
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading file: %v", err)
		return nil, nil, 0
	}

	println("Sorting", len(entries), "words")

	// Sort using the proper comparison logic
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i].Word, entries[j].Word
		minLen := len(a)
		if len(b) < minLen {
			minLen = len(b)
		}

		// Compare content first
		for k := 0; k < minLen; k++ {
			if a[k] != b[k] {
				return a[k] < b[k]
			}
		}

		// If content equal, shorter string comes first
		return len(a) < len(b)
	})

	// Extract to separate slices
	wordlist := make([]string, len(entries))
	wordlistLengths := make([]uint8, len(entries))

	for i, entry := range entries {
		wordlist[i] = entry.Word
		wordlistLengths[i] = entry.Length
	}

	return wordlist, wordlistLengths, sum
}

// Load a wordlists, hash it with xxhash64, and return an array of sorted uint64's.
func loadHashedWordlist(inputFile string) []uint64 {
	defer timer("loadHashedWordlist")()
	wordlistLineCount, _ := lineCounter(inputFile)
	passwordQueue := make(chan []string, 100)
	resultQueue := make(chan []uint64, 100)
	threadCount := runtime.NumCPU()
	wgQueue := sync.WaitGroup{}
	wgResult := sync.WaitGroup{}

	// Hash Wordlist
	for i := 0; i < threadCount; i++ {
		wgQueue.Add(1)
		go func() {
			defer wgQueue.Done()
			for bufferQueue := range passwordQueue {
				var bufferResults []uint64
				for _, rawLine := range bufferQueue {
					if len(rawLine) <= MaxLen {
						bufferResults = append(bufferResults, xxhash.Sum64String(rawLine))
					}
				}
				resultQueue <- bufferResults
			}
		}()
	}
	// Finish Hashing

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
		log.Fatal("Error opening file:", err)
		return []uint64{}
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	var resultSlice []uint64
	wgResult.Add(1)
	go func() {
		defer wgResult.Done()
		for obj := range resultQueue {
			resultSlice = append(resultSlice, obj...)
		}
	}()

	// Enum wordlist
	var buffer []string
	bufferSize := 10000
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}
		buffer = append(buffer, line)
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
	// Finish enumerating wordlist

	close(passwordQueue)
	wgQueue.Wait()
	close(resultQueue)
	wgResult.Wait()

	log.Printf("Sorting %d Hashes. This can take a while.", wordlistLineCount)
	sort.Slice(resultSlice, func(i, j int) bool { return resultSlice[i] < resultSlice[j] })
	return resultSlice
}

func removeAfromB(A, B []string) []string {
	// Create a set of elements in A for fast lookup
	setA := make(map[string]bool)
	for _, item := range A {
		setA[item] = true
	}

	// Filter B to exclude elements present in A
	result := make([]string, 0)
	for _, item := range B {
		if !setA[item] {
			result = append(result, item)
		}
	}

	return result
}

// Convert 10000000 to 10MB formatting
func ByteCountSI(b int) string {
	const unit = 1000
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB",
		float64(b)/float64(div), "kMGTPE"[exp])
}
