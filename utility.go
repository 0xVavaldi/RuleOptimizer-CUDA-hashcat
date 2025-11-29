package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"slices"
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

// cleanWords returns elements in targetHashes that are not in foundHashes (A-B)
// Both input slices must be sorted in ascending order
func cleanWords(targetHashes []uint64, foundHashes []uint64) ([]uint64, int) {
	targetCount := len(targetHashes)
	if targetCount == 0 {
		return nil, 0
	}

	result := make([]uint64, 0, targetCount)
	i, j := 0, 0
	originalCount := targetCount

	for i < targetCount && j < len(foundHashes) {
		if targetHashes[i] < foundHashes[j] {
			// Element not in foundHashes, keep it
			result = append(result, targetHashes[i])
			i++
		} else if targetHashes[i] == foundHashes[j] {
			// Element found in foundHashes, skip it
			i++
			j++
		} else { // targetHashes[i] > foundHashes[j]
			// Move to next found hash
			j++
		}
	}

	// Add remaining elements from targetHashes
	for i < targetCount {
		result = append(result, targetHashes[i])
		i++
	}

	removedCount := originalCount - len(result)
	return result, removedCount
}

// Output Writer
func appendScoreToFile(ruleFileName string, hits uint64, rule *ruleObj) {
	outputFile, err := os.OpenFile(ruleFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Println("Error opening or creating output file:", err)
		return
	}

	if _, err = outputFile.WriteString(strconv.FormatUint(hits, 10) + "\t" + FormatAllRules(rule.RuleLine) + "\n"); err != nil {
		log.Println("Error writing to output file:", err)
		return
	}

	err = outputFile.Close()
	if err != nil {
		return
	}
}

// Idea is to load a wordlist that's binary searchable later.
func loadWordlist(inputFile string) ([]string, []uint8, int, int) {
	defer timer("loadWordlist")()

	file, err := os.Open(inputFile)
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
		return nil, nil, 0, 0
	}
	defer file.Close()

	type WordEntry struct {
		Word   string
		Length uint8
	}

	var entries []WordEntry
	sum := 0
	count := 0

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 || len(line) > MaxLen {
			continue
		}

		length := uint8(len(line))
		sum += int(length)
		count++

		entries = append(entries, WordEntry{
			Word:   line,
			Length: length,
		})
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading file: %v", err)
		return nil, nil, 0, 0
	}

	println("Sorting", len(entries), "words")

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

	return wordlist, wordlistLengths, count, sum
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

	// empty buffer
	if len(buffer) > 0 {
		passwordQueue <- buffer
		wordlistBar.Add(len(buffer))
	}
	wordlistBar.Finish()
	wordlistBar.Close()
	// Finish enumerating wordlist

	close(passwordQueue)
	wgQueue.Wait()
	close(resultQueue)
	wgResult.Wait()

	log.Printf("Sorting %d Hashes. This can take a while.", wordlistLineCount)
	ParallelSortUint64(resultSlice)
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

// ParallelSortUint64 sorts data in-place using up to maxWorkers concurrent workers.
// If maxWorkers <= 0 it defaults to runtime.NumCPU().
func ParallelSortUint64(data []uint64) {
	n := len(data)
	if n <= 1 {
		return
	}

	maxWorkers := runtime.NumCPU()

	// Small inputs: prefer single-threaded path.
	if n <= 2048 || maxWorkers == 1 {
		slices.Sort(data)
		return
	}

	// Choose workers but avoid tiny chunks.
	workers := maxWorkers
	minChunk := 512 // smaller than for strings because uint64 moves are cheap
	if n/workers < minChunk {
		workers = n / minChunk
		if workers < 1 {
			workers = 1
		}
	}

	// Final guard
	if workers < 1 {
		workers = 1
	}

	// Compute chunkSize and explicit chunk ranges so we know exactly how many chunks we'll sort.
	chunkSize := (n + workers - 1) / workers // ceil(n/workers)

	type rng struct{ s, e int }
	var chunks []rng
	for i := 0; i < workers; i++ {
		start := i * chunkSize
		if start >= n {
			break
		}
		end := start + chunkSize
		if end > n {
			end = n
		}
		chunks = append(chunks, rng{start, end})
	}

	// Phase 1: sort each chunk in parallel.
	var sortWg sync.WaitGroup
	sortWg.Add(len(chunks))
	for _, c := range chunks {
		// capture c locally
		s, e := c.s, c.e
		go func(s, e int) {
			defer sortWg.Done()
			// limit capacity of subslice to avoid accidental retention of larger backing arrays
			sub := data[s:e:e]
			slices.Sort(sub)
		}(s, e)
	}
	sortWg.Wait()

	// If only one chunk exists, sorted already.
	if len(chunks) <= 1 {
		return
	}

	// Phase 2: iterative bottom-up merging.
	// Allocate a single temp buffer (tmpBuf) once, because merges are needed.
	tmpBuf := make([]uint64, n)

	// Use explicit names for buffers so intent is clear.
	srcBuf := data   // source for the current pass (initially original data)
	dstBuf := tmpBuf // destination for the current pass

	// Bound concurrent merges to avoid too many goroutines.
	maxMergeConcurrency := maxWorkers
	sem := make(chan struct{}, maxMergeConcurrency)

	// width is run size (starts at chunkSize)
	for width := chunkSize; width < n; width *= 2 {
		var passWg sync.WaitGroup

		for start := 0; start < n; start += 2 * width {
			mid := start + width
			if mid > n {
				mid = n
			}
			end := start + 2*width
			if end > n {
				end = n
			}

			if mid >= end {
				// No pair to merge: copy remaining run from src to dst (must preserve order).
				// Copy inline (no goroutine) for small ranges; cheap for uint64 slices.
				copy(dstBuf[start:end], srcBuf[start:end])
				continue
			}

			passWg.Add(1)
			sem <- struct{}{} // acquire slot
			// launch merge worker for [start:mid) and [mid:end) -> dst[start:end)
			go func(s, m, e int) {
				defer passWg.Done()
				mergeIntoUint64(dstBuf, srcBuf, s, m, e)
				<-sem // release slot
			}(start, mid, end)
		}

		passWg.Wait()
		// swap buffers for next pass
		srcBuf, dstBuf = dstBuf, srcBuf
	}

	// If final sorted data resides in tmpBuf (i.e., srcBuf != data), copy it back once.
	// Note: this copy is necessary only if we performed an odd number of passes.
	copy(data, srcBuf)
}

// mergeIntoUint64 merges src[start:mid] and src[mid:end] into dst[start:end].
func mergeIntoUint64(dst, src []uint64, start, mid, end int) {
	i, j, k := start, mid, start
	for i < mid && j < end {
		if src[i] <= src[j] {
			dst[k] = src[i]
			i++
		} else {
			dst[k] = src[j]
			j++
		}
		k++
	}
	for i < mid {
		dst[k] = src[i]
		i++
		k++
	}
	for j < end {
		dst[k] = src[j]
		j++
		k++
	}
}
