package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
)

func saveState(filename string, filenameHashes string, rules *[]ruleObj, compareDictHashes []uint64) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating state file: %v", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	for _, rule := range *rules {
		// Convert rule line back to hashcat format
		ruleLine := FormatAllRules(rule.RuleLine)
		if strings.TrimSpace(ruleLine) == "" {
			continue // Skip empty rules
		}

		// Format: Fitness\tRuleLine[tsv] (same as score file)
		_, err = fmt.Fprintf(writer, "%d\t%d\t%s\n", rule.Fitness, rule.LastFitness, ruleLine)
		if err != nil {
			return fmt.Errorf("error writing to state file: %v", err)
		}
	}

	// Save CompareDict Hashes to separate file
	fileHashes, err := os.Create(filenameHashes)
	if err != nil {
		return fmt.Errorf("error creating compare dict state file: %v", err)
	}
	defer fileHashes.Close()

	writerHashes := bufio.NewWriter(fileHashes)
	defer writerHashes.Flush()

	// Use uint64 for count to ensure fixed size
	count := uint64(len(compareDictHashes))
	if err := binary.Write(writerHashes, binary.LittleEndian, count); err != nil {
		return fmt.Errorf("error writing hash count: %v", err)
	}

	// Write all the hashes
	for _, hash := range compareDictHashes {
		if err := binary.Write(writerHashes, binary.LittleEndian, hash); err != nil {
			return fmt.Errorf("error writing hash: %v", err)
		}
	}
	return nil
}

func loadCompareDictState(filenameHashes string) ([]uint64, error) {
	file, err := os.Open(filenameHashes)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Use binary.Read directly on the file, not through bufio.Reader
	// Binary operations work better directly on the file
	var count uint64
	if err := binary.Read(file, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("error reading hash count: %v", err)
	}

	// Read all the hashes
	hashes := make([]uint64, 0, count)
	for i := uint64(0); i < count; i++ {
		var hash uint64
		if err := binary.Read(file, binary.LittleEndian, &hash); err != nil {
			return hashes, fmt.Errorf("error reading hash at position %d: %v", i, err)
		}
		hashes = append(hashes, hash)
	}

	return hashes, nil
}

func loadStateScores(inputFile string) []ruleObj {
	defer timer("loadStateScores")()
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
				fitness := parseUint64(strings.SplitN(rawLineObj.line, "\t", 3)[0])
				lastFitness := parseUint64(strings.SplitN(rawLineObj.line, "\t", 3)[1])
				rawLineObj.line = strings.SplitN(rawLineObj.line, "\t", 3)[2]
				rawLineObj.line = strings.ReplaceAll(rawLineObj.line, "\t", " ")

				ruleObject, _ := ConvertFromHashcat(rawLineObj.ID, rawLineObj.line)
				hits := make(map[uint64]struct{})
				ruleOutput <- ruleObj{rawLineObj.ID, fitness, lastFitness, ruleObject, false, hits, sync.Mutex{}}
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
