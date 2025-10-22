package main

import (
	"bufio"
	"log"
	"os"
	"runtime"
)

func generateSimulate(cli CLI) {
	originalDictName := cli.Simulate.Wordlist
	originalHashes := loadHashedWordlist(originalDictName)
	originalDictCount := len(originalHashes)
	originalDict := make([]string, 0, originalDictCount)

	compareDictName := cli.Simulate.Target
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
	rules := loadRulesFast(cli.Simulate.RuleFile)
	log.Printf("Loaded %d Rules", len(rules))

	deviceID := 0
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	CUDASetDevice(deviceID)

	originalDictGPUArray := make([]byte, originalDictCount*MaxLen)
	originalDictGPUArrayLengths := make([]uint32, originalDictCount)
	for i, word := range originalDict {
		copy(originalDictGPUArray[i*MaxLen:], word)
		originalDictGPUArrayLengths[i] = uint32(len(word))
	}

	// Initialize GPU memory once
	d_originalDict, d_originalDictLengths, d_originalHashes, d_compareHashes, stream := CUDAInitialize(&originalDictGPUArray, &originalDictGPUArrayLengths, &originalHashes, originalDictCount, &compareDictHashes, compareDictCount)

	// Initialize processing buffers once
	hashes := make([]uint64, originalDictCount)
	d_hashes := CUDAInitializeHashHits(&hashes, originalDictCount, stream)
	hashTable := make([]bool, originalDictCount)
	d_hashTable := CUDAInitializeHashTable(&hashTable, originalDictCount, stream)
	d_processedDict, d_processedDictLengths, d_hitCount := CUDAInitializeProcessed(originalDictCount, stream)

	// Defer cleanup
	defer func() {
		CUDADeinitialize(d_originalDict, d_originalDictLengths, stream)
		CUDADeinitializeHashes(d_originalHashes, d_compareHashes, stream)
		CUDADeinitializeHashHits(d_hashes, stream)
		CUDADeinitializeHashTable(d_hashTable, stream)
		CUDADeinitializeProcessed(d_processedDict, d_processedDictLengths, d_hitCount, stream)
		CUDADeinitializeStream(stream)
	}()

	log.Println("Starting simulation mode - processing rules in order")

	// Track total progress
	totalHits := uint64(0)
	processedRules := 0

	// Process each rule sequentially without optimization
	for i := range rules {
		if compareDictCount == 0 {
			log.Println("All target words found, stopping simulation")
			break
		}

		log.Printf("Processing rule %d/%d (remaining targets: %d)", i+1, len(rules), compareDictCount)

		// Process single rule with existing GPU resources
		rule := &rules[i]
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

		if hits > 0 {
			// Get the found hashes from GPU
			hashedWords := CUDAGetHashes(hits, d_hashes, stream)

			// Remove found words from target and calculate hits
			beforeCount := compareDictCount
			compareDictHashes = cleanWords(&compareDictHashes, &hashedWords)
			compareDictCount = len(compareDictHashes)
			ruleHits := beforeCount - compareDictCount

			totalHits += uint64(ruleHits)
			log.Printf("Rule %d found %d new words (total: %d)", i+1, ruleHits, totalHits)
			appendScoreToFile(cli.Simulate.OutputFile, uint64(ruleHits), rule)
		} else {
			appendScoreToFile(cli.Simulate.OutputFile, uint64(0), rule)
		}

		processedRules++

		// Progress update every 100 rules
		if processedRules%100 == 0 {
			log.Printf("Progress: %d/%d rules processed, %d total hits, %d targets remaining",
				processedRules, len(rules), totalHits, compareDictCount)
		}
	}

	log.Printf("Simulation completed: %d rules processed, %d total hits, %d targets remaining",
		processedRules, totalHits, compareDictCount)
}
