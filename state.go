package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"strings"
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
