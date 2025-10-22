package main

import (
	"fmt"
	"os"
	"strconv"
)

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

// Removing sortedArray from targets
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
			// Move to next target
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
