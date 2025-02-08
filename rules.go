package main

/*
#cgo LDFLAGS: -L. -lrules
#include <stdlib.h>
#include <stdint.h>

// No Param
// l
void applyLowerCase(char *words, int *lengths, int numWords);
// u
void applyUpperCase(char *words, int *lengths, int numWords);
// c
void applyCapitalize(char *words, int *lengths, int numWords);
// C
void applyInvertCapitalize(char *words, int *lengths, int numWords);
// t
void applyToggleCase(char *words, int *lengths, int numWords);
// q
void applyDuplicateChars(char *words, int *lengths, int numWords);
// r
void applyReverse(char *words, int *lengths, int numWords);
// k
void applySwapFirstTwo(char *words, int *lengths, int numWords);
// K
void applySwapLastTwo(char *words, int *lengths, int numWords);
// d
void applyDuplicate(char *words, int *lengths, int numWords);
// f
void applyReflect(char *words, int *lengths, int numWords);
// {
void applyRotateLeft(char *words, int *lengths, int numWords);
// }
void applyRotateRight(char *words, int *lengths, int numWords);
// [
void applyDeleteFirst(char *words, int *lengths, int numWords);
// ]
void applyDeleteLast(char *words, int *lengths, int numWords);
// E
void applyTitleCase(char *words, int *lengths, int numWords);
// T
void applyTogglePosition(char *words, int *lengths, int pos, int numWords);
// p
void applyRepeatWord(char *words, int *lengths, int count, int numWords);
// D
void applyDeletePosition(char *words, int *lengths, int pos, int numWords);
// z
void applyPrependFirstChar(char *words, int *lengths, int count, int numWords);
// Z
void applyAppendLastChar(char *words, int *lengths, int count, int numWords);
// '
void applyTruncateAt(char *words, int *lengths, int pos, int numWords);
// s
void applySubstitution(char *words, int *lengths, char oldChar, char newChar, int numWords);
// S
void applyReplaceFirst(char *words, int *lengths, char oldChar, char newChar, int numWords);
// $
void applyAppend(char *words, int *lengths, char appendChar, int numWords);
// ^
void applyPrepend(char *words, int *lengths, char prefixChar, int numWords);
// y
void applyPrependPrefixSubstr(char *words, int *lengths, int count, int numWords);
// Y
void applyAppendSuffixSubstr(char *words, int *lengths, int count, int numWords);
// L
void applyBitShiftLeft(char *words, int *lengths, int pos, int numWords);
// R
void applyBitShiftRight(char *words, int *lengths, int pos, int numWords);
// -
void applyDecrementChar(char *words, int *lengths, int pos, int numWords);
// +
void applyIncrementChar(char *words, int *lengths, int pos, int numWords);
// @
void applyDeleteAllChar(char *words, int *lengths, char target, int numWords);
// .
void applySwapNext(char *words, int *lengths, int pos, int numWords);
// ,

// e

// i

// O

// o

// *

// x

// <

// >

// _

// !

// /

// 3

void computeXXHashes(char* d_words, int* d_lengths, uint64_t seed, uint64_t* d_hashes, int numWords);
void allocateMemoryOnGPU(char **d_words, int **d_lengths, uint64_t **d_hashes, char *h_words, int *h_lengths, uint64_t *h_hashes, int numWords);
void copyMemoryBackToHost(char *h_words, int *h_lengths, uint64_t *h_hashes, char *d_words, int *d_lengths, uint64_t *d_hashes, int numWords);
void freeMemoryOnGPU(char *d_words, int *d_lengths, uint64_t *d_hashes);

//void allocateXXHashMemoryOnGPU(char **d_words, int **d_lengths, uint64_t **d_hashes, char *h_words, int *h_lengths, uint64_t *h_hashes, int numWords);
//void copyXXHashMemoryBackToHost(char *h_words, int *h_lengths, uint64_t *h_hashes, char *d_words, int *d_lengths, uint64_t *d_hashes, int numWords);
//void freeXXHashMemoryOnGPU(char *d_words, int *d_lengths, uint64_t *d_hashes);

*/
import "C"
import (
	"errors"
	"fmt"
	"github.com/cespare/xxhash/v2"
	"sort"
	"strings"
	"unicode"
	"unsafe"
)

// Rule contains the core data structure
// Todo: make attributes private
type Rule struct {
	Function          string
	Parameter1        string
	NumericParameter1 int
	Parameter2        string
	NumericParameter2 int
	Process           func(string) string // A lambda function for improved processing
}

// Helper function to reverse a string.
func reverseString(s string) string {
	runes := []rune(s)
	sort.SliceStable(runes, func(i, j int) bool {
		return i > j
	})
	return string(runes)
}

// CreateTestWords generates an array of words used to validate and verify if Rules are unique with UniqueID
//
//	testWords := rule.CreateTestWords()
//	Rules, _ := rule.ConvertFromHashcat(1, "$1 ]")
//	id := rule.UniqueID(testWords, Rules)
//	Rules2, _ := rule.ConvertFromHashcat(1, ": $2 ]")
//	id2 := rule.UniqueID(testWords, Rules2)
func CreateTestWords() []string {
	var testWords []string

	// Create 256 strings, each containing 37 characters of the same value (0x0 to 0xff)
	for i := 0x0; i <= 0xff; i++ {
		testWords = append(testWords, strings.Repeat(string(rune(i)), 37))
	}

	// Create a string with all possible hex values repeated 37 times with an appended 'a'
	var allChars strings.Builder
	for i := 0x0; i <= 0xff; i++ {
		for j := 0; j < 37; j++ {
			allChars.WriteRune(rune(i))
			allChars.WriteRune('a')
		}
	}
	testWords = append(testWords, allChars.String())

	// Reverse the string created above
	allCharsReversed := reverseString(allChars.String())
	testWords = append(testWords, allCharsReversed)

	// Create alphanumeric strings of different lengths
	alphabet := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	for i := 0; i < 37; i++ {
		// Reverse every other alphabet
		currentAlphabet := alphabet
		if i%2 == 0 {
			currentAlphabet = reverseString(currentAlphabet)
		}
		// Erase characters from the start to keep only the last (i + 1) characters
		if len(currentAlphabet) > (i + 1) {
			currentAlphabet = currentAlphabet[len(currentAlphabet)-i-1:]
		}
		testWords = append(testWords, currentAlphabet)
	}
	return testWords
}

// UniqueID Generates an ID unique to the set of rules allowing you to compare two lines to each-other
func UniqueID(testWords *[]string, rules []Rule) uint64 {
	result := xxhash.New()
	for _, testWord := range *testWords {
		for _, rule := range rules {
			testWord = rule.Process(testWord)
		}
		_, err := result.WriteString(testWord)
		if err != nil {
			return 0
		}
	}
	return result.Sum64()
}

// ParameterCountRule displays the amount of required parameters
func ParameterCountRule(rule string) (int, error) {
	if len(rule) == 0 {
		return -1, errors.New("Empty String")
	}
	//////////////////////////////////////////////
	switch rule[0] {
	case ':':
		return 0, nil
	case 'l':
		return 0, nil
	case 'u':
		return 0, nil
	case 'c':
		return 0, nil
	case 'C':
		return 0, nil
	case 't':
		return 0, nil
	case 'r':
		return 0, nil
	case 'd':
		return 0, nil
	case 'f':
		return 0, nil
	case '{':
		return 0, nil
	case '}':
		return 0, nil
	case '[':
		return 0, nil
	case ']':
		return 0, nil
	case 'k':
		return 0, nil
	case 'K':
		return 0, nil
	case 'q':
		return 0, nil
	case 'E':
		return 0, nil
	case '@':
		return 1, nil
	case 'T':
		return 1, nil
	case 'p':
		return 1, nil
	case 'D':
		return 1, nil
	case 'Z':
		return 1, nil
	case 'z':
		return 1, nil
	case '$':
		return 1, nil
	case '^':
		return 1, nil
	case '<':
		return 1, nil
	case '>':
		return 1, nil
	case '_':
		return 1, nil
	case '\'':
		return 1, nil
	case '!':
		return 1, nil
	case '/':
		return 1, nil
	case 'y':
		return 1, nil
	case 'Y':
		return 1, nil
	case '-':
		return 1, nil
	case '+':
		return 1, nil
	case 'e':
		return 1, nil
	case '.':
		return 1, nil
	case ',':
		return 1, nil
	case 'L':
		return 1, nil
	case 'R':
		return 1, nil
	case 's':
		return 2, nil
	case 'S':
		return 2, nil
	case 'x':
		return 2, nil
	case 'O':
		return 2, nil
	case 'o':
		return 2, nil
	case 'i':
		return 2, nil
	case '3':
		return 2, nil
	case '*':
		return 2, nil
	}
	return -1, errors.New("Unknown Function")
}

// ParseTSVRules Takes TSV input and converts it to an array of Rules
func ParseTSVRules(lineCounter uint64, line string) ([]Rule, error) {
	rulesRaw := strings.Split(line, "\t")

	var parsedRules []Rule
	for _, rule := range rulesRaw {
		parsedRule, err := ParseSingleRule(rule)
		if err != nil {
			if err.Error() != "Empty String" {
				newError := errors.New(fmt.Sprintf("Invalid Rule \"%s\" with error \"%s\" on line [%d]: \"%s\"\n", rule, err.Error(), lineCounter, line))
				return parsedRules, newError
			}
			continue
		}
		parsedRules = append(parsedRules, parsedRule)
	}
	return parsedRules, nil
}

// ParseSingleRule Parses a single rule such as : or s31 or $b
func ParseSingleRule(originalRule string) (Rule, error) {
	parameterCount, err := ParameterCountRule(originalRule)
	if parameterCount == -1 {
		return Rule{}, err
	}
	if len([]rune(originalRule))-1 < parameterCount {
		return Rule{}, errors.New("Missing parameters")
	}

	myRule := Rule{}
	if parameterCount >= 0 {
		myRule.Function = originalRule[0:1]
	}
	if parameterCount >= 1 {
		myRule.Parameter1 = originalRule[1:2]
	}
	if parameterCount >= 2 {
		myRule.Parameter2 = originalRule[2:3]
	}

	alphabet := "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	if strings.Contains(alphabet, myRule.Parameter1) && len(myRule.Parameter1) > 0 {
		myRule.NumericParameter1 = strings.IndexRune(alphabet, rune(myRule.Parameter1[0]))
	}
	if strings.Contains(alphabet, myRule.Parameter2) && len(myRule.Parameter2) > 0 {
		myRule.NumericParameter2 = strings.IndexRune(alphabet, rune(myRule.Parameter2[0]))
	}

	switch myRule.Function {
	case ":":
		myRule.Process = func(input string) string {
			return input
		}
	case "l":
		myRule.Process = func(input string) string {
			return strings.ToLower(input)
		}
	case "u":
		myRule.Process = func(input string) string {
			return strings.ToUpper(input)
		}
	case "c":
		myRule.Process = func(input string) string {
			if len(input) <= 1 {
				return strings.ToUpper(input)
			}
			lower := strings.ToLower(input)
			return strings.ToUpper(string(lower[0])) + lower[1:]
		}
	case "C":
		myRule.Process = func(input string) string {
			if len(input) <= 1 {
				return strings.ToLower(input)
			}
			upper := strings.ToUpper(input)
			return strings.ToLower(string(upper[0])) + upper[1:]
		}
	case "t":
		myRule.Process = func(input string) string {
			var result []rune
			for _, r := range input {
				if unicode.IsLower(r) {
					result = append(result, unicode.ToUpper(r))
				} else if unicode.IsUpper(r) {
					result = append(result, unicode.ToLower(r))
				} else {
					result = append(result, r)
				}
			}
			return string(result)
		}
	case "q":
		myRule.Process = func(input string) string {
			var result strings.Builder
			for _, r := range input {
				result.WriteRune(r)
				result.WriteRune(r)
			}
			return result.String()
		}
	case "r":
		myRule.Process = func(input string) string {
			runes := []rune(input)
			for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
				runes[i], runes[j] = runes[j], runes[i]
			}
			return string(runes)
		}
	case "k":
		myRule.Process = func(input string) string {
			if len([]rune(input)) < 2 {
				return input
			}
			runes := []rune(input)
			runes[0], runes[1] = runes[1], runes[0]
			return string(runes)
		}
	case "K":
		myRule.Process = func(input string) string {
			if len([]rune(input)) < 2 {
				return input
			}
			runes := []rune(input)
			runes[len(runes)-1], runes[len(runes)-2] = runes[len(runes)-2], runes[len(runes)-1]
			return string(runes)
		}
	case "d":
		myRule.Process = func(input string) string {
			return input + input
		}
	case "f":
		myRule.Process = func(input string) string {
			runes := []rune(input)
			// Reverse the string
			for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
				runes[i], runes[j] = runes[j], runes[i]
			}
			return input + string(runes)
		}
	case "{":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			return input[1:] + string(input[0])
		}
	case "}":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			return string(input[len(input)-1]) + input[:len(input)-1]
		}
	case "[":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			return input[1:]
		}
	case "]":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			return input[:len(input)-1]
		}
	case "E":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			input = strings.ToLower(input)
			runes := []rune(input)
			runes[0] = unicode.ToUpper(runes[0])
			for i := 1; i < len(runes); i++ {
				if runes[i-1] == ' ' {
					runes[i] = unicode.ToUpper(runes[i])
				}
			}
			return string(runes)
		}
	case "T":
		myRule.Process = func(input string) string {
			if len(input) == 0 || myRule.NumericParameter1 >= len([]rune(input)) {
				return input
			}
			runes := []rune(input)
			if unicode.IsLower(runes[myRule.NumericParameter1]) {
				runes[myRule.NumericParameter1] = unicode.ToUpper(runes[myRule.NumericParameter1])
			} else if unicode.IsUpper(runes[myRule.NumericParameter1]) {
				runes[myRule.NumericParameter1] = unicode.ToLower(runes[myRule.NumericParameter1])
			}
			return string(runes)
		}
	case "p":
		myRule.Process = func(input string) string {
			var result strings.Builder
			for i := 0; i <= myRule.NumericParameter1; i++ {
				result.WriteString(input)
			}
			return result.String()
		}
	case "D":
		myRule.Process = func(input string) string {
			if len(input) == 0 || myRule.NumericParameter1 > len(input)-1 {
				return input
			}
			return input[:myRule.NumericParameter1] + input[myRule.NumericParameter1+1:]
		}
	case "z":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			return strings.Repeat(string(input[0]), myRule.NumericParameter1) + input
		}
	case "Z":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			return input + strings.Repeat(string(input[len(input)-1]), myRule.NumericParameter1)
		}
	case "'":
		myRule.Process = func(input string) string {
			if len(input) == 0 || myRule.NumericParameter1 > len(input) {
				return input
			}
			return input[:myRule.NumericParameter1]
		}
	case "s":
		myRule.Process = func(input string) string {
			return strings.ReplaceAll(input, myRule.Parameter1, myRule.Parameter2)
		}
	case "S":
		myRule.Process = func(input string) string {
			return strings.Replace(input, myRule.Parameter1, myRule.Parameter2, 1)
		}
	case "$":
		myRule.Process = func(input string) string {
			return input + myRule.Parameter1
		}
	case "^":
		myRule.Process = func(input string) string {
			return myRule.Parameter1 + input
		}
	case "y":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 == 0 {
				return input
			}
			if myRule.NumericParameter1 > len(input) {
				return input
			}
			return input[:myRule.NumericParameter1] + input
		}

	case "Y":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 == 0 {
				return input
			}
			if myRule.NumericParameter1 > len(input) {
				return input + input
			}
			return input + input[len(input)-myRule.NumericParameter1:]
		}

	case "L":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len(input) {
				return input
			}
			char := input[myRule.NumericParameter1]
			modifiedChar := char << 1
			return input[:myRule.NumericParameter1] + string(modifiedChar) + input[myRule.NumericParameter1+1:]
		}

	case "R":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len(input) {
				return input
			}
			char := input[myRule.NumericParameter1]
			modifiedChar := char >> 1
			return input[:myRule.NumericParameter1] + string(modifiedChar) + input[myRule.NumericParameter1+1:]
		}

	case "-":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len(input) {
				return input
			}
			char := input[myRule.NumericParameter1]
			modifiedChar := char - 1
			return input[:myRule.NumericParameter1] + string(modifiedChar) + input[myRule.NumericParameter1+1:]
		}

	case "+":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len(input) {
				return input
			}
			char := input[myRule.NumericParameter1]
			modifiedChar := char + 1
			return input[:myRule.NumericParameter1] + string(modifiedChar) + input[myRule.NumericParameter1+1:]
		}

	case "@":
		myRule.Process = func(input string) string {
			return strings.ReplaceAll(input, string(myRule.Parameter1[0]), "")
		}

	case ".":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1+1 >= len(input) {
				return input
			}
			modified := input[:myRule.NumericParameter1] + string(input[myRule.NumericParameter1+1]) + input[myRule.NumericParameter1+1:]
			return modified[:len(modified)-1]
		}

	case ",":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len(input) || myRule.NumericParameter1 <= 0 {
				return input
			}
			modified := input[:myRule.NumericParameter1] + string(input[myRule.NumericParameter1-1]) + input[myRule.NumericParameter1+1:]
			return modified
		}

	case "e":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			lower := strings.ToLower(input)
			result := strings.ToUpper(string(lower[0])) + lower[1:]
			for i := 1; i < len(result); i++ {
				if i+1 < len(result) && result[i] == myRule.Parameter1[0] {
					result = result[:i+1] + strings.ToUpper(string(result[i+1])) + result[i+2:]
				}
			}
			return result
		}

	case "i":
		myRule.Process = func(input string) string {
			if len(input) == 0 || myRule.NumericParameter1 > len(input) {
				return input
			}
			return input[:myRule.NumericParameter1] + myRule.Parameter2 + input[myRule.NumericParameter1:]
		}

	case "O":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len(input) || len(input) == 0 {
				return input
			}
			if myRule.NumericParameter1+myRule.NumericParameter2 < len(input) {
				return input[:myRule.NumericParameter1] + input[myRule.NumericParameter1+myRule.NumericParameter2:]
			}
			return input
		}

	case "o":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 < 0 || len(input) == 0 || myRule.NumericParameter1+len(myRule.Parameter2)-1 >= len(input) {
				return input
			}
			modified := input[:myRule.NumericParameter1] + myRule.Parameter2
			if myRule.NumericParameter1+len(myRule.Parameter2) < len(input) {
				modified += input[myRule.NumericParameter1+len(myRule.Parameter2):]
			}
			return modified
		}
	case "*":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 >= len([]rune(input)) || myRule.NumericParameter2 >= len([]rune(input)) || len(input) == 0 {
				return input
			}
			runes := []rune(input)
			runes[myRule.NumericParameter1], runes[myRule.NumericParameter2] = runes[myRule.NumericParameter2], runes[myRule.NumericParameter1]
			return string(runes)
		}
	case "x":
		myRule.Process = func(input string) string {
			if len(input) == 0 || myRule.NumericParameter1 >= len(input) {
				return input
			}
			if myRule.NumericParameter1+myRule.NumericParameter2 <= len(input) {
				input = input[myRule.NumericParameter1 : myRule.NumericParameter1+myRule.NumericParameter2]
			}
			return input
		}

	case "<":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 < len(input) {
				return input
			}
			return ""
		}

	case ">":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 > len(input) {
				return input
			}
			return ""
		}

	case "_":
		myRule.Process = func(input string) string {
			if myRule.NumericParameter1 == len(input) {
				return ""
			}
			return input
		}

	case "!":
		myRule.Process = func(input string) string {
			if strings.Contains(input, myRule.Parameter1) {
				return ""
			}
			return input
		}

	case "/":
		myRule.Process = func(input string) string {
			if !strings.Contains(input, myRule.Parameter1) {
				return ""
			}
			return input
		}

	case "3":
		myRule.Process = func(input string) string {
			if len(input) == 0 {
				return input
			}
			instance := -1
			runes := []rune(input)
			for i, char := range runes {
				if instance == myRule.NumericParameter1 {
					if unicode.IsLower(char) {
						runes[i] = unicode.ToUpper(char)
					} else if unicode.IsUpper(char) {
						runes[i] = unicode.ToLower(char)
					}
					break
				}
				if char == rune(myRule.Parameter2[0]) {
					instance++
				}
			}
			return string(runes)
		}
	}

	return myRule, nil
}

// ConvertFromHashcat converts a line of hashcat compatible rules to an array of Rule objects.
// Done by first converting to TSV and then using ParseTSVRules to convert to Rule objects.
func ConvertFromHashcat(lineCounter uint64, rawLine string) ([]Rule, error) {
	// Sets of each rawLine width
	singleWide := ":lucCtrdf{}[]kKqE"
	doubleWide := "TpDZz$^<>_'!/@-+yYLR.,e"
	tripleWide := "sxOoi*3"

	var formattedRule strings.Builder
	offset := 0

	for offset < len(rawLine) {
		baseRule := rawLine[offset]
		// Skip if it's the space separator
		if baseRule == ' ' {
			offset++
		} else if strings.Contains(singleWide, string(baseRule)) {
			// Check if the rawLine is 1 character wide
			formattedRule.WriteString(rawLine[offset:offset+1] + "\t")
			offset++
		} else if strings.Contains(doubleWide, string(baseRule)) {
			// Check if the rawLine is 2 characters wide
			if offset+3 < len(rawLine) && rawLine[offset+1:offset+3] == "\\x" {
				// Check for hex notation
				if offset+5 > len(rawLine) {
					return nil, errors.New(fmt.Sprintf("Missing rule parameters on line [%d]: \"%s\"\n", lineCounter, rawLine))
				}
				formattedRule.WriteString(rawLine[offset:offset+5] + "\t")
				offset += 5
			} else {
				if offset+2 > len(rawLine) {
					return nil, errors.New(fmt.Sprintf("Missing rule parameters on line [%d]: \"%s\"\n", lineCounter, rawLine))
				}
				formattedRule.WriteString(rawLine[offset:offset+2] + "\t")
				offset += 2
			}
		} else if strings.Contains(tripleWide, string(baseRule)) {
			// Check if the rawLine is 3 characters wide
			if offset+9 <= len(rawLine) && rawLine[offset+1:offset+3] == "\\x" && rawLine[offset+5:offset+7] == "\\x" {
				// Check for double hex notation
				formattedRule.WriteString(rawLine[offset:offset+9] + "\t")
				offset += 9
			} else if offset+6 <= len(rawLine) && (rawLine[offset+1:offset+3] == "\\x" || rawLine[offset+2:offset+4] == "\\x") {
				// Check for single hex notation
				formattedRule.WriteString(rawLine[offset:offset+6] + "\t")
				offset += 6
			} else {
				if offset+3 > len(rawLine) {
					return nil, errors.New(fmt.Sprintf("Missing rule parameters on line [%d]: \"%s\"\n", lineCounter, rawLine))
				}
				formattedRule.WriteString(rawLine[offset:offset+3] + "\t")
				offset += 3
			}
		} else if baseRule == '#' {
			// Ignore if the line is a comment
			offset = 254
		} else {
			// Error if the baseRule is unknown
			offset = 254
			return nil, errors.New(fmt.Sprintf("Unknown rule function \"%c\" on line [%d]: \"%s\"\n", baseRule, lineCounter, rawLine))
		}
	}
	// Remove last tab character
	TSVResult := formattedRule.String()
	if strings.HasSuffix(TSVResult, "\t") {
		TSVResult = TSVResult[:len(TSVResult)-1]
	}

	// Convert from TSV to actual Rule Objects
	rules, _ := ParseTSVRules(lineCounter, TSVResult)
	return rules, nil
}

// Returns a format that's easily printable.
func (r *Rule) PrintFormat() string {
	rule1Copy := r.Parameter1
	rule2Copy := r.Parameter2
	var debugString string

	// Replace tabs with hex encoded tabs for improved compatability between formats
	rule1Copy = strings.Replace(rule1Copy, "\t", "\\x09", -1)
	rule2Copy = strings.Replace(rule2Copy, "\t", "\\x09", -1)

	// Rule identification
	paramCount, _ := ParameterCountRule(r.Function)
	switch paramCount {
	case 0:
		debugString += r.Function
	case 1:
		debugString += r.Function + rule1Copy
	case 2:
		// Handle multibyte rules
		if len(r.Parameter1) > 1 {
			rule1Copy = strings.Replace(rule1Copy, "/", "\\/", -1)
			rule2Copy = strings.Replace(rule2Copy, "/", "\\/", -1)

			debugString += r.Function + "/" + rule1Copy + "/" + rule2Copy
		} else if len(r.Parameter2) > 1 { // if the second rule is multibyte
			rule1Copy = strings.Replace(rule1Copy, "/", "\\/", -1)
			rule2Copy = strings.Replace(rule2Copy, "/", "\\/", -1)

			debugString += r.Function + "/" + rule1Copy + "/" + rule2Copy
		} else { // rule 1 is not multibyte
			debugString += r.Function + rule1Copy + rule2Copy
		}
	}
	return debugString
}

func GetLinePerformance(ruleObjs []Rule) float32 {
	totalPerformance := float32(0)
	for _, ruleObj := range ruleObjs {
		totalPerformance += ruleObj.GetPerformance()
	}
	return totalPerformance
}

func (r *Rule) GetPerformance() float32 {
	// obtained by testing on a single NVIDIA 3070v1
	// perl -e 'print "D4\n" x 200000' > rule       (rule to test)
	// ./hashcat --potfile-disable -m900 afe04867ec7a3845145579a95f72e000 -O D:\Wordlists\rockyou.txt -r rule -n 64 -u 256 -T 512 --backend-vector-width 1 --force
	// Take the GH/s value, one decimal place rounded to nearest quarter
	switch r.Function {
	case ":":
		return 25
	case "l":
		return 16.5
	case "u":
		return 16.75
	case "c":
		return 16
	case "C":
		return 15.75
	case "t":
		return 16
	case "T":
		return 19.5
	case "r":
		return 16.25
	case "d":
		return 18
	case "p": // pA
		return 21.5
	case "f":
		return 14.5
	case "{":
		return 20
	case "}":
		return 19.5
	case "$": // $a
		return 22.75
	case "^": // ^a
		return 21.0
	case "[":
		return 22.75
	case "]":
		return 21
	case "D":
		return 21.75
	case "x": // x46
		return 19.75
	case "O": // O31
		return 20.5
	case "i": // i4c
		return 19
	case "o": // o5e
		return 19.5
	case "\"": // 5"
		return 20
	case "s":
		return 11.5
	case "@": // this one can go from 10-15 GH/s quite easily by choosing e or x respectively.
		return 10.25
	case "z":
		return 18.75
	case "Z":
		return 9.75
	case "q":
		return 18.5
	case "k":
		return 20.5
	case "K":
		return 19.75
	case "*":
		return 19.5
	case "L": // L4
		return 19.75
	case "R": // R4
		return 19.75
	case "+": // +4
		return 19.75
	case "-": // -4
		return 19.75
	case ".": // .4
		return 10
	case ",": // ,4
		return 10
	case "y": // y4
		return 17.75
	case "Y": // Y4
		return 18
	case "E":
		return 11.5
	case "e": // e-
		return 11
	case "3": // 30-
		return 14.5
	}
	return 15 // default a bit in the middle (lower end)
}

func FormatAllRules(allRules []Rule, optionalDelimiter ...string) string {
	delimiter := "\t"
	if len(optionalDelimiter) > 0 {
		delimiter = optionalDelimiter[0]
	}

	returnString := ""
	for _, rule := range allRules {
		returnString += rule.PrintFormat() + delimiter
	}
	if strings.HasSuffix(returnString, delimiter) {
		returnString = returnString[:len(returnString)-len(delimiter)]
	}
	return returnString
}

func CUDASingleRule(ruleLine *[]Rule, originalDict *[]string, originalDictCount uint64) *map[uint64]struct{} {
	//// Convert Go strings to a flat C array
	wordArray := make([]byte, (originalDictCount)*MaxLen)
	lengths := make([]int32, originalDictCount)
	hashes := make([]uint64, originalDictCount)

	for i, word := range *originalDict {
		copy(wordArray[i*MaxLen:], word)
		lengths[i] = int32(len(word))
	}

	// Convert Go slices to C pointers
	wordPtr := (*C.char)(unsafe.Pointer(&wordArray[0]))
	hashPtr := (*C.uint64_t)(unsafe.Pointer(&hashes[0]))
	lengthPtr := (*C.int)(unsafe.Pointer(&lengths[0]))
	// Allocate memory on GPU
	var d_words *C.char
	var d_hashes *C.uint64_t
	var d_lengths *C.int

	C.allocateMemoryOnGPU(&d_words, &d_lengths, &d_hashes, wordPtr, lengthPtr, hashPtr, C.int(originalDictCount))
	for _, rule := range *ruleLine {
		if rule.Function == "l" {
			C.applyLowerCase(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "u" {
			C.applyUpperCase(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "c" {
			C.applyCapitalize(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "C" {
			C.applyInvertCapitalize(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "t" {
			C.applyToggleCase(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "q" {
			C.applyDuplicateChars(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "r" {
			C.applyReverse(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "k" {
			C.applySwapFirstTwo(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "K" {
			C.applySwapLastTwo(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "d" {
			C.applyDuplicate(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "f" {
			C.applyReflect(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "{" {
			C.applyRotateLeft(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "}" {
			C.applyRotateRight(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "[" {
			C.applyDeleteFirst(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "]" {
			C.applyDeleteLast(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "E" {
			C.applyTitleCase(d_words, d_lengths, C.int(originalDictCount))
		}
		if rule.Function == "T" {
			C.applyTogglePosition(d_words, d_lengths, C.int(rule.NumericParameter1), C.int(originalDictCount))
		}
		if rule.Function == "$" {
			C.applyAppend(d_words, d_lengths, C.char(rule.Parameter1[0]), C.int(originalDictCount))
		}
		if rule.Function == "^" {
			C.applyPrepend(d_words, d_lengths, C.char(rule.Parameter1[0]), C.int(originalDictCount))
		}
		if rule.Function == "s" {
			C.applySubstitution(d_words, d_lengths, C.char(rule.Parameter1[0]), C.char(rule.Parameter2[0]), C.int(originalDictCount))
		}
		if rule.Function == "u" {
			C.applyUpperCase(d_words, d_lengths, C.int(originalDictCount))
		}
	}
	C.computeXXHashes(d_words, d_lengths, 0, d_hashes, C.int(originalDictCount))
	C.copyMemoryBackToHost(wordPtr, lengthPtr, hashPtr, d_words, d_lengths, d_hashes, C.int(originalDictCount))

	// Synchronize and convert to usable so we can cleanly get rid of memory.
	goHashes := make(map[uint64]struct{}, originalDictCount)
	for i := uint64(0); i < originalDictCount; i++ {
		// Convert the result element to a uint64.
		goHashes[uint64(*(*C.uint64_t)(unsafe.Pointer(&(hashes)[i])))] = struct{}{}
	}
	defer C.freeMemoryOnGPU(d_words, d_lengths, d_hashes)
	return &goHashes
}
