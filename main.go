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
	"fmt"
	"log"
	"os"
	"regexp"
	"sync"

	"github.com/alecthomas/kong"
)

type CLI struct {
	Score struct {
		Wordlist   string `arg:"" help:"Path to wordlist file" placeholder:"small_wordlist.txt"`
		Target     string `arg:"" help:"Path to target data file" placeholder:"big_wordlist.txt"`
		RuleFile   string `short:"r" help:"Rule file to analyse." required:"" placeholder:"best66.rule"`
		OutputFile string `short:"o" help:"Score File to output results to." required:"" placeholder:"best66.score"`
	} `cmd:"" help:"Score rule files."`
	Optimize struct {
		Wordlist   string `arg:"" help:"Path to wordlist file"`
		Target     string `arg:"" help:"Path to target data file"`
		ScoreFile  string `short:"s" help:"Aggregated score file TSV." required:"" placeholder:"best66.score"`
		OutputFile string `short:"o" help:"Score File to output results to." required:"" placeholder:"best66.optimized"`
		SaveEvery  int    `help:"Save progress every x rules." default:"1000"`
	} `cmd:"" help:"Optimize a score file."`
	Simulate struct {
		Wordlist   string `arg:"" help:"Path to wordlist file"`
		Target     string `arg:"" help:"Path to target data file"`
		RuleFile   string `short:"r" help:"Rule file to analyse." required:"" placeholder:"best66.rule"`
		OutputFile string `short:"o" help:"Score File to output results to." required:"" placeholder:"best66.sim"`
		DeviceID   int    `short:"d" help:"Device ID." placeholder:"0"`
	} `cmd:"" help:"Run a simulation on the target list."`
	Format struct { // Add the ability to remove 0 scores
		ScoreFile  string `short:"s" help:"Aggregated score file TSV." required:"" placeholder:"best66.score"`
		OutputFile string `short:"o" help:"Hashcat rule file output." required:"" placeholder:"best66.rule"`
	} `cmd:"" help:"Remove the scores from the TSV file and transform it into a hashcat-compatible file."`
	Session string `short:"x" help:"Session Name." default:"default" placeholder:"default"`
	Version struct {
	} `cmd:"" help:"Version & Author information"`
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

// Max length of words to support. Note that this will increase VRAM usage significantly.
// 32 was chosen to match the Hashcat -O kernel limits.
const MaxLen = 32

var stateFile string
var stateHashFile string

func main() {
	var cli CLI
	ctx := kong.Parse(&cli,
		kong.Name("ruleSetOptimizer"),
		kong.Description("An application that optimizes Hashcat rules using set coverage optimization theory based on rule performance."),
		kong.UsageOnError(),
	)
	isAlpha := regexp.MustCompile(`^[A-Za-z]+$`).MatchString
	if !isAlpha(cli.Session) {
		log.Fatalf("The session name must be alphanumreic\n")
		return
	}
	stateFile = fmt.Sprintf("rules-%s.state", cli.Session)
	stateHashFile = fmt.Sprintf("hashes-%s.state", cli.Session)

	if fileExists(stateFile) && !fileExists(stateHashFile) {
		log.Println("Rule state exists, but hash state does not. Both should exist.")
		os.Exit(-1)
	}

	if !fileExists(stateFile) && fileExists(stateHashFile) {
		log.Println("Hash state exists, but rule state does not. Both should exist.")
		os.Exit(-1)
	}

	switch ctx.Command() {
	case "score <wordlist> <target>":
		score(cli)
		break
	case "optimize <wordlist> <target>":
		optimize(cli)
		break
	case "simulate <wordlist> <target>":
		simulate(cli)
		break
	case "format":
		formatScore(cli)
		break
	case "version":
		printVersion()
		break
	default:
		panic(ctx.Command())
	}
	log.Println("Thank you for using this tool.")
}

// Score will score each rule individually, not keeping track of what other rules have found but calculating the abosolute
// maximum amount of founds that a single rule will obtain. This is required to efficiently precalculate the amount of possible founds that can be found.
func score(cli CLI) {
	if !fileExists(cli.Score.RuleFile) {
		log.Println("The rule file cannot be found, please verify that it exists.")
		os.Exit(-1)
	}
	_, err := os.Stat(cli.Score.OutputFile)
	if err == nil && !fileExists(stateFile) {
		log.Println("Output file exists, quitting.")
		os.Exit(-1)
	}
	generatePhase1(cli)
}

// Optimize will find the best rule and remove the founds, then recalculating all hashes until no better rules can be found.
// If no better rules can be found it will take that as 'best rule' and remove the founds, repeating this process until either all
// rules run out or there are no more plains to find. By default, (not configurable) it will append the remainder of rules.
// The output is a TSV file with the first column containing the amount of new cracks. You must remove all with a score 0.
func optimize(cli CLI) {
	_, err := os.Stat(cli.Optimize.OutputFile)
	if err == nil && !fileExists(stateFile) {
		log.Println("Output file exists, quitting.")
		os.Exit(-1)
	}
	generatePhase2(cli)
}

// Simulate will attempt to predict how each rule will crack hashes in hashcat line by line, procedurally removing founds as if cracking them.
// The output is a TSV file where the first column is amount of founds that hashcat would find on the target wordlist.
// It simulates a hashcat run with the wordlist, rules, and target wordlist and shows how many cracks are obtained line by line. Great to graph.
func simulate(cli CLI) {
	_, err := os.Stat(cli.Simulate.OutputFile)

	if !fileExists(cli.Simulate.RuleFile) {
		log.Println("The rule file cannot be found, please verify that it exists.")
		os.Exit(-1)
	}
	if err == nil && !fileExists(stateFile) {
		log.Println("Output file already exists. Quitting.")
		os.Exit(-1)
	}
	generateSimulate(cli)
}

// Convert scores to rules
func formatScore(cli CLI) {
	if fileExists(cli.Format.OutputFile) {
		log.Println("The rule file already exists, quitting.")
		os.Exit(-1)
	}
	if !fileExists(cli.Format.ScoreFile) {
		log.Println("The score file does not exist, quitting.")
		os.Exit(-1)
	}

	scores := loadRuleScores(cli.Format.ScoreFile)
	outputFile, err := os.OpenFile(cli.Format.OutputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening or creating output file:", err)
		return
	}

	for _, rule := range scores {
		if _, err = outputFile.WriteString(FormatAllRules(rule.RuleLine, " ") + "\n"); err != nil {
			fmt.Println("Error writing to output file:", err)
			return
		}
	}
}

func printVersion() {
	fmt.Println("You are running version: 1.0")
	fmt.Println("Authored by: Vavaldi (vavaldi@hashmob.net)")
}
