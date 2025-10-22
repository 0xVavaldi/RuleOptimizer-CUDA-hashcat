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
	"log"
	"os"
	"sync"

	"github.com/alecthomas/kong"
)

type CLI struct {
	Score struct {
		Wordlist   string `arg:"" help:"Path to wordlist file"`
		Target     string `arg:"" help:"Path to target data file"`
		RuleFile   string `short:"r" help:"Rule file to analyse."`
		OutputFile string `short:"o" help:"Score File to output results to."`
	} `cmd:"" help:"Score rule files."`
	Evaluate struct {
		Wordlist   string `arg:"" help:"Path to wordlist file"`
		Target     string `arg:"" help:"Path to target data file"`
		ScoreFile  string `short:"s" help:"Aggregated score file TSV."`
		OutputFile string `short:"o" help:"Score File to output results to."`
	} `cmd:"" help:"Optimize a score file."`
	Simulate struct {
		Wordlist   string `arg:"" help:"Path to wordlist file"`
		Target     string `arg:"" help:"Path to target data file"`
		RuleFile   string `short:"r" help:"Rule file to analyse."`
		OutputFile string `short:"o" help:"Score File to output results to."`
	} `cmd:"" help:"Run a simulation on the target list."`
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
const stateFile = "rules.state"
const stateHashFile = "hashes.state"

func main() {
	var cli CLI
	ctx := kong.Parse(&cli,
		kong.Name("CudaRuleScoreOptimizer"),
		kong.Description("An application that optimizes rule scores with set optimization theory based on performance."),
		kong.UsageOnError(),
	)

	if fileExists(stateFile) && !fileExists(stateHashFile) {
		log.Println("Rule state exists, but hash state does not. Both should exist.")
		os.Exit(-1)
	}

	if !fileExists(stateFile) && fileExists(stateHashFile) {
		log.Println("Hash state exists, but rule state does not. Both should exist.")
		os.Exit(-1)
	}

	switch ctx.Command() {
	case "score":
		score(cli)
		break
	case "evaluate":
		evaluate(cli)
		break
	case "simulate":
		simulate(cli)
		break
	default:
		panic(ctx.Command())
	}
}

func score(cli CLI) {
	_, err := os.Stat(cli.Score.OutputFile)
	if err == nil && !fileExists(stateFile) {
		log.Println("Output file exists, quitting.")
		os.Exit(-1)
	}
	generatePhase1(cli)
}

func evaluate(cli CLI) {
	generatePhase2(cli)
}

func simulate(cli CLI) {
	generateSimulate(cli)
}
