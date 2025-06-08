package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"

	tea "github.com/charmbracelet/bubbletea"
)

const defaultRbfsNodeLimit = 25000000

func main() {
	size := flag.Int("size", 3, "Puzzle size 'n' (e.g., 3 for 3x3, min 2)")
	scramble := flag.Int("scramble", 0, "Number of random scramble moves (default: n*n*n)")
	showSteps := flag.Bool("show-steps", false, "Show solution steps (TUI support)")
	forceGC := flag.Bool("forcegc", false, "Force garbage collection between algorithms")
	cpuprofile := flag.String("cpuprofile", "", "write cpu profile to `file`")
	memprofile := flag.String("memprofile", "", "write memory profile to `file`")

	collectMetrics := flag.Bool("collect-metrics", false, "Run benchmarks for sizes 2-6 and output JSON data for charts.")
	rbfsLimit := flag.Int("rbfs-limit", defaultRbfsNodeLimit, "Node expansion limit for RBFS to prevent very long runs.")

	flag.Parse()

	if *collectMetrics {
		fmt.Println("Metric collection mode enabled.")
		runMetricsCollection(*rbfsLimit)
		return
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *size < 2 {
		fmt.Println("Error: Puzzle size must be at least 2.")
		os.Exit(1)
	}
	actualScrambleMoves := *scramble
	if actualScrambleMoves == 0 {
		actualScrambleMoves = (*size) * (*size) * (*size)
	}
	if actualScrambleMoves < 0 {
		fmt.Println("Error: Scramble moves cannot be negative.")
		os.Exit(1)
	}

	m := initialModel(*size, actualScrambleMoves, *showSteps, *forceGC, *rbfsLimit)
	p := tea.NewProgram(m, tea.WithAltScreen())

	if _, err := p.Run(); err != nil {
		fmt.Printf("Error running Bubbletea program: %v\n", err)
		os.Exit(1)
	}

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal("could not create memory profile: ", err)
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
		fmt.Printf("Memory profile written to %s\n", *memprofile)
	}
}
