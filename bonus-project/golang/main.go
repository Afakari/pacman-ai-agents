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

func main() {
	size := flag.Int("size", 3, "Puzzle size 'n' (e.g., 3 for 3x3, min 2)")
	scramble := flag.Int("scramble", 0, "Number of random scramble moves (default: n*n*n)")
	showSteps := flag.Bool("show-steps", false, "Show solution steps (basic TUI support)")
	forceGC := flag.Bool("forcegc", false, "Force garbage collection between algorithms (not used in TUI directly)")
	cpuprofile := flag.String("cpuprofile", "", "write cpu profile to `file`")
	memprofile := flag.String("memprofile", "", "write memory profile to `file`")

	flag.Parse()

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

	m := initialModel(*size, actualScrambleMoves, *showSteps, *forceGC)
	p := tea.NewProgram(m, tea.WithAltScreen()) 

	if err := p.Start(); err != nil {
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
