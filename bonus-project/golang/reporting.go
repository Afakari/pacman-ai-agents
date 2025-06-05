package main

import (
	"fmt"
	"runtime"
	"strings"
	"time"
)

// Metrics stores the performance data of a search algorithm.
type Metrics struct {
	AlgorithmName string
	SolutionFound bool
	TimeTaken     time.Duration
	MemoryUsedMB  float64
	NodesExpanded int
	PathLength    int
}

func getMemUsageMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	// m.Alloc is bytes allocated and not yet freed.
	// m.TotalAlloc is total bytes allocated (even if freed).
	// m.Sys is total bytes of memory obtained from the OS.
	// For a rough equivalent to psutil RSS, Sys is closer, but Alloc is often used for heap usage.
	return float64(m.Alloc) / (1024 * 1024)
}

func PrintPuzzle(board [][]int, size int, title string) {
	fmt.Printf("\n--- %s ---\n", title)
	if board == nil {
		fmt.Println(" (nil board) ")
		return
	}
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if board[r][c] == 0 {
				fmt.Printf("%3s", " ")
			} else {
				fmt.Printf("%3d", board[r][c])
			}
		}
		fmt.Println()
	}
	fmt.Println(strings.Repeat("-", len(title)+6))
}

func ReconstructPath(solutionNode *PuzzleState) [][][]int {
	var path [][][]int
	current := solutionNode
	for current != nil {
		path = append(path, current.Board) // Prepend to get correct order
		current = current.Parent
	}
	// Reverse the path
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

func PrintSolution(result *PuzzleState, algorithmName string, timeTaken time.Duration,
	memoryUsed float64, nodesExpanded int, puzzleSize int, showSteps bool) *Metrics {

	metrics := &Metrics{
		AlgorithmName: algorithmName,
		TimeTaken:     timeTaken,
		MemoryUsedMB:  memoryUsed,
		NodesExpanded: nodesExpanded,
	}

	fmt.Printf("\n=== %s Search Results ===\n", algorithmName)
	if result == nil {
		fmt.Println("Solution Found: No")
		metrics.SolutionFound = false
	} else {
		fmt.Println("Solution Found: Yes")
		metrics.SolutionFound = true
		path := ReconstructPath(result)
		metrics.PathLength = len(path) - 1
		fmt.Printf("Path Length: %d moves\n", metrics.PathLength)

		if showSteps {
			fmt.Println("Solution Steps:")
			for i, boardState := range path {
				PrintPuzzle(boardState, puzzleSize, fmt.Sprintf("Step %d", i))
			}
		}
	}
	fmt.Printf("Time Taken: %.4f s\n", timeTaken.Seconds())
	fmt.Printf("Memory Used (approx Go heap): %.2f MB\n", memoryUsed)
	fmt.Printf("Nodes Expanded: %d\n", nodesExpanded)
	fmt.Println(strings.Repeat("=", len(algorithmName)+20))
	return metrics
}

func PrintComparisonTable(aStarMetrics, rbfsMetrics *Metrics) {
	fmt.Printf("\n=============== A* vs RBFS Comparison ===============\n")
	fmt.Printf("%-18s | %-20s | %-20s\n", "Metric", "A*", "RBFS")
	fmt.Println(strings.Repeat("-", 60))

	formatBool := func(b bool) string {
		if b {
			return "Yes"
		}
		return "No"
	}
	formatNA := func(val interface{}, found bool, formatStr string, unit string) string {
		if !found {
			return "N/A"
		}
		return fmt.Sprintf(formatStr, val) + unit
	}

	if aStarMetrics == nil { aStarMetrics = &Metrics{} } // Ensure not nil for access
	if rbfsMetrics == nil { rbfsMetrics = &Metrics{} }


	fmt.Printf("%-18s | %-20s | %-20s\n", "Solution Found", formatBool(aStarMetrics.SolutionFound), formatBool(rbfsMetrics.SolutionFound))
	fmt.Printf("%-18s | %-20s | %-20s\n", "Time Taken",
		formatNA(aStarMetrics.TimeTaken.Seconds(), aStarMetrics.SolutionFound || aStarMetrics.TimeTaken > 0, "%.4f", " s"),
		formatNA(rbfsMetrics.TimeTaken.Seconds(), rbfsMetrics.SolutionFound || rbfsMetrics.TimeTaken > 0, "%.4f", " s"))
	fmt.Printf("%-18s | %-20s | %-20s\n", "Memory Used (MB)",
		formatNA(aStarMetrics.MemoryUsedMB, aStarMetrics.SolutionFound || aStarMetrics.MemoryUsedMB >= 0, "%.2f", ""),
		formatNA(rbfsMetrics.MemoryUsedMB, rbfsMetrics.SolutionFound || rbfsMetrics.MemoryUsedMB >= 0, "%.2f", ""))
	fmt.Printf("%-18s | %-20s | %-20s\n", "Nodes Expanded",
		formatNA(aStarMetrics.NodesExpanded, aStarMetrics.SolutionFound || aStarMetrics.NodesExpanded > 0, "%d", ""),
		formatNA(rbfsMetrics.NodesExpanded, rbfsMetrics.SolutionFound || rbfsMetrics.NodesExpanded > 0, "%d", ""))
	fmt.Printf("%-18s | %-20s | %-20s\n", "Path Length",
		formatNA(aStarMetrics.PathLength, aStarMetrics.SolutionFound, "%d", ""),
		formatNA(rbfsMetrics.PathLength, rbfsMetrics.SolutionFound, "%d", ""))
	fmt.Println(strings.Repeat("=", 60))
}
