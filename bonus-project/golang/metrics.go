package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

type Job struct {
	ID       int
	Size     int
	Scramble int
}

type Result struct {
	JobID         int
	Label         string
	AStarMetric   AlgorithmRunMetric
	RBFSMetric    AlgorithmRunMetric
	PuzzleSuccess bool
}

type AlgorithmRunMetric struct {
	ExecutionTime float64
	MemoryUsage   float64
	NodesExpanded int
	PathLength    int
	Solved        bool
}

type AlgorithmMetrics struct {
	ExecutionTime []float64 `json:"execution_time"`
	MemoryUsage   []float64 `json:"memory_usage"`
	NodesExpanded []int     `json:"nodes_expanded"`
	PathLength    []int     `json:"path_length"`
}

type MetricsData struct {
	AStar AlgorithmMetrics `json:"a_star"`
	RBFS  AlgorithmMetrics `json:"rbfs"`
}

func worker(id int, jobs <-chan Job, results chan<- Result, rbfsNodeLimit int, wg *sync.WaitGroup) {
	defer wg.Done()
	for j := range jobs {
		log.Printf("[Worker %d] Starting job ID %d (%dx%d)...", id, j.ID, j.Size, j.Scramble)
		board, err := GenerateSolvablePuzzle(j.Size, j.Scramble)
		if err != nil {
			log.Printf("[Worker %d] Error: Failed to generate puzzle for job %d. Skipping. Error: %v", id, j.ID, err)
			results <- Result{JobID: j.ID, PuzzleSuccess: false}
			continue
		}

		boardCopyAStar := deepCopyBoard(board)
		solutionAStar, durationAStar, memAStar, nodesAStar := AStarSearch(boardCopyAStar, j.Size)
		pathLenAStar := -1
		if solutionAStar != nil {
			pathLenAStar = len(ReconstructPath(solutionAStar)) - 1
		}
		aStarResult := AlgorithmRunMetric{
			ExecutionTime: durationAStar.Seconds(),
			MemoryUsage:   memAStar,
			NodesExpanded: nodesAStar,
			PathLength:    pathLenAStar,
			Solved:        solutionAStar != nil,
		}
		log.Printf("[Worker %d] Job %d A* Result -> Solved: %v, Nodes: %d, Time: %.4fs", id, j.ID, aStarResult.Solved, aStarResult.NodesExpanded, aStarResult.ExecutionTime)
		runtime.GC()

		boardCopyRBFS := deepCopyBoard(board)
		solutionRBFS, durationRBFS, memRBFS, nodesRBFS := SolveRBFS(boardCopyRBFS, j.Size, rbfsNodeLimit)
		pathLenRBFS := -1
		if solutionRBFS != nil {
			pathLenRBFS = len(ReconstructPath(solutionRBFS)) - 1
		}
		rbfsResult := AlgorithmRunMetric{
			ExecutionTime: durationRBFS.Seconds(),
			MemoryUsage:   memRBFS,
			NodesExpanded: nodesRBFS,
			PathLength:    pathLenRBFS,
			Solved:        solutionRBFS != nil,
		}
		log.Printf("[Worker %d] Job %d RBFS Result -> Solved: %v, Nodes: %d, Time: %.4fs", id, j.ID, rbfsResult.Solved, rbfsResult.NodesExpanded, rbfsResult.ExecutionTime)
		runtime.GC()

		log.Printf("[Worker %d] Finished job ID %d.", id, j.ID)
		results <- Result{
			JobID:         j.ID,
			Label:         fmt.Sprintf("%dx%d", j.Size, j.Scramble),
			AStarMetric:   aStarResult,
			RBFSMetric:    rbfsResult,
			PuzzleSuccess: true,
		}
	}
}

func runMetricsCollection(rbfsNodeLimit int) {
	startTime := time.Now()

	numWorkers := runtime.NumCPU() / 4
	if numWorkers == 0 {
		numWorkers = 1
	}

	log.Printf("Starting metric collection with %d worker(s) (RBFS node limit: %d)...\n", numWorkers, rbfsNodeLimit)

	sizes := []int{2, 3, 4, 5, 6}
	scrambles := []int{20, 50, 80, 110, 140}
	numTestsPerConfig := 3
	totalJobs := len(sizes) * len(scrambles) * numTestsPerConfig

	jobs := make(chan Job, totalJobs)
	results := make(chan Result, totalJobs)
	var wg sync.WaitGroup

	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go worker(w, jobs, results, rbfsNodeLimit, &wg)
	}

	jobIDCounter := 0
	for _, size := range sizes {
		for _, scramble := range scrambles {
			for i := 0; i < numTestsPerConfig; i++ {
				jobs <- Job{ID: jobIDCounter, Size: size, Scramble: scramble}
				jobIDCounter++
			}
		}
	}
	close(jobs)
	log.Printf("All %d jobs have been dispatched. Waiting for workers to complete...", totalJobs)

	wg.Wait()
	close(results)
	log.Println("All workers have completed their jobs.")
	log.Println("Aggregating results...")

	resultsByLabel := make(map[string][]Result)
	for r := range results {
		if r.PuzzleSuccess {
			resultsByLabel[r.Label] = append(resultsByLabel[r.Label], r)
		}
	}

	var labels []string
	var finalAStarMetrics, finalRBFSMetrics AlgorithmMetrics

	for label := range resultsByLabel {
		labels = append(labels, label)
	}
	sort.Strings(labels)

	for _, label := range labels {
		var totalAStarTime, totalAStarMem, totalRBFS_Time, totalRBFS_Mem float64
		var totalAStarNodes, totalAStarPath, totalRBFS_Nodes, totalRBFS_Path, aStarSuccess, rbfsSuccess int

		for _, res := range resultsByLabel[label] {
			totalAStarTime += res.AStarMetric.ExecutionTime
			totalAStarMem += res.AStarMetric.MemoryUsage
			totalAStarNodes += res.AStarMetric.NodesExpanded
			if res.AStarMetric.Solved {
				aStarSuccess++
				totalAStarPath += res.AStarMetric.PathLength
			}

			totalRBFS_Time += res.RBFSMetric.ExecutionTime
			totalRBFS_Mem += res.RBFSMetric.MemoryUsage
			totalRBFS_Nodes += res.RBFSMetric.NodesExpanded
			if res.RBFSMetric.Solved {
				rbfsSuccess++
				totalRBFS_Path += res.RBFSMetric.PathLength
			}
		}

		numResults := float64(len(resultsByLabel[label]))
		finalAStarMetrics.ExecutionTime = append(finalAStarMetrics.ExecutionTime, totalAStarTime/numResults)
		finalAStarMetrics.MemoryUsage = append(finalAStarMetrics.MemoryUsage, totalAStarMem/numResults)
		finalAStarMetrics.NodesExpanded = append(finalAStarMetrics.NodesExpanded, int(float64(totalAStarNodes)/numResults))
		finalAStarMetrics.PathLength = append(finalAStarMetrics.PathLength, Ternary(aStarSuccess > 0, totalAStarPath/aStarSuccess, 0))

		finalRBFSMetrics.ExecutionTime = append(finalRBFSMetrics.ExecutionTime, totalRBFS_Time/numResults)
		finalRBFSMetrics.MemoryUsage = append(finalRBFSMetrics.MemoryUsage, totalRBFS_Mem/numResults)
		finalRBFSMetrics.NodesExpanded = append(finalRBFSMetrics.NodesExpanded, int(float64(totalRBFS_Nodes)/numResults))
		finalRBFSMetrics.PathLength = append(finalRBFSMetrics.PathLength, Ternary(rbfsSuccess > 0, totalRBFS_Path/rbfsSuccess, 0))
	}

	allMetrics := MetricsData{AStar: finalAStarMetrics, RBFS: finalRBFSMetrics}
	dataForJS := struct {
		Metrics MetricsData `json:"metrics"`
	}{Metrics: allMetrics}
	jsonData, err := json.MarshalIndent(dataForJS, "", "    ")
	if err != nil {
		log.Fatalf("Failed to marshal metrics data: %v", err)
	}

	var labelsBuilder strings.Builder
	labelsBuilder.WriteString("const labels = [\n")
	for i, label := range labels {
		labelsBuilder.WriteString(fmt.Sprintf("    '%s'", label))
		if i < len(labels)-1 {
			labelsBuilder.WriteString(",\n")
		}
	}
	labelsBuilder.WriteString("\n];\n")

	log.Printf("Metric collection finished in %s.", time.Since(startTime))
	fmt.Println("//============================================================")
	fmt.Println(labelsBuilder.String())
	fmt.Printf("const data = %s;\n", string(jsonData))

	fullChartData := struct {
		Labels  []string    `json:"labels"`
		Metrics MetricsData `json:"metrics"`
	}{Labels: labels, Metrics: allMetrics}
	fullJsonData, err := json.MarshalIndent(fullChartData, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal full chart data: %v", err)
	}
	err = os.WriteFile("metrics_data.json", fullJsonData, 0644)
	if err != nil {
		log.Fatalf("Failed to write metrics_data.json file: %v", err)
	}
	fmt.Println("\n// Raw JSON data for both labels and metrics saved to metrics_data.json")
}

func Ternary[T any](condition bool, a, b T) T {
	if condition {
		return a
	}
	return b
}
