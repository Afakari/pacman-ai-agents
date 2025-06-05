package main

import (
	"math"
	"sort"
	"time"
)

type rbfsNodeCounter struct {
	count int
}

func SolveRBFS(initialBoard [][]int, size int) (solution *PuzzleState, duration time.Duration, memUsageMB float64, nodesExpanded int) {
	startTime := time.Now()
	counter := &rbfsNodeCounter{count: 0}

	if !IsSolvable(initialBoard, size) {
		duration = time.Since(startTime)
		memUsageMB = getMemUsageMB()
		return nil, duration, memUsageMB, 0
	}

	initialState := NewPuzzleState(initialBoard, size, 0, nil)
	initialState.FCostValue = float64(initialState.FCostAStar()) 

	goalBoard := GetGoalBoard(size)
	goalStateForComparison := NewPuzzleState(goalBoard, size, 0, nil)
	goalKey := goalStateForComparison.Key()


	resultState, _ := rbfsRecursive(initialState, goalKey, math.Inf(1), counter)

	duration = time.Since(startTime)
	memUsageMB = getMemUsageMB()
	nodesExpanded = counter.count
	return resultState, duration, memUsageMB, nodesExpanded
}

func rbfsRecursive(state *PuzzleState, goalKey string, fLimit float64, counter *rbfsNodeCounter) (*PuzzleState, float64) {
	if state.Key() == goalKey {
		return state, state.FCostValue 
	}

	counter.count++

	successors := GetNeighbors(state)
	if len(successors) == 0 {
		return nil, math.Inf(1)
	}

	for _, sNode := range successors {
		sNode.FCostValue = math.Max(float64(sNode.Moves+sNode.Manhattan), state.FCostValue)
	}

	for {
		sort.Slice(successors, func(i, j int) bool {
			return successors[i].FCostValue < successors[j].FCostValue
		})

		bestSuccessor := successors[0]

		if bestSuccessor.FCostValue > fLimit {
			return nil, bestSuccessor.FCostValue
		}

		alternativeFValue := math.Inf(1)
		if len(successors) > 1 {
			alternativeFValue = successors[1].FCostValue
		}

		newFLimit := math.Min(fLimit, alternativeFValue)
		resultState, bestFUpdated := rbfsRecursive(bestSuccessor, goalKey, newFLimit, counter)
		
		bestSuccessor.FCostValue = bestFUpdated 

		if resultState != nil {
			return resultState, bestFUpdated 
		}
	}
}
