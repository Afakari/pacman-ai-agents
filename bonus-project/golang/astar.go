package main

import (
	"container/heap"
	"time"
)

type PriorityQueue []*PuzzleState

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].FCostAStar() < pq[j].FCostAStar()
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	item := x.(*PuzzleState)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*pq = old[0 : n-1]
	return item
}

func AStarSearch(initialBoard [][]int, size int) (solution *PuzzleState, duration time.Duration, memUsageMB float64, nodesExpanded int) {
	startTime := time.Now()
	nodesExpanded = 0

	initialState := NewPuzzleState(initialBoard, size, 0, nil)
	goalBoard := GetGoalBoard(size)
	goalStateForComparison := NewPuzzleState(goalBoard, size, 0, nil)
	goalKey := goalStateForComparison.Key()

	openList := &PriorityQueue{}
	heap.Init(openList)
	heap.Push(openList, initialState)

	closedSet := make(map[string]struct{})

	for openList.Len() > 0 {
		currentState := heap.Pop(openList).(*PuzzleState)
		currentKey := currentState.Key()

		if currentKey == goalKey {
			duration = time.Since(startTime)
			memUsageMB = getMemUsageMB()
			return currentState, duration, memUsageMB, nodesExpanded
		}

		if _, found := closedSet[currentKey]; found {
			continue
		}
		closedSet[currentKey] = struct{}{}

		nodesExpanded++
		for _, neighbor := range GetNeighbors(currentState) {
			neighborKey := neighbor.Key()
			if _, found := closedSet[neighborKey]; !found {
				heap.Push(openList, neighbor)
			}
		}
	}

	duration = time.Since(startTime)
	memUsageMB = getMemUsageMB()
	return nil, duration, memUsageMB, nodesExpanded
}
