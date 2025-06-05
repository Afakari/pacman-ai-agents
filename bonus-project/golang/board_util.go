package main

import (
	"math/rand"
	"time"
	"fmt"
)

func init() {
	rand.Seed(time.Now().UnixNano()) 
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func deepCopyBoard(board [][]int) [][]int {
	if board == nil {
		return nil
	}
	duplicate := make([][]int, len(board))
	for i := range board {
		duplicate[i] = make([]int, len(board[i]))
		copy(duplicate[i], board[i])
	}
	return duplicate
}

func GetNeighbors(state *PuzzleState) []*PuzzleState {
	neighbors := []*PuzzleState{}
	directions := []struct{ dr, dc int }{
		{-1, 0}, // Up
		{1, 0},  // Down
		{0, -1}, // Left
		{0, 1},  // Right
	}

	row, col := state.BlankPos.Row, state.BlankPos.Col

	for _, dir := range directions {
		newRow, newCol := row+dir.dr, col+dir.dc

		if newRow >= 0 && newRow < state.Size && newCol >= 0 && newCol < state.Size {
			newBoard := deepCopyBoard(state.Board)
			newBoard[row][col], newBoard[newRow][newCol] = newBoard[newRow][newCol], newBoard[row][col]
			neighbors = append(neighbors, NewPuzzleState(newBoard, state.Size, state.Moves+1, state))
		}
	}
	return neighbors
}

func GetGoalBoard(size int) [][]int {
	board := make([][]int, size)
	for i := 0; i < size; i++ {
		board[i] = make([]int, size)
		for j := 0; j < size; j++ {
			board[i][j] = i*size + j + 1
		}
	}
	board[size-1][size-1] = 0 
	return board
}

func IsSolvable(board [][]int, size int) bool {
	flatBoard := []int{}
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if board[r][c] != 0 {
				flatBoard = append(flatBoard, board[r][c])
			}
		}
	}

	inversions := 0
	for i := 0; i < len(flatBoard); i++ {
		for j := i + 1; j < len(flatBoard); j++ {
			if flatBoard[i] > flatBoard[j] {
				inversions++
			}
		}
	}

	blankRow := -1
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if board[r][c] == 0 {
				blankRow = r
				break
			}
		}
		if blankRow != -1 {
			break
		}
	}
	if blankRow == -1 {
		panic("No blank tile found for solvability check")
	}

	if size%2 == 1 { // Odd grid
		return inversions%2 == 0
	}
	// Even grid
	blankRowFromBottom := size - blankRow
	return (inversions+blankRowFromBottom)%2 == 1

}

func GenerateSolvablePuzzle(n int, scrambleMoves int) ([][]int, error) {
	if n < 2 {
		return nil, fmt.Errorf("puzzle size must be at least 2x2")
	}

	if scrambleMoves <= 0 { 
		scrambleMoves = n * n * n
	}

	maxAttempts := 100
	for attempt := 0; attempt < maxAttempts; attempt++ {
		board := GetGoalBoard(n)
		blankR, blankC := n-1, n-1
		moves := []struct{ dr, dc int }{
			{-1, 0}, {1, 0}, {0, -1}, {0, 1},
		}

		for i := 0; i < scrambleMoves; i++ {
			validMoves := []struct{ nr, nc int }{}
			for _, move := range moves {
				nr, nc := blankR+move.dr, blankC+move.dc
				if nr >= 0 && nr < n && nc >= 0 && nc < n {
					validMoves = append(validMoves, struct{ nr, nc int }{nr, nc})
				}
			}
			if len(validMoves) == 0 {
				continue
			}
			chosenMove := validMoves[rand.Intn(len(validMoves))]
			board[blankR][blankC], board[chosenMove.nr][chosenMove.nc] = board[chosenMove.nr][chosenMove.nc], board[blankR][blankC]
			blankR, blankC = chosenMove.nr, chosenMove.nc
		}

		if IsSolvable(board, n) {
			return board, nil
		}
	}
	return nil, fmt.Errorf("could not generate a solvable puzzle after %d attempts", maxAttempts)
}
