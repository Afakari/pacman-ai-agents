package main

import (
	"bytes"
	"strconv"
)

type Position struct {
	Row int
	Col int
}

type PuzzleState struct {
	Board      [][]int
	Size       int
	Moves      int        
	Parent     *PuzzleState
	BlankPos   Position
	Manhattan  int        
	FCostValue float64    
	boardKey   string     
}

func NewPuzzleState(board [][]int, size int, moves int, parent *PuzzleState) *PuzzleState {
	ps := &PuzzleState{
		Board:  deepCopyBoard(board), 
		Size:   size,
		Moves:  moves,
		Parent: parent,
	}
	ps.BlankPos = ps.findBlank()
	ps.Manhattan = ps.calculateManhattan()
	return ps
}

func (ps *PuzzleState) findBlank() Position {
	for r := 0; r < ps.Size; r++ {
		for c := 0; c < ps.Size; c++ {
			if ps.Board[r][c] == 0 {
				return Position{Row: r, Col: c}
			}
		}
	}
	panic("No blank tile found in the puzzle board.")
}

func (ps *PuzzleState) calculateManhattan() int {
	totalDistance := 0
	for r := 0; r < ps.Size; r++ {
		for c := 0; c < ps.Size; c++ {
			value := ps.Board[r][c]
			if value != 0 {
				targetRow := (value - 1) / ps.Size
				targetCol := (value - 1) % ps.Size
				totalDistance += abs(r-targetRow) + abs(c-targetCol)
			}
		}
	}
	return totalDistance
}

func (ps *PuzzleState) FCostAStar() int {
	return ps.Moves + ps.Manhattan
}

func (ps *PuzzleState) Key() string {
	if ps.boardKey != "" {
		return ps.boardKey
	}
	var b bytes.Buffer
	for i := 0; i < ps.Size; i++ {
		for j := 0; j < ps.Size; j++ {
			b.WriteString(strconv.Itoa(ps.Board[i][j]))
			if !(i == ps.Size-1 && j == ps.Size-1) {
				b.WriteString(",")
			}
		}
	}
	ps.boardKey = b.String()
	return ps.boardKey
}

func (ps *PuzzleState) IsGoal(goalBoardKey string) bool {
	return ps.Key() == goalBoardKey
}
