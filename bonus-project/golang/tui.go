package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)


type appState int

const (
	stateInitializing appState = iota
	stateGeneratingPuzzle
	statePuzzleGenerated 
	stateRunningAStar
	stateAStarDone
	stateRunningRBFS
	stateRBFSDone
	stateComparingResults
	stateShowingError
	stateQuitting
)

// Messages
type puzzleGeneratedMsg struct {
	board [][]int
	err   error
}
type aStarResultMsg struct {
	result    *PuzzleState
	path      [][][]int
	duration  time.Duration
	memory    float64
	nodes     int
	err       error
}
type rbfsResultMsg struct {
	result    *PuzzleState
	path      [][][]int
	duration  time.Duration
	memory    float64
	nodes     int
	err       error
}

type model struct {
	puzzleSize    int
	scrambleMoves int
	showSteps     bool 
	forceGC       bool

	state          appState
	spinner        spinner.Model
	loadingMessage string
	errorMessage   string
	width, height  int 

	initialBoard [][]int
	goalBoardKey string 

	aStarResult    *PuzzleState
	aStarPath      [][][]int
	aStarDuration  time.Duration
	aStarMemory    float64
	aStarNodes     int
	aStarSolved    bool
	aStarError     error

	rbfsResult    *PuzzleState
	rbfsPath      [][][]int
	rbfsDuration  time.Duration
	rbfsMemory    float64
	rbfsNodes     int
	rbfsSolved    bool
	rbfsError     error

	styleHelp      lipgloss.Style
	styleError     lipgloss.Style
	styleHeader    lipgloss.Style
	styleMuted     lipgloss.Style
	styleStrong    lipgloss.Style
	styleBorderBox lipgloss.Style
}

func initialModel(pSize, sMoves int, sSteps, fGC bool) model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205")) 

	gb := GetGoalBoard(pSize)
	gKey := NewPuzzleState(gb, pSize, 0, nil).Key()


	return model{
		puzzleSize:    pSize,
		scrambleMoves: sMoves,
		showSteps:     sSteps, 
		forceGC:       fGC,
		state:         stateInitializing,
		spinner:       s,
		goalBoardKey:  gKey,

		styleHelp:      lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Margin(1, 0),
		styleError:     lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true),
		styleHeader:    lipgloss.NewStyle().Bold(true).Padding(0,1).Background(lipgloss.Color("62")).Foreground(lipgloss.Color("230")),
		styleMuted:     lipgloss.NewStyle().Foreground(lipgloss.Color("244")),
		styleStrong:    lipgloss.NewStyle().Bold(true),
		styleBorderBox: lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0,1),
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(m.spinner.Tick, m.generatePuzzleCmd())
}

func (m model) generatePuzzleCmd() tea.Cmd {
	return func() tea.Msg {
		m.loadingMessage = "Generating puzzle..."
		board, err := GenerateSolvablePuzzle(m.puzzleSize, m.scrambleMoves)
		if err != nil {
			board, err = GenerateSolvablePuzzle(m.puzzleSize, m.scrambleMoves+1)
		}
		return puzzleGeneratedMsg{board: board, err: err}
	}
}

func (m model) runAStarCmd() tea.Cmd {
	return func() tea.Msg {
		res, dur, mem, nodes := AStarSearch(deepCopyBoard(m.initialBoard), m.puzzleSize) 
		var path [][][]int
		if res != nil {
			path = ReconstructPath(res)
		}
		return aStarResultMsg{result: res, path: path, duration: dur, memory: mem, nodes: nodes}
	}
}

func (m model) runRBFSCmd() tea.Cmd {
	return func() tea.Msg {
		res, dur, mem, nodes := SolveRBFS(deepCopyBoard(m.initialBoard), m.puzzleSize) 
		var path [][][]int
		if res != nil {
			path = ReconstructPath(res)
		}
		return rbfsResultMsg{result: res, path: path, duration: dur, memory: mem, nodes: nodes}
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			m.state = stateQuitting
			return m, tea.Quit
		case "enter":
			switch m.state {
			case statePuzzleGenerated:
				m.state = stateRunningAStar
				m.loadingMessage = "Running A* Search..."
				cmds = append(cmds, m.spinner.Tick, m.runAStarCmd())
			case stateAStarDone:
				m.state = stateRunningRBFS
				m.loadingMessage = "Running RBFS Search..."
				cmds = append(cmds, m.spinner.Tick, m.runRBFSCmd())
			case stateRBFSDone:
				m.state = stateComparingResults
			case stateComparingResults:
				m.state = stateQuitting 
				return m, tea.Quit
			}
		}

	case spinner.TickMsg:
		if m.state == stateGeneratingPuzzle || m.state == stateRunningAStar || m.state == stateRunningRBFS {
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}

	case puzzleGeneratedMsg:
		if msg.err != nil {
			m.state = stateShowingError
			m.errorMessage = fmt.Sprintf("Fatal: Could not generate a solvable puzzle: %v", msg.err)
		} else {
			m.initialBoard = msg.board
			m.state = statePuzzleGenerated
			m.loadingMessage = ""
		}

	case aStarResultMsg:
		m.aStarResult = msg.result
		m.aStarPath = msg.path
		m.aStarDuration = msg.duration
		m.aStarMemory = msg.memory
		m.aStarNodes = msg.nodes
		m.aStarError = msg.err
		m.aStarSolved = msg.result != nil
		m.state = stateAStarDone
		m.loadingMessage = ""

	case rbfsResultMsg:
		m.rbfsResult = msg.result
		m.rbfsPath = msg.path
		m.rbfsDuration = msg.duration
		m.rbfsMemory = msg.memory
		m.rbfsNodes = msg.nodes
		m.rbfsError = msg.err
		m.rbfsSolved = msg.result != nil
		m.state = stateRBFSDone
		m.loadingMessage = ""
	}

	if m.state == stateInitializing { 
		m.state = stateGeneratingPuzzle
		m.loadingMessage = "Initializing N-Puzzle Solver..."
	}

	return m, tea.Batch(cmds...)
}

func (m model) View() string {
	if m.state == stateQuitting {
		return "Quitting N-Puzzle Solver.\n"
	}
	if m.width == 0 { 
		return "Initializing..."
	}

	var s strings.Builder
	s.WriteString(m.styleHeader.Render("N-Puzzle Solver TUI") + "\n\n")

	switch m.state {
	case stateInitializing, stateGeneratingPuzzle, stateRunningAStar, stateRunningRBFS:
		s.WriteString(fmt.Sprintf("%s %s\n\n", m.spinner.View(), m.loadingMessage))
		if m.state == stateGeneratingPuzzle && m.initialBoard != nil { 
			s.WriteString(formatPuzzleBoard(m.initialBoard, m.puzzleSize, "Generating..."))
		}
	case statePuzzleGenerated:
		s.WriteString("Puzzle Generated Successfully!\n")
		s.WriteString(formatPuzzleBoard(m.initialBoard, m.puzzleSize, "Initial Puzzle State"))
		s.WriteString(m.styleHelp.Render("\nPress [Enter] to run A* Search, or [q] to quit."))
	case stateAStarDone:
		s.WriteString(m.styleStrong.Render("A* Search Results:\n"))
		s.WriteString(formatAlgorithmResult(
			"A*", m.aStarSolved, m.aStarPath, m.aStarDuration, m.aStarMemory, m.aStarNodes, m.aStarError,
		))
		if m.showSteps && m.aStarSolved {
			s.WriteString("\n--- A* Solution Path (first 5 steps) ---\n")
			s.WriteString(formatSolutionPath(m.aStarPath, m.puzzleSize, 5))
		}
		s.WriteString(m.styleHelp.Render("\nPress [Enter] to run RBFS Search, or [q] to quit."))
	case stateRBFSDone:
		s.WriteString(m.styleStrong.Render("RBFS Search Results:\n"))
		s.WriteString(formatAlgorithmResult(
			"RBFS", m.rbfsSolved, m.rbfsPath, m.rbfsDuration, m.rbfsMemory, m.rbfsNodes, m.rbfsError,
		))
		if m.showSteps && m.rbfsSolved {
			s.WriteString("\n--- RBFS Solution Path (first 5 steps) ---\n")
			s.WriteString(formatSolutionPath(m.rbfsPath, m.puzzleSize, 5))
		}
		s.WriteString(m.styleHelp.Render("\nPress [Enter] to view comparison, or [q] to quit."))
	case stateComparingResults:
		s.WriteString(m.styleStrong.Render("Algorithm Comparison:\n"))
		s.WriteString(formatComparisonTable(&m))
		s.WriteString(m.styleHelp.Render("\nPress [Enter] or [q] to quit."))
	case stateShowingError:
		s.WriteString(m.styleError.Render(m.errorMessage) + "\n")
		s.WriteString(m.styleHelp.Render("Press [q] to quit."))
	}
	return m.styleBorderBox.Width(m.width -2).Render(s.String())
}


func formatPuzzleBoard(board [][]int, size int, title string) string {
	if board == nil {
		return fmt.Sprintf("\n--- %s ---\n (nil board) \n", title)
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("\n--- %s (%dx%d) ---\n", title, size, size))
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if board[r][c] == 0 {
				sb.WriteString(fmt.Sprintf("%3s", " "))
			} else {
				sb.WriteString(fmt.Sprintf("%3d", board[r][c]))
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func formatSolutionPath(path [][][]int, size int, maxSteps int) string {
	var sb strings.Builder
	if path == nil {
		return " (No path to display)\n"
	}
	for i, boardState := range path {
		if i >= maxSteps && maxSteps > 0 {
			sb.WriteString(fmt.Sprintf("... and %d more steps ...\n", len(path)-maxSteps))
			break
		}
		sb.WriteString(formatPuzzleBoard(boardState, size, fmt.Sprintf("Step %d", i)))
	}
	return sb.String()
}


func formatAlgorithmResult(name string, solved bool, path [][][]int, dur time.Duration, mem float64, nodes int, err error) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Algorithm: %s\n", name))
	if err != nil {
		sb.WriteString(fmt.Sprintf("  Error: %v\n", err))
		return sb.String()
	}
	if solved {
		sb.WriteString("  Solution Found: Yes\n")
		sb.WriteString(fmt.Sprintf("  Path Length: %d moves\n", len(path)-1))
	} else {
		sb.WriteString("  Solution Found: No\n")
	}
	sb.WriteString(fmt.Sprintf("  Time Taken: %.4f s\n", dur.Seconds()))
	sb.WriteString(fmt.Sprintf("  Memory Used (approx Go heap): %.2f MB\n", mem))
	sb.WriteString(fmt.Sprintf("  Nodes Expanded: %d\n", nodes))
	return sb.String()
}

func formatComparisonTable(m *model) string {
	var sb strings.Builder
	header := []string{"Metric", "A*", "RBFS"}
	data := [][]string{
		{"Solution Found", boolToString(m.aStarSolved), boolToString(m.rbfsSolved)},
		{"Time Taken (s)", fmt.Sprintf("%.4f", m.aStarDuration.Seconds()), fmt.Sprintf("%.4f", m.rbfsDuration.Seconds())},
		{"Memory (MB)", fmt.Sprintf("%.2f", m.aStarMemory), fmt.Sprintf("%.2f", m.rbfsMemory)},
		{"Nodes Expanded", fmt.Sprintf("%d", m.aStarNodes), fmt.Sprintf("%d", m.rbfsNodes)},
		{"Path Length", pathLenToString(m.aStarPath), pathLenToString(m.rbfsPath)},
	}

	colWidths := make([]int, len(header))
	for i, h := range header {
		colWidths[i] = len(h)
	}
	for _, row := range data {
		for i, cell := range row {
			if len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	for i, h := range header {
		sb.WriteString(fmt.Sprintf("%-*s", colWidths[i]+2, h)) 
	}
	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("-", sumInts(colWidths)+len(colWidths)*2) + "\n")

	for _, row := range data {
		for i, cell := range row {
			sb.WriteString(fmt.Sprintf("%-*s", colWidths[i]+2, cell))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func boolToString(b bool) string {
	if b {
		return "Yes"
	}
	return "No"
}
func pathLenToString(path [][][]int) string {
	if path == nil { return "N/A" }
	return fmt.Sprintf("%d", len(path)-1)
}
func sumInts(arr []int) int {
	sum := 0
	for _, v := range arr { sum += v}
	return sum
}
