package offer

func exist(board [][]byte, word string) bool {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if dfs(board, i, j, 0, word) {
				return true
			}
		}
	}
	return false
}

func dfs(board [][]byte, i, j, k int, word string) bool {
	if i < 0 || j < 0 || i == len(board) || j == len(board[0]) || board[i][j] != word[k] {
		return false
	}
	//字符串全部匹配，返回true
	if k == len(word)-1 {
		return true
	}
	//进入DFS之前，将board[i][j]变量存于tmp，修改为字符‘ ’，代表此元素已访问过，避免搜索重复访问
	tmp := board[i][j]
	board[i][j] = ' '
	//沿上下左右四个方向搜索
	if dfs(board, i-1, j, k+1, word) || dfs(board, i+1, j, k+1, word) || dfs(board, i, j-1, k+1, word) || dfs(board, i, j+1, k+1, word) {
		return true
	}

	//从DFS出来后，还原当前矩阵元素，可以避免重复进入
	board[i][j] = tmp
	return false
}
