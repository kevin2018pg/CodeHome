func uniquePathsWithObstacles(obstacleGrid [][]int) int {
    row, col := len(obstacleGrid),len(obstacleGrid[0])
    // 定义一个dp数组
    dp := make([][]int,row)
    for i, _ := range dp {
		dp[i] = make([]int, col)
	}
	// 初始化, 如果是障碍物, 后面的就都是0, 不用循环了
	for i:=0;i<col && obstacleGrid[0][i]!=1;i++{
        dp[0][i] = 1
    }
    for i:=0;i<row && obstacleGrid[i][0]!=1;i++{
        dp[i][0] = 1
    }
    for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			// 如果obstacleGrid[i][j]这个点是障碍物, 那么dp[i][j]保持为0
			if obstacleGrid[i][j] != 1 {
				// 否则我们需要计算当前点可以到达的路径数
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[row-1][col-1]
}