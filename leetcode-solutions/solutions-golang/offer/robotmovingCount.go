package offer

// 主函数调用
func movingCount(m int, n int, k int) int {
	//建立存储节点map
	visited := make(map[string]bool)
	return matdfs(m, n, 0, 0, k, visited)
}

// DFS函数
func matdfs(m, n, i, j, k int, v map[string]bool) int {
	// map里有值说明遍历过了
	key := string(i) + string(j)
	_, ok := v[key]
	//终止条件：当①行列索引越界 ②数位和超出目标值k ③当前元素已访问过时，返回0 ，代表不计入可达解
	if i >= m || j >= n || sums(i)+sums(j) > k || ok {
		return 0
	}
	//标记当前单元格：将索引(i, j)存入Set visited中，代表此单元格已被访问过
	v[key] = true
	//计算当前元素的下、右两个方向元素的数位和，并开启下层递归
	return 1 + matdfs(m, n, i, j+1, k, v) + matdfs(m, n, i+1, j, k, v)
}

//求位数和的方法
func sums(n int) int {
	var sum int
	for n > 0 {
		//个位
		sum += n % 10
		//非个位
		n /= 10
	}
	return sum
}

/*
//双百题解
func movingCount(m int, n int, k int) int {
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	return dfs(m, n, 0, 0, k, dp)
}

func dfs(m, n, i, j, k int, dp [][]int) int {
	if i < 0 || j < 0 || i >= m || j >= n || dp[i][j] == 1 || (sumPos(i)+sumPos(j)) > k {
		return 0
	}

	dp[i][j] = 1

	sum := 1
	sum += dfs(m, n, i, j+1, k, dp)
	sum += dfs(m, n, i, j-1, k, dp)
	sum += dfs(m, n, i+1, j, k, dp)
	sum += dfs(m, n, i-1, j, k, dp)
	return sum
}

// 求所有位之和
func sumPos(n int) int {
	var sum int

	for n > 0 {
		sum += n % 10
		n = n / 10
	}

	return sum
}
*/
