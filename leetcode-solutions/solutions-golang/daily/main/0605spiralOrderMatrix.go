package main

func spiralOrder(matrix [][]int) []int {
	row := len(matrix) //计算行数
	if row == 0 {
		return nil
	}
	col := len(matrix[0]) //计算列数
	if col == 0 {
		return nil
	}
	ans := []int{}
	//首先从助阵的top边开始遍历
	direction := "top"
	//初始化 左l 右r 上t 下b 四个游标
	for l, r, t, b := 0, col-1, 0, row-1; l <= r && t <= b; {
		switch direction {
		case "top":
			for i := l; i <= r; i++ {
				ans = append(ans, matrix[t][i])
			}
			direction = "right" //切换matrix right边
			t++
		case "right":
			for i := t; i <= b; i++ {
				ans = append(ans, matrix[i][r])
			}
			direction = "bottom" //切换切换matrix bottom边
			r--
		case "bottom":
			for i := r; i >= l; i-- {
				ans = append(ans, matrix[b][i])
			}
			b--
			direction = "left" //切换切换matrix left边
		case "left":
			for i := b; i >= t; i-- {
				ans = append(ans, matrix[i][l])
			}
			l++
			direction = "top" //切换切换matrix top边
		}
	}
	return ans
}
