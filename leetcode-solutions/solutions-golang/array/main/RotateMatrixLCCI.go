package main

func rotate(matrix [][]int) {
	if len(matrix) == 0 {
		return
	}
	// 目前题目的意思是m=n
	m, n := len(matrix), len(matrix[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 交换
			if j > i {
				matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
			}
			// 循环到后半部分的时候进行翻转
			if j >= n/2 {
				matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
			}
		}
	}
}

//func main() {
//	mat := [][]int{[[1,2,3],[4,5,6],[7,8,9]]}
//	rotate(mat)
//}
