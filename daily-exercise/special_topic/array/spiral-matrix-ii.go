func generateMatrix(n int) [][]int {
    // 初始化矩阵
    nums := make([][]int,n)
    for i := range nums{
        nums[i] = make([]int, n)
    }
    // 迭代次数
    loop,mid := n/2,n/2
    // 起始点
    startX,startY := 0,0
    count := 1
    for offset := 1;offset<loop+1;offset++{
        for i := startY;i<n-offset;i++{ // 从左至右，左闭右开
            nums[startX][i] = count
            count++
        }
        for i:= startX;i<n-offset;i++{  // 从上至下
            nums[i][n - offset] = count
            count++
        }
        for i:=n - offset;i>startY;i--{ // 从右至左
            nums[n - offset][i] = count
            count++
        }
        for i:= n-offset;i>startX;i--{  // 从下至上
            nums[i][startY] = count
            count++
        }
        // 更新起始点
        startX++
        startY++
    }
    if n %2 != 0{   //n为奇数时，填充中心点
        nums[mid][mid] = count
    }
    return nums
}