package offer

//打印从1到最大的n位数
func printNumbers(n int) []int {
	//计算终点
	max := 1
	for n > 0 {
		max *= 10
		n--
	}
	//加入数组
	var res []int
	for i := 1; i < max; i++ {
		res = append(res, i)
	}
	return res
}
