package main

func sumNums_s1(n int) int {
	_ = n > 0 && func() bool { n += sumNums_s1(n - 1); return true }()
	return n
}

//Go 语言中整数和 bool 类型是不能隐式转换的，所以下面这种写法会编译失败。
//n > 0 && n += sumNums(n - 1)
//但是可以使用匿名函数封装一下

var res int // 在别的函数中调用，全局变量

func sumNums_s2(n int) int {
	res = 0
	sum_s2(n)
	return res
}

func sum_s2(n int) bool {
	res += n
	return n > 1 && sum_s2(n-1) // n=1时不满足第一个条件，造成短路后面不执行退出递归 */
}

// 指针操作方法
func sumNums_s3(n int) int {
	res := 0
	sum_s3(n, &res)
	return res
}

func sum_s3(n int, res *int) bool {
	*res += n
	return n > 1 && sum_s3(n-1, res)
}
