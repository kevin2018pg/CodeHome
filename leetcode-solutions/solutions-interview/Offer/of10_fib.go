/*
 * @Author: your name
 * @Date: 2020-11-18 17:44:12
 * @LastEditTime: 2020-11-18 17:54:25
 * @LastEditors: Please set LastEditors
 * Description: 剑指 Offer 10- I. 斐波那契数列

  写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：

  F(0) = 0,   F(1) = 1
  F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
  斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

  答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
 * @FilePath: \learningflow\Review\OfferSolution\of10_fib.go
*/

//斐波那契数列

func fib10(n int) int {
	if n < 2 {
		return n
	}
	var (
		a    = 0
		b    = 1
		temp int
	)
	for i := 2; i <= n; i++ {
		temp = b
		b = (a + b) % 1000000007
		a = temp
	}
	return b
}

// 传统DP方法
func fib11(n int) int {
	if n == 0 || n == 1 {
		return n
	}
	var ans = make([]int, n+1)
	ans[0] = 0
	ans[1] = 1
	for i := 2; i <= n; i++ {
		ans[i] = (ans[i-1] + ans[i-2]) % 1000000007
	}
	return ans[n]
}

//会超时传统递归方法
func fib2(n int) int {
	if n == 0 || n == 1 {
		return n
	}
	return fib2(n-1) + fib2(n-2)
}

//最优化DP方法
//由于 Python 中整形数字的大小限制 取决计算机的内存 （可理解为无限大），因此可不考虑大数越界问题。
func fib3(n int) int {
	a := 0
	b := 1
	for i := 1; i <= n; i++ {
		a, b = b, (a+b)%1000000007
	}
	return a
}