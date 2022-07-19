package offer

//数值的整数次方
//自定义幂次方pow函数

func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	//需要判断n小于0的情况
	if n < 0 {
		return 1 / myPow(x, -n)
	}
	//判断奇偶数
	if n&1 == 1 {
		//return x * myPow(x*x, n/2)
		return x * myPow(x, n-1)
	} else {
		return myPow(x*x, n/2)
	}
}
