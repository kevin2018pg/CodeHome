package main

func productExceptSelf_s1(nums []int) []int {
	length := len(nums)
	// L 和 R 分别表示左右两侧的乘积列表
	L, R, answer := make([]int, length), make([]int, length), make([]int, length)
	// L[i] 为索引 i 左侧所有元素的乘积
	// 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
	L[0] = 1
	for i := 1; i < length; i++ {
		L[i] = nums[i-1] * L[i-1]
	}
	// R[i] 为索引 i 右侧所有元素的乘积
	// 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
	R[length-1] = 1
	for i := length - 2; i > -1; i-- {
		R[i] = nums[i+1] * R[i+1]
	}
	// 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
	for i := 0; i < length; i++ {
		answer[i] = L[i] * R[i]
	}
	return answer
}

func productExceptSelf_s2(nums []int) []int {
	//初始化数组长度n,初始化res=[0,0,...,0]
	n := len(nums)
	res := make([]int, n)
	//初试化乘积k=1
	k := 1
	//从左向右遍历，遍历区间[0,n)：
	//res每个位置保存它左侧所有元素的乘积。即res[i]=k,k*=nums[i]
	for i := 0; i < n; i++ {
		res[i] = k
		k *= nums[i]
	}
	//重置乘积k=1，用来保存元素右边的乘积和
	k = 1
	//从右向左遍历，遍历区间(n,0]：
	//res[i]*=k，表示将当前位置的左积乘以右积。
	//更新右积k*=nums[i],返回res
	for i := n - 1; i > -1; i-- {
		res[i] *= k
		k *= nums[i]
	}
	return res
}
