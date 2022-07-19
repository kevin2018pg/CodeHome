package main

import "fmt"

func canThreePartsEqualSum_s1(A []int) bool {
	var sum, temp, count int
	for _, v := range A {
		sum += v
	}
	if sum%3 != 0 {
		return false
	}
	sum /= 3

	for _, value := range A {
		if count == 2 {
			return true
		}
		temp += value
		if temp == sum {
			count++
			temp = 0
		}
	}
	return false
}

func canThreePartsEqualSum_s2(A []int) bool {
	// 双指针法，先求两端，左端满足1/3结果值即求右端，且最后满足中间还有值即返回
	var sum, lef, righ int
	for _, v := range A {
		sum += v
	}
	if sum%3 != 0 {
		return false
	}
	var i, j int
	sum /= 3
	for i = 0; i < len(A); i++ {
		lef += A[i]
		if lef == sum {
			break
		}
	}
	for j = len(A) - 1; j >= 0; j-- {
		righ += A[j]
		if righ == sum {
			break
		}
	}
	if i+1 < j {
		return true
	}
	return false
}
func main() {
	A := []int{0, 2, 1, -6, 6, -7, 9, 1, 2, 0, 1}
	bo1 := canThreePartsEqualSum_s1(A)
	bo2 := canThreePartsEqualSum_s2(A)
	fmt.Println(bo1)
	fmt.Println(bo2)
}
