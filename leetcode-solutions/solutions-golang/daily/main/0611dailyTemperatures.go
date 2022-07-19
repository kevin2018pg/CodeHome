package main

import (
	"math"
)

func dailyTemperatures_s1(T []int) []int {
	length := len(T)
	ans := make([]int, length)
	next := make([]int, 101)
	for i := 0; i < 101; i++ {
		next[i] = math.MaxInt32
	}
	for i := length - 1; i >= 0; i-- {
		warmerIndex := math.MaxInt32
		for t := T[i] + 1; t <= 100; t++ {
			if next[t] < warmerIndex {
				warmerIndex = next[t]
			}
		}
		if warmerIndex < math.MaxInt32 {
			ans[i] = warmerIndex - i
		}
		next[T[i]] = i
	}
	return ans
}

func dailyTemperatures_s2(T []int) []int {
	length := len(T)
	ans := make([]int, length)
	stack := []int{}
	for i := 0; i < length; i++ {
		temperature := T[i]
		for len(stack) > 0 && temperature > T[stack[len(stack)-1]] { // 栈不为空 && 栈顶温度小于当前温度
			prevIndex := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ans[prevIndex] = i - prevIndex
		}
		stack = append(stack, i)
	}
	return ans
}
