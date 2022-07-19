package main

import "fmt"

func twoSum_s1(nums []int, target int) []int {
	m := make(map[int]int)
	for index, value := range nums {
		diff := target - value
		if v, ok := m[diff]; ok {
			return []int{index, v}
		}
		m[value] = index
	}
	return nil
}
func twoSum_s2(nums []int, target int) []int {
	m := make(map[int]int)
	for index, value := range nums {
		m[value] = index
	}
	for i, value := range nums {
		if j, ok := m[target-value]; ok && i != j {
			return []int{i, j}
		}
	}
	return nil
}

func main() {
	tar := 6
	nums := []int{2, 3, 3, 15}
	ind1 := twoSum_s1(nums, tar)
	ind2 := twoSum_s2(nums, tar)
	fmt.Println(ind1)
	fmt.Println(ind2)
}
