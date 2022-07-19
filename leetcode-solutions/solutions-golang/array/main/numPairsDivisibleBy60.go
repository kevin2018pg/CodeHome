package main

import "fmt"

func numPairsDivisibleBy60_s1(time []int) int {
	res := 0
	m := make([]int, 60)
	for _, t := range time {
		if t%60 == 0 {
			res += m[0]
		} else {
			res += m[60-t%60]
		}
		m[t%60]++
	}
	return res
}

func numPairsDivisibleBy60_s2(time []int) int {
	res := 0
	m := make(map[int]int, 60)
	for _, t := range time {
		if t%60 == 0 {
			res += m[0]
		} else {
			res += m[60-t%60]
		}
		m[t%60]++
		fmt.Println(m)
	}

	return res
}
func main() {
	time_array := []int{60, 60, 30, 20, 20, 150, 100, 40}
	res1 := numPairsDivisibleBy60_s1(time_array)
	res2 := numPairsDivisibleBy60_s2(time_array)
	fmt.Println(res1)
	fmt.Println(res2)
}
