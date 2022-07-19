package main

import "fmt"

func numOfSubarrays_s1(arr []int, k int, threshold int) int {
	var ans, res int
	target := k * threshold
	for i := 0; i < len(arr); i++ {
		res += arr[i]
		if i >= k-1 {
			if res >= target {
				ans++
			}
			res -= arr[i-(k-1)]
		}
	}
	return ans
}

func numOfSubarrays_s2(arr []int, k int, threshold int) int {
	var ans int
	res := []int{0}
	target := k * threshold
	for i := 0; i < len(arr); i++ {
		res = append(res, res[len(res)-1]+arr[i])
	}
	for i := 0; i < (len(res) - k); i++ {
		a := res[i]
		b := res[i+k]
		if (b - a) >= target {
			ans += 1
		}
	}
	return ans
}

func main() {
	arr := []int{2, 2, 2, 2, 5, 5, 5, 8}
	ans1 := numOfSubarrays_s1(arr, 3, 4)
	ans2 := numOfSubarrays_s2(arr, 3, 4)
	fmt.Println(ans1)
	fmt.Println(ans2)
}
