package main

import "sort"

func kidsWithCandies_s1(candies []int, extraCandies int) []bool {
	target := make([]int, len(candies))
	res := make([]bool, len(candies))
	copy(target, candies)
	sort.Ints(target)
	max := target[len(target)-1]
	for i := 0; i < len(candies); i++ {
		if candies[i]+extraCandies >= max {
			res[i] = true
		}
	}
	return res
}

func kidsWithCandies_s2(candies []int, extraCandies int) []bool {
	var max int
	for i := 0; i < len(candies); i++ {
		if candies[i] > max {
			max = candies[i]
		}
	}
	results := make([]bool, len(candies))
	for i := 0; i < len(candies); i++ {
		results[i] = candies[i]+extraCandies >= max
	}
	return results
}
