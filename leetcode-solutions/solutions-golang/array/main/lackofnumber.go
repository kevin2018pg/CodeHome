package main

func missingNumber_s1(nums []int) int {
	i, j := 0, len(nums)-1
	for i <= j {
		m := (i + j) / 2
		if nums[m] == m {
			i = m + 1
		} else {
			j = m - 1
		}
	}
	return nums[i]
}

func missingNumber_s2(nums []int) int {
	sum01 := len(nums) * (1 + len(nums)) / 2
	sum02 := 0
	for _, i := range nums {
		sum02 += i
	}
	return sum01 - sum02
}

func missingNumber_s3(nums []int) int {
	sum01 := len(nums) * (1 + len(nums)) / 2
	for _, i := range nums {
		sum01 -= i
	}
	return sum01
}

//数学法和二分查找法
