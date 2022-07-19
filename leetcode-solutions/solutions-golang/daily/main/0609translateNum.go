package main

import "strconv"

//目前报错，方法不通过
func translateNum(num int) int {
	s := strconv.Itoa(num)
	a, b := 1, 1
	var c int
	for i := 2; i < len(s)+1; i++ {
		tmp := s[i-2 : i]
		if tmp <= "25" && tmp >= "10" {
			c = a + b
		} else {
			c = b
		}
		a = c
		b = a
	}
	return a
}
