package main

//横向扫描
//LCP(S1…Sn)=LCP(LCP(LCP(S1,S2),S3),…Sn)
func longestCommonPrefix_s1(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	prefix, length := strs[0], len(strs)
	for i := 1; i < length; i++ {
		prefix = lcp(prefix, strs[i])
		if len(prefix) == 0 {
			break
		}
	}
	return prefix
}

func lcp(str1 string, str2 string) string {
	minlength, index := min(len(str1), len(str2)), 0
	for index < minlength && str1[index] == str2[index] {
		index++
	}
	return str1[:index]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

//纵向扫描
//纵向比较，比较每一列，遇到长度相等或者列所在字母不相等即可返回
//第一个单次如果是最短的且都是共同，循环完返回第一个单次；若不是最短的，返回的是切片形式
func longestCommonPrefix_s2(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	for i := 0; i < len(strs[0]); i++ {
		for j := 1; j < len(strs); j++ {
			if i == len(strs[j]) || strs[j][i] != strs[0][i] {
				return strs[0][:i]
			}
		}
	}
	return strs[0]
}

//二分查找法
func longestCommonPrefix_s3(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	isCommonPrefix := func(length int) bool {
		str0, count := strs[0][:length], len(strs)
		for i := 1; i < count; i++ {
			if strs[i][:length] != str0 {
				return false
			}
		}
		return true
	}
	minLength := len(strs[0])
	for _, s := range strs {
		if len(s) < minLength {
			minLength = len(s)
		}
	}
	low, high := 0, minLength
	for low < high {
		mid := (high-low+1)/2 + low
		if isCommonPrefix(mid) {
			low = mid
		} else {
			high = mid - 1
		}
	}
	return strs[0][:low]
}
