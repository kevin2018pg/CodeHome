/*
 * @Author: your name
 * @Date: 2020-11-11 12:19:29
 * @LastEditTime: 2020-11-12 12:54:49
 * @LastEditors: Please set LastEditors
 * @Description: 剑指 Offer 05. 替换空格

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
 * @FilePath: \learningflow\Review\OfferSolution\of05_replaceSpace.go
 */

Package OfferSolution

// 字符串拼接，Tips：+ 效率不如Join
func replaceSpace_01(s string) string {
	var str string = ""
	for _, val := range s {
		if val == " "{
			str += "%20"
		} else {
			str += string(v)
		}
	}
	return str
}

// 按空格切分，以%20拼接
func replaceSpace_02(s string) string {
	// Tips：strings.Fields(s)也可切分，但是测试用例 "   " 无法通过，切分规则：[256]uint8{'\t': 1, '\n': 1, '\v': 1, '\f': 1, '\r': 1, ' ': 1}
	return strings.Join(strings.Split(s," "),"%20")
}

// 构建字符切片方法
func replaceSpace_03(s string) string {
    var b []rune
	for _, v := range s {
		if v == 32 {
			b = append(b, 37, 50, 48)
		} else {
			b = append(b, v)
		}
	}
	return string(b)
}

// 简洁替换方法
func replaceSpace_04(s string) string {
    return strings.Replace(s," ","%20",-1)
}

// 最优方法：效率双百
func replaceSpace_05(s string) string {
	var res strings.Builder
	for _, i := range s {
		if i == 32 {
			res.WriteString("%20")
		} else {
			res.WriteRune(i)
		}
	}
	return res.String()
}