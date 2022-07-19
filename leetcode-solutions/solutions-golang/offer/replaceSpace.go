package offer

import "strings"

func replaceSpace_s1(s string) string {
	var str string = ""
	for _, v := range s {
		//if v == ' ' {
		if v == 32 {
			str += "%20"
		} else {
			str += string(v)
		}
	}
	return str
}

func replaceSpace_s2(s string) string {
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

func replaceSpace_s3(s string) string {
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
