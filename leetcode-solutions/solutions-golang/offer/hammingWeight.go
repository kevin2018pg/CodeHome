package offer

//数n的二进制数中1的个数
import (
	"fmt"
	"strings"
)

//方法1：转二进制计数
func hammingWeight_s1(num uint32) int {
	return strings.Count(fmt.Sprintf("%b", num), "1")
}

//法2：转二进制计数
func hammingWeight_s2(num uint32) int {
	n := 0
	nums := fmt.Sprintf("%b", num)
	for _, char := range nums {
		if char == '1' {
			n++
		}
	}
	return n
}

//法3：位运算
// 位运算，逐位判断
// n & 1 = 0，则n二进制最后一位为0
// n & 1 = 1，则n二进制最后一位为1
func hammingWeight_s3(num uint32) int {
	res := 0
	for num > 0 {
		// num % 2
		if num&1 == 1 {
			res++
		}
		// num /= 2
		num >>= 1
	}
	return res
}
