/*
 * @Author: your name
 * @Date: 2020-11-10 13:36:59
 * @LastEditTime: 2020-11-10 23:39:27
 * @LastEditors: Please set LastEditors
 * @Description: 剑指 Offer 03. 找出数组中重复的数字

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3

 * @FilePath: \learningflow\Review\OfferSolution\of03_findRepeatNumber.go
*/

Package offer_excise
import "sort"


//1、哈希方法
func findRepeatNumber_01(nums []int) int {
	num_map := make(map[int]bool)
	for _, num := range nums {
		if num_map[num] {
			return num
		} else {
			num_map[num] = true
		}
	}
	return -1
}

//2、排序法
func findRepeatNumber_02(nums []int) int {
	sort.Ints(nums)
	last_num := nums[0]
	for i:= 1;i <= len(nums);i++{
		if last_num == nums[i] {
			return last_num
		} else{
			last_num = nums[i]
		}
	}
	return -1
}

func main() {

}