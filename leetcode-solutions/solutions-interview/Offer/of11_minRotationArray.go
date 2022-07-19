/*
 * @Author: your name
 * @Date: 2020-11-19 13:17:17
 * @LastEditTime: 2020-11-19 13:57:44
 * @LastEditors: Please set LastEditors
 * @Description: 剑指 Offer 11. 旋转数组的最小数字

	把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。

	示例 1：

	输入：[3,4,5,1,2]
	输出：1

 * @FilePath: \learningflow\Review\OfferSolution\of11_minRotationArray.go
*/

// 寻找数组最小旋转点	-二分查找方法
func minArray(numbers []int) int {
	//确定上下确界
	low := 0
	high := len(numbers) - 1
	// 当上下界相同，二分查找结束
	for low < high {
		pivot := low + (high-low)/2         // 二分点pivot，防止溢出的写法
		if numbers[pivot] > numbers[high] { // pivot大于上界，旋转点在m右侧
			low = pivot + 1
		} else if numbers[pivot] < numbers[high] { // pivot小于上确界，旋转点在m左侧，包括自身（pivot可能是旋转点）
			high = pivot
		} else { // 相等无法判断，可忽略右端点，缩小上确界范围
			high--
		}
	}
	return numbers[low]
}