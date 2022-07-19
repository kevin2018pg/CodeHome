/*
 * @Author: your name
 * @Date: 2020-11-12 13:26:56
 * @LastEditTime: 2020-11-12 13:46:33
 * @LastEditors: Please set LastEditors
 * @Description: 剑指 Offer 06. 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
输入：head = [1,3,2]
输出：[2,3,1]

 * @FilePath: \learningflow\Review\OfferSolution\of06_reversePrint.go
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
//  1、辅助栈法：压入栈再出栈
func reversePrint_01(head *ListNode) []int {
	stack := []int{}
	for head != nil {
		stack = append(stack, head.Val)
		head = head.Next
	}
	res := []int{}
	length := len(stack)
	for i := length - 1; i >= 0; i-- {
		res = append(res, stack[i])
	}
	return res
}

// 1.1、辅助栈法的优化，体现在反转数组
func reversePrint_01_01(head *ListNode) []int {
	stack := []int{}
	for head != nil {
		stack = append(stack, head.Val)
		head = head.Next
	}
	length := len(stack)
	for i := 0; i < length/2; i++ {
		stack[i], stack[length-1-i] = stack[length-1-i], stack[i]
	}
	return stack
}

// 2、递归法：每次传入next走到末端，以head=none（链表尾部）为递归终止条件，此时返回空列表。递归回溯时依次将节点值加入列表，当前 list + 当前节点值 [head.val]合并 list，尾部返回 [], 实现节点倒序输出
func reversePrint_01(head *ListNode) []int {
	if head == nil {
		return []int{}
	} else {
		return append(reversePrint_01(head.Next), head.Val)
	}
}