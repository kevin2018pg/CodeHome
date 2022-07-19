package offer

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
//递归法
func reversePrint_s1(head *ListNode) []int {
	if head == nil {
		return []int{}
	} else {
		return append(reversePrint_s1(head.Next), head.Val)
	}

}

//辅助栈法（双百）
func reversePrint_s2(head *ListNode) []int {
	stack := []int{}
	for head != nil {
		stack = append(stack, head.Val)
		head = head.Next
	}
	res := []int{}
	for i := len(stack) - 1; i >= 0; i-- {
		res = append(res, stack[i])
	}
	return res
}

//（节省一个数组的内存空间，双百）
func reversePrint_s3(head *ListNode) []int {
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
