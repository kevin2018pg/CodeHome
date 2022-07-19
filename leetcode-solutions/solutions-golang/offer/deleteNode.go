package offer

//删除链表节点
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

//双指针后移
func deleteNode(head *ListNode, val int) *ListNode {
	if head.Val == val {
		return head.Next
	}
	pre, cur := head, head.Next
	for cur != nil && cur.Val != val {
		pre, cur = cur, cur.Next
	}
	if cur != nil {
		pre.Next = cur.Next
	}
	return head
}

//伪节点单指针循环判断
func deleteNode(head *ListNode, val int) *ListNode {
	dummy := ListNode{}
	dummy.Next = head
	if head.Val == val {
		return head.Next
	}
	for head != nil && head.Next != nil {
		if head.Next.Val == val {
			head.Next = head.Next.Next
		}
		head = head.Next
	}
	return dummy.Next
}
