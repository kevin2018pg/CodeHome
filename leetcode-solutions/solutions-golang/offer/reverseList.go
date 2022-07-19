package offer

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	//空链表或者只有一个节点，返回自身
	if head == nil || head.Next == nil {
		return head
	}
	// pre等于none
	var pre *ListNode
	cur := head
	for cur != nil {
		next := cur.Next
		//当前节点反转，cur指向pre
		cur.Next = pre
		//将两个节点往后移动一位
		pre = cur
		cur = next
	}
	//cur为空即cur.next是空，pre指向最后一个节点，返回pre
	return pre
}
