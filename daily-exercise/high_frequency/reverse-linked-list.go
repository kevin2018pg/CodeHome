# 双指针迭代法，先存再转再后移

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
    var pre *ListNode
    cur := head
    for cur != nil {
        nex = cur.Next
        cur.Next = pre
        pre = cur
        cur = nex
    }
   return pre
}

func reverseListDigui(head *ListNode) *ListNode {
    if head==nil || head.Next==nil {
        return head
    }
    newHead := reverseListDigui(head.Next)
    head.Next.Next = head
    head.Next = nil
   return newHead
}