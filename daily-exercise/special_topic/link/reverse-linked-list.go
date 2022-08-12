//Definition for singly-linked list.
type ListNode struct {
    Val int
    Next *ListNode
}

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