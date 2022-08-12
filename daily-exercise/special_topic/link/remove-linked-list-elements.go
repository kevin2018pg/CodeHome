//Definition for singly-linked list.
type ListNode struct {
    Val int
    Next *ListNode
}
func removeElements(head *ListNode, val int) *ListNode {
    preHead := &ListNode{Next:head}
    curNode := preHead
    for curNode.Next != nil {
        if curNode.Next.Val == val {
            curNode.Next = curNode.Next.Next
        } else {
            curNode = curNode.Next
        }
    }
    return preHead.Next
}