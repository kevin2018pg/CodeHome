// 环形链表2

type ListNode struct {
    Val int
    Next *ListNode
}

// 哈希表
func detectCycle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    hashNode := make(map[*ListNode]bool)
    for head != nil{
        if hashNode[head] {
            return head
        }
        hashNode[head] = true
        head = head.Next
    }
    return nil
}