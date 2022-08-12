type ListNode struct {
    Val int
    Next *ListNode
}

func swapPairs(head *ListNode) *ListNode {
    // 翻转两个局部链表节点需要用到三个节点，因为需要更新前后的指向，所以创建一个虚拟头，更新虚拟头
    preHead := &ListNode{Next:head}
    before := preHead
    for before.Next != nil && before.Next.Next != nil {
        // 三节点：前中后
        middle := before.Next
        after := before.Next.Next
        // 开始翻转（先翻转中后再改变前的指向）
        middle.Next = after.Next
        after.Next = middle
        before.Next = after
        // 更新前节点（虚拟头）
        before = before.Next.Next
    }
    return preHead.Next
}