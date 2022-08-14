//Definition for singly-linked list.
type ListNode struct {
    Val int
    Next *ListNode
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
    // 虚拟头节点
    preHead := &ListNode{Next:head}
    // 左前头节点
    leftBefore := preHead
    for i := 0;i<left-1;i++{
        leftBefore = leftBefore.Next
    }
    // 左右节点
    leftNode,rightNode := leftBefore.Next,leftBefore.Next
    for i:=0;i<right-left;i++{
        rightNode = rightNode.Next
    }
    // 右后节点
    rightAfter := rightNode.Next
    // 切断连接
    leftBefore.Next = nil
    rightNode.Next = nil
    // 翻转局部链表
    reverseLink(leftNode)
    // 建立新连接
    leftBefore.Next = rightNode
    leftNode.Next = rightAfter
    return preHead.Next
}
func reverseLink(head *ListNode) {
    var pre *ListNode
    cur := head
    for cur != nil {
        nex := cur.Next
        cur.Next = pre
        pre, cur = cur, nex
    }
}