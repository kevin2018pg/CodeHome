// Definition for singly-linked list.
type ListNode struct {
    Val int
    Next *ListNode
}
// 哈希表：此题等同于判断链表是否有环；存下A链表节点，判断B节点是否存在
func getIntersectionNodeHash(headA, headB *ListNode) *ListNode {
    hashNode := make(map[*ListNode]bool)
    for headA != nil {
        hashNode[headA] = true
        headA = headA.Next
    }
    for headB != nil {
        if hashNode[headB] {
            return headB
        }
        headB = headB.Next
    }
    return nil
}

// 双指针：尾部相交代表在尾部对齐的情况下，同样位置遍历会遇到相同的则有相交
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    if headA == nil || headB == nil{
        return nil
    }
    // 求链表长度
    lengthA,lengthB,diff := 0,0,0
    tmpNodeA, tmpNodeB := headA,headB
    for tmpNodeA != nil {
        lengthA++
        tmpNodeA = tmpNodeA.Next
    }
    for tmpNodeB != nil {
        lengthB++
        tmpNodeB = tmpNodeB.Next
    }
    // 确定谁更长谁是头
    if lengthA > lengthB{
        diff = lengthA -lengthB
        tmpNodeA,tmpNodeB = headA,headB
    }else {
        diff = lengthB-lengthA
        tmpNodeA,tmpNodeB = headB,headA
    }
    // 尾部对齐
    for i:=0;i<diff;i++{
        tmpNodeA = tmpNodeA.Next
    }
    // 同样位置遍历，求相同
    for tmpNodeA != tmpNodeB {
        tmpNodeA = tmpNodeA.Next
        tmpNodeB = tmpNodeB.Next
    }
    return tmpNodeA
}