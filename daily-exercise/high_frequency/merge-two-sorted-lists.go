/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    preHead := &ListNode{Val:0} // 创建超头节点
    tmpNode := preHead  // 声明移动节点
    for list1 != nil && list2 != nil {  // 条件是l1和l2都没有结束，其中一个结束链表剩余部分可以直接接上已合并链表
        if list1.Val <= list2.Val { // 比较节点值大小，有序
            tmpNode.Next = list1    // 指向
            list1,tmpNode = list1.Next,tmpNode.Next  // 移动list，废除已并入节点；移动tmpNode;
        } else {
            tmpNode.Next = list2
            list2,tmpNode = list2.Next,tmpNode.Next
        }
        //tmpNode = tmpNode.Next
    }
    if list1 != nil {   // l2已结束，直接接上l1，反之
        tmpNode.Next = list1
    } else {
        tmpNode.Next = list2
    }
    return preHead.Next // 返回超头节点next，即头节点
}