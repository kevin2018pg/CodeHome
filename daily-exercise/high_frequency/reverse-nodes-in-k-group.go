/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseKGroup(head *ListNode, k int) *ListNode {
    preHead := &ListNode{Next: head}    // 创造指向head的一个头节点
    pre := preHead  // 新头节点作为首个尾节点指向第一个sub link

    for head != nil {
        tail := pre // 新建tail节点去循环移动k个位置
        for i := 0; i < k; i++ {
            tail = tail.Next    // 移动k位
            if tail == nil {
                return preHead.Next //  不足k返回头节点next即可，不可以返回头结点
            }
        }
        nex := tail.Next    // 预先存储nex节点，帮助还原
        head, tail = myReverse(head, tail)  // 翻转sub link
        // sub link插入原链表
        pre.Next = head
        tail.Next = nex
        // 重新定义pre和head节点对下一个sub link翻转操作
        pre = tail
        head = tail.Next
    }
    // 返回首头节点的next
    return preHead.Next
}

func myReverse(head, tail *ListNode) (*ListNode, *ListNode) {
    pre,cur := tail.Next,head
    // **不能用while cur，cur到tail节点时，next不为None，又指向tail.next，死循环超时;
    // 因此这时候可以结束循环了,此时pre移动至tail，cur移动至tail.next
    for pre != tail {
        nex := cur.Next
        cur.Next = pre
        pre,cur = cur,nex
    }
    return tail, head
}