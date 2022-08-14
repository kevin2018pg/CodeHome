type ListNode struct {
    Val int
    Next *ListNode
}

// 第一种方法，先统计链表长度，找出倒数第n个位置，
func removeNthFromEnd(head *ListNode, n int) *ListNode{
    if head == nil {
        return nil
    }
    // 统计链表节点长度
    length := 0
    tmpHead := head
    for tmpHead != nil {
        length += 1
        tmpHead = tmpHead.Next
    }
    // 找出删除位置
    delIndex := length - n + 1
    // 创建虚拟头，删除返回
    preHead := &ListNode{Next:head}
    curNode := preHead
    count := 0
    for curNode != nil {
        count++
        if count == delIndex {
            curNode.Next = curNode.Next.Next
        }
        curNode = curNode.Next
    }
    return preHead.Next
}

// 第二种方法：双指针
// 双指针的思路，用两个指针确定出n的位置，那么两个指针之间的距离是n，一个在末尾，一个在倒数第n个的前一步，所以：让快指针先走n步慢指针再开始走，这样当快指针走完的时候，慢指针下一步便是倒数第n个节点。
func removeNthFromEnd2(head *ListNode, n int) *ListNode{
    preHead := &ListNode{Next:head}
    slow,fast := preHead,preHead
    count := 0
    for fast != nil {
        fast = fast.Next
        count++
        if count >= n {
            slow = slow.Next
            slow.Next = slow.Next.Next
        }
    }
    return preHead.Next
}