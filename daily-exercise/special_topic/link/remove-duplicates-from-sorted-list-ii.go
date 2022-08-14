// 和删除重复元素1区别就是这个要删除所有，循环判断相等时要再循环删除
//Definition for singly-linked list.
type ListNode struct {
    Val int
    Next *ListNode
}
func deleteDuplicates(head *ListNode) *ListNode {
    preHead := &ListNode{Next:head}
    cur := preHead
    for cur.Next != nil && cur.Next.Next != nil{
        if cur.Next.Val == cur.Next.Next.Val{
            x := cur.Next.Val
            for cur.Next != nil && cur.Next.Val == x{
                cur.Next = cur.Next.Next
            }
        }else {
            cur = cur.Next
        }
    }
    return preHead.Next
}