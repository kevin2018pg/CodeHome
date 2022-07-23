/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func hasCycle(head *ListNode) bool {
    hashLink := make(map[*ListNode]bool)
    for head != nil {
        if value, ok := hashLink[head];ok {
            return value
        }
        hashLink[head] = true
        head = head.Next
    }
    return false
}