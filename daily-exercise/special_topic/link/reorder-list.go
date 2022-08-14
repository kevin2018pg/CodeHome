// 用线性表存储起来，头尾访问即可
func reorderList(head *ListNode) {
    if head == nil {
        return
    }
    nodes := make([]*ListNode,0)
    for node := head; node != nil; node = node.Next {
        nodes = append(nodes, node)
    }
    i, j := 0, len(nodes)-1
    for i < j {
        nodes[i].Next = nodes[j]
        i++
        if i == j {
            break
        }
        nodes[j].Next = nodes[i]
        j--
    }
    nodes[i].Next = nil
}