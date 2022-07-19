package offer

//Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

//根据题目描述,链表l1,l2是递增的，因此容易想到使用双指针l1和l2遍历两链表，
//根据 l1.val和 l2.val的大小关系确定节点添加顺序，两节点指针交替前进，直至遍历完毕

//初始状态合并列表中没有节点，因此循环第一轮时无法将节点添加到合并链表中，
//所以引入伪头节点作为辅助节点，将各个几点加到其后

//伪头结点，迭代方法
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	//初始值设不设置都可以，不设置默认为空
	//var dum = &ListNode{0,nil}
	dum := &ListNode{}
	cur := dum
	//l1或l2为空时跳出
	for l1 != nil && l2 != nil {
		//l1<l2，节点cur指向l1，并且l1向前走一步
		if l1.Val < l2.Val {
			cur.Next = l1
			l1 = l1.Next
		} else { //l1>=l2，节点cur指向l2，并且l2向前走一步
			cur.Next = l2
			l2 = l2.Next
		}
		//移动cur,cur节点向前走一步
		cur = cur.Next
	}
	//当跳出时,l1不为空就全部加到cur的后面,如果l1为空,就把l2加到cur后面
	if l1 == nil {
		cur.Next = l2
	} else {
		cur.Next = l1
	}
	//返回dum的后继节点
	return dum.Next
}
