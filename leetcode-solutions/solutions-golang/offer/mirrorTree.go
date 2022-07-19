package offer

//二叉树镜像
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
//递归方法
func mirrorTree(root *TreeNode) *TreeNode {
	//交换左右子节点
	if root != nil {
		root.Left, root.Right = mirrorTree(root.Right), mirrorTree(root.Left)
	}
	//返回
	return root
}
