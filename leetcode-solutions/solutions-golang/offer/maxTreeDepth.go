package offer

//求树的最大深度

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func maxDepth(root *TreeNode) int {
	//终止条件：root为空，越过叶子节点
	if root == nil {
		return 0
	}
	//递推：分别求左右子树的最大深度
	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}
