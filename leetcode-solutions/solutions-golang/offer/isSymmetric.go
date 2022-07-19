package offer

//判断对称的二叉树

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isSymmetric(root *TreeNode) bool {
	//根节点为空，直接返回true
	if root == nil {
		return true
	}
	//递归函数调用
	return helper(root.Left, root.Right)

}

func helper(L, R *TreeNode) bool {
	//终止条件：
	//L和R同时为空，代表同时越过叶节点，此树从顶至底节点都对称返回true
	if L == nil && R == nil {
		return true
	}
	//L和R中只有一个越过叶节点，或者节点L值不等于R值，此树不对称返回false
	if L == nil || R == nil || L.Val != R.Val {
		return false
	}
	//递推：判断L左节点和R右节点，L右节点和R左节点是否对称，两节点都对称时，才是对称树，用and连接
	return helper(L.Left, R.Right) && helper(L.Right, R.Left)
}
