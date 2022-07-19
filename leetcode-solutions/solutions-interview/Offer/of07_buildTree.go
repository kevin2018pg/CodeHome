/* 
Description: 剑指 Offer 07. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如给出：
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7
*/



/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	// 根节点
	root := &TreeNode{Val: preorder[0]}
	var index int
	for i := range inorder {
		if inorder[i] == preorder[0] {
			index = i
			break
		}
	}
	root.Left = buildTree(preorder[1:index+1],inorder[:index])
	root.Right = buildTree(preorder[],inorder[index+1:])
	return root
}