//层序遍历二叉树节点
// Definition for a binary tree node.
type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}

func levelOrder(root *TreeNode) [][]int {
    if  root==nil {
        return nil
    }
    resVal := make([][]int,0)
    levelNode := []*TreeNode{root}

    for len(levelNode) != 0 {
        levelVal := make([]int,0)
        newLevelNode := make([]*TreeNode,0)
        for _,node := range levelNode {
            if node == nil {
                continue
            }
            levelVal = append(levelVal,node.Val)
            if node.Left != nil {
                newLevelNode = append(newLevelNode,node.Left)
            }
            if node.Right != nil {
                newLevelNode = append(newLevelNode,node.Right)
            }
        }
        resVal = append(resVal,levelVal)
        levelNode = newLevelNode
    }
       return resVal
}