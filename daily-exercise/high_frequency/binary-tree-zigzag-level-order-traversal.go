//Definition for a binary tree node.
//给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）
type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}

func zigzagLevelOrder(root *TreeNode) [][]int {
    if root == nil {    // 根节点不存在
        return nil
    }
    resValue := make([][]int,0) // 最终返回结果集合
    levelNode := []*TreeNode{root}  //层节点
    levelNum := 0   //层计数
    for len(levelNode) != 0 {   //层一直存在
        newLevelNode := make([]*TreeNode,0) //新变量赋值给当前层变量，不需要双端队列实现
        levelValue := make([]int,0) //层值
        for _, node := range levelNode {    //遍历层节点
            if node == nil {
                continue
            }
            levelValue = append(levelValue,node.Val)    //值加入
            if node.Left != nil {
                newLevelNode = append(newLevelNode,node.Left)   // 加入左子树
            }
            if node.Right != nil {
                newLevelNode = append(newLevelNode, node.Right) //加入右子树
            }
        }
        levelNode = newLevelNode    // 赋给当前层
        // 奇数层需要反转层级值切片
        if levelNum % 2 == 1 {  // 奇数层，从右往左
            reverseLevelValue := reverse(levelValue)    // 翻转层值
            resValue = append(resValue, reverseLevelValue)
        } else {
            resValue = append(resValue, levelValue) //偶数层保持原先从左到右
        }
        levelNum++
    }
    return resValue
}
//翻转方法，取中间位索引，交换元素，省去一半时间
func reverse(s []int) []int {
    a := make([]int, len(s))
    copy(a, s)

    for i := len(a)/2 - 1; i >= 0; i-- {
        opp := len(a) - 1 - i
        a[i], a[opp] = a[opp], a[i]
    }
    return a
}