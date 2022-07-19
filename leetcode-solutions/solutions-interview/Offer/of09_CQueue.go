/*
 * @Author: your name
 * @Date: 2020-11-18 12:14:09
 * @LastEditTime: 2020-11-18 13:07:40
 * @LastEditors: Please set LastEditors
 * @Description: 剑指 Offer 09. 用两个栈实现队列

   用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

  输入：
  ["CQueue","appendTail","deleteHead","deleteHead"]
  [[],[3],[],[]]
  输出：[null,null,3,-1]

 * @FilePath: \learningflow\Review\OfferSolution\of09_CQueue.go
*/

type CQueue struct {
	// 栈 A，用于添加元素
	stackA []int
	// 栈 B，用于取出元素
	stackB []int
}

// CQueue 的构造函数
func Constructor() CQueue {
	// 返回一个新的 CQueue
	return CQueue{}
}

// 往队列插入整数
func (this *CQueue) AppendTail(value int) {
	// 在 stackA 的末尾追加 value
	this.stackA = append(this.stackA, value)
}

// 从队列取出整数并删除
func (this *CQueue) DeleteHead() int {
	// 如果 stackB 没有元素则从 stackA 中取出所有
	if len(this.stackB) == 0 {
		// 如果 stackA 里也没有元素，则队列爲空返回 -1
		if len(this.stackA) == 0 {
			return -1
		}
		// 将 stackA 的所有元素转移到 stackB
		for len(this.stackA) > 0 {
			// 获取 stackA 最末尾元素的下标
			index := len(this.stackA) - 1
			// 获取 stackA 最末尾元素的值 value
			value := this.stackA[index]
			// 向 stackB 的末尾追加 value
			this.stackB = append(this.stackB, value)
			// 从 stackA 中裁剪出末尾元素
			this.stackA = this.stackA[:index]
		}
	}
	// 这时候表示 stackB 内已有元素
	// 获取 stackB 最末尾元素的下标
	index := len(this.stackB) - 1
	// 获取 stackB 最末尾元素的值 value
	value := this.stackB[index]
	// 从 stackB 中裁剪出末尾元素
	this.stackB = this.stackB[:index]
	// 返回 value
	return value
}

/**
 * Your CQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AppendTail(value);
 * param_2 := obj.DeleteHead();
 */