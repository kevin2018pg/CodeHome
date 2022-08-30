// 分治模板
func Partition(array []int, low int,high int) int {
    pivot := array[low] // 选取支点
    left,right := low,high
    for left < right {  // 从小到大排序
        for left < right && array[right] >= pivot { // 从后往前找到比支点小的数
            right--
        }
        array[left] = array[right]
        for left < right && array[left] < pivot {   // 再从前往后找到比支点大的数
            left ++
        }
        array[right] = array[left]
    }
    array[left] = pivot // 将存储支点放置于左右指针相遇处
    return left
}

// 进行n次划分
func TopKSplit(array []int,low int,high int,k int) {
    pivot := Partition(array,low,high)
    if pivot > k-1 {    // 在支点左侧
        TopKSplit(array,low,pivot-1,k)
    } else if pivot < k-1 { // 在支点右侧
        TopKSplit(array,pivot +1,high,k)
    } else {
        return
    }
}

func findKthLargest(array []int,k int) int {
    length := len(array)
    topK := length - k + 1  // 第k大的数，即为第len-k+1小的数
    TopKSplit(array,0,length-1,topK)
    return array[topK-1]  // 返回下标，需要-1
}