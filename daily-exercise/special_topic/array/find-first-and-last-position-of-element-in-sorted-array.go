func searchRange(nums []int, target int) []int {
    //先二分查找，找不到返回 [-1,-1
    index := binarySearch(nums,target)
    if index == -1 {
        return []int{-1,-1}
    }
    //再往两侧循环查找，更新起止点
    left, right := index,index
    for i:=left-1;i>=0;i-- {
        if nums[i] == target{
            left = i
        }
    }
    for i:=right+1;i<len(nums);i++ {
        if nums[i] == target{
            right = i
        }
    }
    return []int{left,right}
}

func binarySearch(nums []int, target int) int {
    left, right := 0, len(nums)-1
    for left <= right {
        mid := left + (right-left)/2
        if nums[mid] > target {
            right = mid -1
        } else if nums[mid] < target{
            left = mid +1
        } else {
            return mid
        }
    }
    return -1
}