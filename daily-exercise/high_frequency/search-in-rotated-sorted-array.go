func search(nums []int, target int) int {
    if len(nums) == 0{
        return -1
    }

    left, right := 0,len(nums)-1
    for left <= right {
        mid := (right - left) / 2 + left
        if nums[mid] == target {
            return mid
        }
        if nums[mid] >= nums[left] {    // 左边有序
            if nums[mid] > target && target >= nums[left] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            if nums[mid] < target && target <= nums[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return -1
}