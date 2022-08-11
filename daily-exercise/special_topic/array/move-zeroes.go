func moveZeroes(nums []int)  {
    n := len(nums)
    slow := 0
    for fast :=0;fast< n;fast++{
        if nums[fast] != 0{
            nums[slow],nums[fast] = nums[fast],nums[slow]
            slow++
        }
    }
}