// 快速排序
func quickSort(left, right int, nums []int) {
    if left >= right {
        return
    }
    low, high := left, right
    pivot := nums[low]
    for left < right {
        for left < right && nums[right] > pivot {
            right --
        }
        nums[left] = nums[right]
        for left < right && nums[left] <= pivot {
            left ++
        }
        nums[right] = nums[left]
    }
    nums[left] = pivot
    quickSort(left+1,high,nums)
    quickSort(low,right-1,nums)
}

func threeSum(nums []int) [][]int {
    n := len(nums)
	if nums == nil || n < 3 {
		return nil
	}
	// 排序
	//sort.Ints(nums)
	left,right := 0,n-1
    quickSort(left,right,nums)

    //三指针
	targetArray := make([][]int,0)
    for point := 0;point < (len(nums)-2);point++ {
        if nums[point] > 0 {
            break
        }
        if point > 0  && nums[point] == nums[point-1] {
            continue
        }
        //双指针
        head,tail := point+1 ,n-1
        for head < tail {
            numSum := nums[point]+nums[head]+nums[tail]
            if numSum < 0 {
                head ++
                for head < tail && nums[head] == nums[head-1] {
                    head ++
                }
            } else if numSum > 0{
                tail --
                for head < tail && nums[tail] == nums[tail+1] {
                    tail --
                }
            } else {
                tmpArray := []int{nums[point], nums[head], nums[tail]}
                targetArray = append(targetArray, tmpArray)
                head ++
                tail --
                for head < tail && nums[head] == nums[head-1] {
                    head ++
                }
                for head < tail && nums[tail] == nums[tail+1] {
                    tail --
                }
            }
        }
    }
    return targetArray
}