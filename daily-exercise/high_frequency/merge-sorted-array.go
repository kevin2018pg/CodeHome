// 双指针：创建m+n容量的切片，copy进去，两个指针其中之一结束时，直接把后面加进来
func merge(nums1 []int, m int, nums2 []int, n int)  {
    up, down := 0,0
    newSlice := make([]int,0,m+n)
    for up < m || down < n{
        if up == m {
            newSlice = append(newSlice,nums2[down:]...)
            break
        }
        if down == n {
            newSlice = append(newSlice,nums1[up:]...)
            break
        }
        if nums1[up] < nums2[down]{
            newSlice = append(newSlice,nums1[up])
            up++
        } else {
            newSlice = append(newSlice,nums2[down])
            down++
        }
    }
    copy(nums1,newSlice)
}


func merge_reverse(nums1 []int, m int, nums2 []int, n int)  {
    for up,down,p := m-1, n-1, m+n-1;up>=0||down>=0;p--{
        var cur int
        if up == -1{
            cur = nums2[down]
            down--
        }else if down == -1 {
            cur = nums1[up]
            up--
        }else if nums1[up] > nums2[down]{
            cur = nums1[up]
            up--
        }else {
            cur = nums2[down]
            down--
        }
        nums1[p] = cur
    }
}