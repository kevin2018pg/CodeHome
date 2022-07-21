// 哈希表存储每个数索引，查找target-当前数的差值
func twoSum(nums []int, target int) []int {
    hashNumIndex := make(map[int]int)
    for idx, num := range nums {
        diff := target - num
		if v, ok := hashNumIndex[diff]; ok {
			return []int{idx, v}
		}
        hashNumIndex[num] = idx
    }
    return nil
}