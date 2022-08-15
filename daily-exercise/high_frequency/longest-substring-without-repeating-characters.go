// 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度
func lengthOfLongestSubstring(s string) int {
    sSet := make(map[byte]int)
    left,right,length := 0,0,0
    for i:= 0;i<len(s);i++ {
        if i!= 0 {
            delete(sSet,s[left])
            left++
        }
        for right < len(s) && sSet[s[right]] == 0 {
            sSet[s[right]] ++
            right++
        }
        length = max(length, right-left)
    }
    return length
}

func max(x,y int) int {
    if x < y {
        return y
    }
    return x
}