// 判断子序列，比如ad是否是abcde的子序列
// 双指针法
func isSubsequence(s string, t string) bool {
    n, m := len(s), len(t)
    i, j := 0, 0
    for i < n && j < m {
        if s[i] == t[j] {
            i++
        }
        j++
    }
    return i == n
}
