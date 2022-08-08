func integerBreak(n int) int {
    dp := make([]int,n+1)
    dp[2] = 1
    for i:=3;i<n+1;i++{
        for j:=1;j<i-1;j++{
            // i可以差分为i-j和j。由于需要最大值，故需要通过j遍历所有存在的值，取其中最大的值作为当前i的最大值，在求最大值的时候，一个是j与i-j相乘，一个是j与dp[i-j].
            // 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
            // 1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
            // 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]
            dp[i] = max(dp[i],max(j*(i-j),j*dp[i-j]))
        }
    }
    return dp[n]
}

func max(a,b int) int {
    if a > b{
        return a
    }
    return b
}