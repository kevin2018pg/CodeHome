// byte表示ASCII字节，go中uint8类型；rune表示Unicode字符（包含中文），go中int32类型，占4个byte
// go 字符串循环，不输出中文及byte形式输出，用下标索引；输出中文，用 for range

func isValid(s string) bool {
    n := len(s)
    if n %2 == 1{
        return false
    }

    stack := make([]byte,0)
    parentheses := map[byte]byte{'(':')', '[':']', '{':'}',}
    for i := 0; i < n; i++  {
        if val,ok := parentheses[s[i]];ok {
            stack = append(stack,val)
        } else {
            if len(stack) == 0 || stack[len(stack)-1] != s[i] {
                return false
            } else {
                stack = stack[:len(stack)-1]
            }
        }
    }
    if len(stack) == 0 {
        return true
    } else {
        return false
    }
}