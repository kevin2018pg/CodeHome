func reverseString(s []byte)  {
    left,right := 0,len(s)-1
    for left < right{
        s[left],s[right] = s[right],s[left]
        right--
        left++
    }
}

func reverseString2(s []byte)  {
    length := len(s)
    mid := length/2-1
    for i:=mid;i>=0;i--{
        j := length-i-1
        s[i],s[j] = s[j],s[i]
    }
}

