func reverseStr(s string, k int) string {
    textArray := []byte(s)
    for i:=0;i<len(textArray);i+=2*k{
        if i+k <= len(textArray){
            ReversePart(textArray[i:i+k])
        }else{
            ReversePart(textArray[i:len(textArray)])
        }
    }
    return string(textArray)
}

func ReversePart(textArray []byte) {
    left,right := 0,len(textArray)-1
    for left < right{
        textArray[left],textArray[right] = textArray[right],textArray[left]
        right--
        left++
    }
}