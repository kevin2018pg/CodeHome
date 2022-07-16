// 请你设计并实现一个满足LRU (最近最少使用) 缓存 约束的数据结构。
// 实现 LRUCache 类：
// LRUCache(int capacity) 以 正整数 作为容量capacity 初始化 LRU 缓存
// int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
// void put(int key, int value)如果关键字key 已经存在，则变更其数据值value ；如果不存在，则向缓存中插入该组key-value 。如果插入操作导致关键字数量超过capacity ，则应该 逐出 最久未使用的关键字。
// 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

// 输入
// ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
// [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
// 输出
// [null, null, null, 1, null, -1, null, -1, 3, 4]

// 解释
// LRUCache lRUCache = new LRUCache(2);
// lRUCache.put(1, 1); // 缓存是 {1=1}
// lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
// lRUCache.get(1);    // 返回 1
// lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
// lRUCache.get(2);    // 返回 -1 (未找到)
// lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
// lRUCache.get(1);    // 返回 -1 (未找到)
// lRUCache.get(3);    // 返回 3
// lRUCache.get(4);    // 返回 4


type LinkNode struct {
    key int
    val int
    pre *LinkNode
    next *LinkNode
}
func initListNode(key,val int) *LinkNode {
    return &LinkNode{key:key,val:val}
}

type LRUCache struct {
    hashmap map[int]*LinkNode
    capacity int
    head    *LinkNode
    tail    *LinkNode

}


func Constructor(capacity int) LRUCache {
    lru := LRUCache{
        hashmap:make(map[int]*LinkNode),
        capacity:capacity,
        head:initListNode(0,0),
        tail:initListNode(0,0),
    }
    lru.head.next = lru.tail
    lru.tail.pre = lru.head
    return lru
}


func (this *LRUCache) Get(key int) int {
    if _,ok := this.hashmap[key];!ok {
        return -1
    }
    this.moveNodeToTail(key)
    return this.hashmap[key].val
}


func (this *LRUCache) Put(key int, val int)  {
    if _,ok := this.hashmap[key];ok {
        this.hashmap[key].val = val
        this.moveNodeToTail(key)
    } else {
        if len(this.hashmap) == this.capacity {
            delete(this.hashmap, this.head.next.key)
            this.head.next = this.head.next.next
            this.head.next.pre = this.head
        }
        newNode := initListNode(key,val)
        this.hashmap[key] = newNode
        // 建立新节点指向
        newNode.pre = this.tail.pre
        newNode.next = this.tail
        this.tail.pre.next = newNode
        this.tail.pre = newNode
    }
}

func (this *LRUCache) moveNodeToTail(key int)  {
    // 先将节点从hashmap拎出来
    node := this.hashmap[key]
    node.pre.next = node.next
    node.next.pre = node.pre
    // 再将node插入到尾节点之前，建立指向
    node.next = this.tail
    node.pre = this.tail.pre
    this.tail.pre = node
    this.tail.pre.next = node
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */
