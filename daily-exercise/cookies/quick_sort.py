# 快速排序的四种python实现
# https://cloud.tencent.com/developer/article/1571782
import random


# 同向双指针法
class Solution:
    def randomized_partition(self, nums, l, r):
        pivot = random.randint(l, r)
        nums[pivot], nums[r] = nums[r], nums[pivot]
        i = l - 1
        for j in range(l, r):  # j++,j != r
            if nums[j] < nums[r]:
                i += 1  # 当前i已满足条件或刚开始，所以找到合适的要与i下一个位置换位
                nums[j], nums[i] = nums[i], nums[j]
        # j 移动到末尾的情况，nums[j]==nums[r],交换支点到当前位置
        i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    def randomized_quicksort(self, nums, l, r):
        # if r - l <= 0:
        if l >= r:  # l必须<r
            return
        mid = self.randomized_partition(nums, l, r)
        self.randomized_quicksort(nums, l, mid - 1)
        self.randomized_quicksort(nums, mid + 1, r)

    def sortArray(self, nums):
        self.randomized_quicksort(nums, 0, len(nums) - 1)
        return nums


# 内缩头尾双指针
def quick_sort(nums, left, right):
    if left >= right:
        return
    # 可加入随机流程避免最坏情况（完全顺序或逆序）
    # idx = random.randint(left, right)
    # nums[idx], nums[left] = nums[left], nums[idx]
    low, high = left, right
    pivot = nums[low]
    while left < right:
        while left < right and nums[right] > pivot:
            right -= 1
        nums[left] = nums[right]
        while left < right and nums[left] <= pivot:
            left += 1
        nums[right] = nums[left]
    nums[left] = pivot
    quick_sort(nums, low, left - 1)
    quick_sort(nums, left + 1, high)


def sorter(nums):
    left, right = 0, len(nums) - 1
    quick_sort(nums, left, right)


# partition模板写法
def partition(nums, left, right):
    pivot = nums[left]  # 初始化一个待比较数据
    i, j = left, right
    while (i < j):
        while (i < j and nums[j] >= pivot):  # 从后往前查找，直到找到一个比pivot更小的数
            j -= 1
        nums[i] = nums[j]  # 将更小的数放入左边
        while (i < j and nums[i] <= pivot):  # 从前往后找，直到找到一个比pivot更大的数
            i += 1
        nums[j] = nums[i]  # 将更大的数放入右边
    # 循环结束，i与j相等
    nums[i] = pivot  # 待比较数据放入最终位置
    return i  # 返回待比较数据最终位置


# 快速排序
def quicksort(nums, left, right):
    if left < right:
        index = partition(nums, left, right)
        quicksort(nums, left, index - 1)
        quicksort(nums, index + 1, right)
