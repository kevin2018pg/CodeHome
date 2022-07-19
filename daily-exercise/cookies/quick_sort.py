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
