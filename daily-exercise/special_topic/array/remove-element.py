class Solution:
    # 头尾双指针
    def removeElement(self, nums, val):
        left, right = 0, len(nums) - 1
        while left <= right:
            if nums[left] == val:
                nums[left] = nums[right]
                right -= 1
            else:
                left += 1
        print(nums, left, right)
        return left


# 快慢指针
def removeElement(nums, val):
    slow = 0
    for f in range(len(nums)):
        if nums[f] != val:
            nums[slow] = nums[f]
            slow += 1


test = Solution()
index = test.removeElement([0, 1, 2, 2, 3, 0, 4, 2], 2)
