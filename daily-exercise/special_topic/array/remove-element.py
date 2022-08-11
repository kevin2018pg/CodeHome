class Solution:
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


test = Solution()
index = test.removeElement([0, 1, 2, 2, 3, 0, 4, 2], 2)
