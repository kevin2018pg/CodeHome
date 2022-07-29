func numIslands(grid [][]byte) int {
    var ans int
    var m,n = len(grid), len(grid[0])
    var dirs = [][]int{{-1,0}, {1,0}, {0,-1}, {0,1}}
    var dfs func(x,y int)
    dfs = func(x,y int){
        grid[x][y] = '0'
        for _,dir := range dirs{
            if x + dir[0] >=0 && x + dir[0] < m && y + dir[1] >=0 && y + dir[1] < n && grid[dir[0]+x][dir[1]+y]=='1'{
                dfs(x+dir[0], y+dir[1])
            }
        }
    }
    for i :=0; i < m; i++{
        for j:=0; j < n; j++{
            if grid[i][j] == '1'{
                ans++
                dfs(i, j)
            }
        }
    }
    return ans
}