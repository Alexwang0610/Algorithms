package com.algorithms;

import java.util.*;

public class HelloWorld {
    // 用来标记上下左右的方向
    private int[][] direction = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };

    /**
     * 删除数组中重复的元素
     *
     * @param nums 需要删除重复元素的数组
     * @return 返回值为数组的长度
     */
    public static int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n == 0)
            return 0;
        int slow = 0;
        int fast = 1;

        while (fast < n) {
            if (nums[slow] == nums[fast]) {
                fast++;
            } else {
                nums[++slow] = nums[fast++];
            }
        }
        return slow + 1;
    }

    /**
     * 最简路径
     *
     * @param path
     * @return
     */
    public static String simplifyPath(String path) {
        Deque<String> stack = new LinkedList<>();// 用栈来模拟
        String[] strArr = path.split("/");// 用/分割
        for (String str : strArr) {// 遍历
            // 如果等于空或者等于.，那就没有影响
            if (str.equals("") || str.equals(".")) {
                continue;
            }
            // 如果等于..，那就要返回上一级目录，因此栈中弹出当前目录
            // 此时可能栈是空
            if (str.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
                continue;
            }
            // 否则，栈中压入当前目录
            stack.push(str);
        }
        StringBuffer sb = new StringBuffer();
        while (!stack.isEmpty()) {
            String tmp = stack.removeLast();
            sb.append("/").append(tmp);
        }
        if (sb.length() == 0) {
            sb.append("/");
        }
        String res = sb.toString();
        return res;
    }

    /**
     * 最接近的三数之和,找出 nums 中的三个整数，使得它们的和与 target 最接近
     *
     * @param nums
     * @param target
     * @return 返回三个数的和
     */
    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int min = Integer.MAX_VALUE;
        for (int k = 0; k < nums.length; k++) {
            // 跳过与k位置相等的值
            if (k > 0 && nums[k] == nums[k - 1])
                continue;
            int i = k + 1, j = nums.length - 1;
            while (i < j) {
                int sum = nums[i] + nums[j] + nums[k];
                // 如果三数之和与target相等，直接返回，最接近就是0
                if (sum == target) {
                    return sum;
                }
                // 如果当前的和更接近target，最接近的距离等于sum
                if (Math.abs(sum - target) < Math.abs(min - target)) {
                    min = sum;
                }

                if (sum > target) {
                    int index = j - 1;
                    while (i < index && nums[j] == nums[index]) {
                        index--;
                    }
                    j = index;
                } else {
                    int index = i + 1;
                    while (index < j && nums[i] == nums[index]) {
                        index++;
                    }
                    i = index;
                }
            }
        }
        return min;
    }

    /**
     * 有效的异位词，如果一个两个字符串包含的字母数量相同，但是位置不同，则两个字符串是异位词
     *
     * @param s
     * @param t
     * @return
     */
    public static boolean isAnagram(String s, String t) {
        HashMap<Character, Integer> mapS = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (mapS.containsKey(ch)) {
                mapS.put(ch, mapS.get(ch) + 1);
            } else {
                mapS.put(ch, 1);
            }
        }

        for (int i = 0; i < t.length(); i++) {
            char ch = t.charAt(i);
            if (mapS.containsKey(ch)) {
                mapS.put(ch, mapS.get(ch) - 1);
            } else {
                return false;
            }
        }

        for (Character key : mapS.keySet()) {
            if (mapS.get(key) != 0)
                return false;
        }
        return true;
    }

    /**
     * 字符串匹配，返回字符开始的下标
     *
     * @param haystack
     * @param needle
     * @return
     */
    public static int strStr(String haystack, String needle) {
        if (needle.length() == 0)
            return 0;
        int i = 0, j = 0;
        int index = -1;
        while (i < haystack.length() && j < needle.length()) {
            if (haystack.charAt(i) == needle.charAt(j)) {
                i++;
                j++;
            } else {
                i++;
                j = 0;
            }
        }
        return j == 0 ? index : i - needle.length();
    }

    /**
     * 输出指定位置的丑数
     *
     * @param n 指定的位置
     * @return
     */
    public static int nthUglyNumber(int n) {
        if (n == 1)
            return 1;
        // 记录个数
        int count = 0;
        int num = 1;
        for (; count < n; num++) {
            // 临时保存num
            int temp = num;
            // 判断是否为丑数
            for (int i = 2; i < 6 && temp > 0; i++) {
                while (temp % i == 0) {
                    temp /= i;
                }
            }
            // 如果是丑数，数量+1
            if (temp == 1) {
                count++;
            }
        }
        return num - 1;
    }

    /**
     * 存在重复元素
     *
     * @param nums 数组
     * @param k    abs(nums[i] - nums[j]) <= k
     * @param t    abs(i - j) <= t
     * @return
     */
    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int n = nums.length;
        if (n < 2) {
            return false;
        }
        long temp = t;

        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                long absNum = Math.abs(nums[i] - nums[j]);
                int absIdx = Math.abs(i - j);
                System.out.println("absNum->" + absNum + ";absIdx->" + absIdx);
                if (absNum <= temp && absIdx <= k)
                    return true;
            }
        }
        return false;
    }

    /**
     * 电话号码字母组合
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        HashMap<Character, String> letterMap = new HashMap<>();
        letterMap.put('2', "abc");
        letterMap.put('3', "def");
        letterMap.put('4', "ghi");
        letterMap.put('5', "gkl");
        letterMap.put('6', "mno");
        letterMap.put('7', "pqrs");
        letterMap.put('8', "tuv");
        letterMap.put('9', "wxyz");
        return null;
    }

    /**
     * 解码数字
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        // 建立密码表
        HashMap<Character, Integer> decodeMap = new HashMap<>();
        for (int i = 1; i < 27; i++) {
            char key = (char) (i + 64);
            decodeMap.put(key, i);
        }
        int n = s.length();
        int[] f = new int[n + 1];
        f[0] = 1;
        for (int i = 1; i <= n; ++i) {
            if (s.charAt(i - 1) != '0') {
                f[i] += f[i - 1];
            }
            if (i > 1 && s.charAt(i - 2) != '0' && ((s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0') <= 26)) {
                f[i] += f[i - 2];
            }
        }
        return f[n];
    }

    /**
     * 最大整除子集
     *
     * @param nums
     * @return
     */
    public List<Integer> largestDivisibleSubset(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int[] f = new int[n];
        int[] g = new int[n];
        for (int i = 0; i < n; i++) {
            // 至少包含自身一个数，因此起始长度为 1，由自身转移而来
            int len = 1, prev = i;
            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0) {
                    // 如果能接在更长的序列后面，则更新「最大长度」&「从何转移而来」
                    if (f[j] + 1 > len) {
                        len = f[j] + 1;
                        prev = j;
                    }
                }
            }
            // 记录「最终长度」&「从何转移而来」
            f[i] = len;
            g[i] = prev;
        }

        // 遍历所有的 f[i]，取得「最大长度」和「对应下标」
        int max = -1, idx = -1;
        for (int i = 0; i < n; i++) {
            if (f[i] > max) {
                idx = i;
                max = f[i];
            }
        }

        // 使用 g[] 数组回溯出具体方案
        List<Integer> ans = new ArrayList<>();
        while (ans.size() != max) {
            ans.add(nums[idx]);
            idx = g[idx];
        }
        return ans;
    }

    /**
     * 生成括号 left 随时可以加，只要不超标 right 左括号个数>右括号个数
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        _generator(0, 0, n, "", result);
        return result;
    }

    private void _generator(int left, int right, int n, String target, List res) {
        // terminator
        if (left == n && right == n) {
            res.add(target);
            return;
        }

        // process

        // drill down
        if (left < n) {
            _generator(left + 1, right, n, target + "(", res);
        }
        if (left > right)
            _generator(left, right + 1, n, target + ")", res);

        // reverse states
    }

    /**
     * 四数之和
     *
     * @param nums   给定数组
     * @param target 数组中四个数的和
     * @return
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            for (int j = i + 1; j < nums.length - 2; j++) {
                for (int k = j + 1, pivot = nums.length - 1; k < nums.length - 1; k++) {
                    List<Integer> tempList = new ArrayList<>();
                    int sum = nums[pivot] + nums[i] + nums[j] + nums[k];
                    System.out.println(sum);
                    if (sum > target) {
                        pivot--;
                    } else if (sum < target) {
                        continue;
                    } else {
                        tempList.add(nums[i]);
                        tempList.add(nums[j]);
                        tempList.add(nums[k]);
                        tempList.add(nums[pivot]);
                        result.add(tempList);
                        break;
                    }
                }
            }
        }
        return result;
    }

    /**
     * 全排列
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> resultList = new ArrayList<>();
        int n = nums.length;
        if (n == 0) {
            return resultList;
        }

        Deque<Integer> path = new ArrayDeque<>();
        boolean[] used = new boolean[n];

        _dfs(nums, n, 0, path, used, resultList);
        return resultList;
    }

    /**
     * 递归遍历
     *
     * @param nums       整个数组
     * @param len        数组的长度
     * @param depth      当前遍历的层数，也可以理解为当前已经选择数字的数量
     * @param path       已经选择的数
     * @param used       数字是否使用过的状态保存
     * @param resultList 结果列表
     */
    private void _dfs(int[] nums, int len, int depth, Deque<Integer> path, boolean[] used,
            List<List<Integer>> resultList) {
        // 如果已经选择的数量和数组的长度相等，说明已经选择结束
        if (depth == len) {
            resultList.add(new ArrayList<>(path));
        }

        // 遍历数组
        for (int i = 0; i < nums.length; i++) {
            // 如果当前数字使用过，继续循环
            if (used[i]) {
                continue;
            }
            // 如果没有使用过的数字，加入到栈里
            path.addLast(nums[i]);
            // 把当前数字的使用状态设为true
            used[i] = true;
            // 继续递归
            _dfs(nums, len, depth + 1, path, used, resultList);
            // 选择完成之后把当前数移除
            path.removeLast();
            // 使用的状态设为false
            used[i] = false;
        }
    }

    /**
     * 在D天内送达包裹的能力，返回最低的运载重量
     *
     * @param weights 包裹重量的数组
     * @param D       需要的天数
     * @return
     */
    public int shipWithinDays(int[] weights, int D) {
        return 0;
    }

    /**
     * 只出现一次的数字
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int single = 0;
        for (int num : nums) {
            single ^= num;
        }
        return single;
    }

    /**
     * 买卖股票的最佳时机
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int n = prices.length;
        if (n == 1)
            return 0;
        int minPrice = Integer.MAX_VALUE;
        int max = 0;
        for (int i = 0; i < n; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else if (prices[i] - minPrice > 0) {
                max = Math.max(prices[i] - minPrice, max);
            }
        }
        return max;
    }

    /**
     * 下一个排列，字典序的下一个排列，如果没有选择最小的排列
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        int len = nums.length;
        for (int i = len - 1; i > 0; i--) {
            // 从后向前遍历，如果发现不是降序
            if (nums[i] > nums[i - 1]) {
                // 把从i之后的数组进行升序排列
                Arrays.sort(nums, i, len);
                // 从重新排序的部分找出一个大于i-1的数，进行交换
                for (int j = i; j < len; j++) {
                    if (nums[j] > nums[i - 1]) {
                        int temp = nums[j];
                        nums[j] = nums[i - 1];
                        nums[i - 1] = temp;
                        return;
                    }
                }
            }
        }
        Arrays.sort(nums);
        return;
    }

    /**
     * 平方数之和
     *
     * @param c
     * @return
     */
    public boolean judgeSquareSum(int c) {
        // 如果定义成int 平方之后可能会造成越界
        long a = 0;
        long b = (long) Math.sqrt(c);

        while (a <= b) {
            long sum = a * a + b * b;
            // 1.如果平方和等于c，说明存在
            // 2.如果平方和大于c，说明b需要缩小
            // 3.如果平方和小于c，说明a需要增大
            if (sum == c) {
                return true;
            } else if (sum > c) {
                b--;
            } else {
                a++;
            }
        }
        return false;
    }

    /**
     * 跳跃游戏 给定一个非负整数数组nums ，你最初位于数组的 第一个下标 。
     * <p>
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * <p>
     * 判断你是否能够到达最后一个下标。
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        // 最后一格肯定是可以跳到最后一个
        int jump = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            // 如果当前的格子可以跳到jump，jump等与当前格子下标
            if (nums[i] + i >= jump) {
                jump = i;
            }
        }
        // 最后判断格子的下标是不是在数组的开头
        return jump == 0;
    }

    /**
     * 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
     * <p>
     * 如果数组中不存在目标值 target，返回[-1, -1]。
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        int[] res = { -1, -1 };
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                while (mid >= 0 && nums[mid] == target) {
                    mid--;
                }
                res[0] = ++mid;
                while (mid < nums.length && nums[mid] == target) {
                    mid++;
                }
                res[1] = --mid;
                break;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    /**
     * 给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        boolean[] isContain = new boolean[s.length() + 1];
        // 空字符串肯定是可以拆分的
        isContain[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                // 判断当前位置是否合法，以及当前位置到i位置是否合法，如果合法，把i置为false
                if (isContain[j] && set.contains(s.substring(j, i))) {
                    isContain[i] = true;
                    break;
                }
            }
        }
        return isContain[s.length()];
    }

    /**
     * 给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums, int count) {
        int[] counts = new int[32];
        for (int num : nums) {
            for (int j = 0; j < 32; j++) {
                counts[j] += num & 1;
                num >>>= 1;
            }
        }
        int res = 0, m = 3;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            res |= counts[31 - i] % m;
        }
        return res;

    }

    /**
     * 计算员工的重要性
     *
     * @param employees 员工的列表
     * @param id        目标员工的id
     * @return
     */
    public int getImportance(List<Employee> employees, int id) {
        if (employees.isEmpty())
            return 0;
        HashMap<Integer, Employee> map = new HashMap<>();
        for (Employee temp : employees) {
            map.put(temp.id, temp);
        }
        return getImportance(map, id);
    }

    /**
     * 获取员工的重要性
     *
     * @param map 员工的map
     * @param id  员工的id
     * @return
     */
    private int getImportance(HashMap<Integer, Employee> map, int id) {
        Employee employee = map.get(id);
        int importance = employee.importance;
        for (Integer subordinate : employee.subordinates) {
            importance += getImportance(map, subordinate);
        }
        return importance;
    }

    /**
     * 砖墙 找出怎样画才能使这条线 穿过的砖块数量最少 ，并且返回 穿过的砖块数量 。
     *
     * @param wall
     * @return
     */
    public int leastBricks(List<List<Integer>> wall) {
        int size = wall.size();
        int count = size;
        HashMap<Integer, Integer> map = new HashMap<>();
        // 遍历每一层墙壁
        for (int i = 0; i < wall.size(); i++) {
            int width = 0;
            // 遍历每一层墙壁的每一块砖（除了最后一块），记录每块砖到左边界的距离
            for (int j = 0; j < wall.get(i).size() - 1; j++) {
                width += wall.get(i).get(j);
                map.put(width, map.getOrDefault(width, 0) + 1);
            }
        }

        // 遍历map，找到边缘与左侧边界宽度相同的最大值，用墙壁的高度，减去这个数量即为答案
        for (Integer integer : map.keySet()) {
            count = Math.min(count, size - map.get(integer));
        }
        return count;
    }

    /**
     * 单词搜索 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        int height = board.length;
        int width = board[0].length;
        // 记录当前位置是否已经访问过
        boolean[][] visited = new boolean[height][width];

        // 遍历二维数组，按顺序比较字符串中的每一个字符
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                boolean flag = check(board, visited, i, j, word, 0);
                // 如果找到了，直接返回false
                if (flag)
                    return true;
            }
        }
        return false;
    }

    private boolean check(char[][] board, boolean[][] visited, int x, int y, String word, int cur) {
        // 如果当前数和当前的字母不匹配，直接返回false
        if (board[x][y] != word.charAt(cur)) {
            return false;
        } else if (cur == word.length() - 1) {
            // 如果当前已经是最后一个字母并且已经匹配，返回true
            return true;
        }
        // 把当前数的访问状态设为true
        visited[x][y] = true;
        // 方向数组
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        boolean res = false;
        for (int[] dir : directions) {
            int newX = x + dir[0], newY = y + dir[1];
            // 判断即将访问的位置是否超过边界
            if (newX >= 0 && newX < board.length && newY >= 0 && newY < board[0].length) {
                if (!visited[newX][newY]) {
                    if (check(board, visited, newX, newY, word, cur + 1)) {
                        res = true;
                        break;
                    }
                }
            }
        }
        // 当前位置的上下左右字母都不满足，把当前位置的访问状态设为false，回溯
        visited[x][y] = false;
        return res;
    }

    /**
     * 子集 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     * <p>
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(0, nums, res, new ArrayList<>());
        return res;

    }

    private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
        res.add(new ArrayList<>(tmp));
        for (int j = i; j < nums.length; j++) {
            tmp.add(nums[j]);
            backtrack(j + 1, nums, res, tmp);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 整数反转
     * 
     * @param x
     * @return
     */
    public int reverse(int x) {
        int ans = 0;
        while (x != 0) {
            int pop = x % 10;
            if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && pop > 7))
                return 0;
            if (ans < Integer.MIN_VALUE / 10 || (ans == Integer.MIN_VALUE / 10 && pop < -8))
                return 0;
            ans = ans * 10 + pop;
            x /= 10;
        }
        return ans;
    }

    /**
     * 给定一个包含红色、白色和蓝色，一共n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
     *
     * 此题中，我们使用整数 0、1 和 2 分别表示红色、白色和蓝色。
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        int n = nums.length;
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            // 如果值为2，把当前的值与p2交换，这里的循环是为了保证之后不会漏掉交换之后的元素
            while (i <= p2 && nums[i] == 2) {
                int temp = nums[i];
                nums[i] = nums[p2];
                nums[p2] = temp;
                --p2;
            }
            // 如果值为0，把当前的值与p0交换
            if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[p0];
                nums[p0] = temp;
                ++p0;
            }
        }
    }

    /**
     * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     *
     * 说明：每次只能向下或者向右移动一步。
     * 
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        int height = grid.length;
        int width = grid[0].length;
        int[][] dp = new int[height][width];
        dp[0][0] = grid[0][0];
        // 走最左侧边界
        for (int i = 1; i < height; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        // 走最上边边界
        for (int i = 1; i < width; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }

        // 记录非边界的数字
        for (int i = 1; i < height; i++) {
            for (int j = 1; j < width; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[height - 1][width - 1];
    }

    /**
     * 一个机器人位于一个 m x n网格的左上角 （起始点在下图中标记为 “Start” ）。
     *
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     *
     * 问总共有多少条不同的路径？
     * 
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][n - 1] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[m - 1][i] = 1;
        }

        for (int i = m - 2; i >= 0; i--) {
            for (int j = n - 2; j >= 0; j--) {
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
            }
        }
        return dp[0][0];
    }

    /**
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
     * 请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
     *
     * 输入：intervals = [[1,3],[2,6],[8,10],[15,18]] 输出：[[1,6],[8,10],[15,18]] 解释：区间
     * [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        // 按照区间的起始位置进行排序
        Arrays.sort(intervals, (v1, v2) -> Integer.compare(v1[0], v2[0]));
        // 遍历区间
        int[][] res = new int[intervals.length][2];
        int idx = -1;
        for (int[] interval : intervals) {
            // 如果结果数组是空的，或者当前区间的起始位置 > 结果数组最后一个区间的终止位置
            // 则直接加入结果数组，不合并区间
            if (idx == -1 || interval[0] > res[idx][1]) {
                res[++idx] = interval;
            } else {
                // 反之说明区间存在重叠，则将当前区间合并
                res[idx][1] = Math.max(res[idx][1], interval[1]);
            }
        }
        return Arrays.copyOf(res, idx + 1);
    }

    /**
     * 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     *
     *
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            // 把字符串转成字符数组，进行排序后组成的字符串相等即为异位词
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String s = new String(array);
            // 把当前的字符串加到对应的list中，如果不存在创建一个list
            List<String> list = map.getOrDefault(s, new ArrayList<>());
            list.add(str);
            map.put(s, list);
        }
        ArrayList<List<String>> res = new ArrayList<>(map.values());
        return res;
    }

    /**
     * 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
     *
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
     *
     * @param text1
     * @param text2
     * @return
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();

        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i < dp.length; i++) {
            char c1 = text1.charAt(i - 1);
            for (int j = 1; j < dp[0].length; j++) {
                char c2 = text2.charAt(j - 1);
                // 如果最后一个字符相等，向前继续检查
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    // 如果最后一个字符不相等，则为前一个的最大值
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 给定一个 n×n 的二维矩阵matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     *
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int[][] temp = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp[j][n - 1 - i] = matrix[i][j];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = temp[i][j];
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.println();
        }
    }

    /**
     * 给定一个无重复元素的数组candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。
     *
     * candidates中的数字可以无限制重复被选取。
     *
     * 说明：
     *
     * 所有数字（包括target）都是正整数。 解集不能包含重复的组合。
     *
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        Deque<Integer> path = new ArrayDeque<>();
        dfs(candidates, 0, len, target, path, res);
        return res;
    }

    /**
     * @param candidates 候选数组
     * @param begin      搜索起点
     * @param len        冗余变量，是 candidates 里的属性，可以不传
     * @param target     每减去一个元素，目标值变小
     * @param path       从根结点到叶子结点的路径，是一个栈
     * @param res        结果集列表
     */
    private void dfs(int[] candidates, int begin, int len, int target, Deque<Integer> path, List<List<Integer>> res) {
        // target 为负数和 0 的时候不再产生新的孩子结点
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        // 重点理解这里从 begin 开始搜索的语意
        for (int i = begin; i < len; i++) {
            path.addLast(candidates[i]);

            // 注意：由于每一个元素可以重复使用，下一轮搜索的起点依然是 i，这里非常容易弄错
            dfs(candidates, i, len, target - candidates[i], path, res);

            // 状态重置
            path.removeLast();
        }
    }

    /**
     * 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 n/2 的元素。
     *
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        return nums[n / 2];
    }

    /**
     * 给定一个范围在 1 ≤ a[i] ≤ n (n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
     *
     * 找到所有在 [1, n] 范围之间没有出现在数组中的数字。
     *
     * @param nums
     * @return
     */
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        List<Integer> res = new ArrayList<>();
        return res;
    }

    /**
     * lru design
     * 
     * @param operators int整型二维数组 the ops
     * @param k         int整型 the k
     * @return int整型一维数组
     */
    public int[] LRU(int[][] operators, int k) {
        // write code here
        Map<Integer, Integer> map = new LinkedHashMap<>();
        LinkedList<Integer> list = new LinkedList<>();
        for (int[] operator : operators) {
            // 获取要操作的key
            int key = operator[1];
            // switch根据opt选择
            switch (operator[0]) {
                // opt = 1,set缓存
                case 1:
                    int value = operator[2];
                    // 如果还没超过k大小，继续添加
                    if (map.size() < k) {
                        map.put(key, value);
                    } else {
                        // 如果达到了k大小,移除最不常用的记录
                        Iterator it = map.keySet().iterator();
                        map.remove(it.next());
                        map.put(key, value);
                    }
                    break;
                // 如果opt = 2,获取key对应的value
                case 2:
                    if (map.containsKey(key)) {
                        int val = map.get(key);
                        list.add(val);
                        map.remove(key);
                        map.put(key, val);
                    } else {
                        list.add(-1);
                    }
                    break;
                default:
            }
        }
        int[] res = new int[list.size()];
        int i = 0;
        for (int val : list) {
            res[i++] = val;
        }
        return res;
    }

    /**
     * 最大数
     * 
     * @param nums
     * @return
     */
    public String solve(int[] nums) {
        // write code here
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strings, (a, b) -> Integer.parseInt(b + a) - Integer.parseInt(a + b));
        StringBuilder res = new StringBuilder();
        if (strings[0].equals("0")) {
            return "0";
        }
        for (String string : strings) {
            res.append(string);
        }
        return String.valueOf(res);
    }

    /**
     * 给你一个整数数组nums，你可以对它进行一些操作。
     *
     * 每次操作中，选择任意一个nums[i]，删除它并获得nums[i]的点数。 之后，你必须删除每个等于nums[i] - 1或nums[i] + 1的元素。
     *
     * 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
     *
     * @param nums
     * @return
     */
    // public int deleteAndEarn(int[] nums) {
    // int n = nums.length;
    // boolean[] used = new boolean[n];
    // int score = 0;
    // for (int i = 0; i < nums.length; i++) {
    // used[i] = true;
    // int score1 = deleteAndEarn(nums,used,nums[i],nums[i] - 1,1,nums[i]);
    // int score2 = deleteAndEarn(nums,used,nums[i],nums[i] + 1,1,nums[i]);
    // score = Math.max(score, Math.max(score1,score2));
    // used[i] = false;
    // }
    // return score;
    // }
    //
    // private int deleteAndEarn(int[] nums, boolean[] used, int cur, int next, int
    // len, int score) {
    // if (len > nums.length) {
    // return score;
    // }
    // System.out.println("score->"+score+" cur->"+cur);
    // for (int i = 0; i < nums.length; i++) {
    // if (used[i] || nums[i] != next) {
    // continue;
    // }
    // used[i] = true;
    // len++;
    // int score1 = deleteAndEarn(nums,used,nums[i],nums[i] - 1,len,score +
    // nums[i]);
    // int score2 = deleteAndEarn(nums,used,nums[i],nums[i] + 1,len,score +
    // nums[i]);
    // score = Math.max(score, Math.max(score1,score2));
    // }
    // return score;
    // }

    public int deleteAndEarn(int[] nums) {
        int maxVal = 0;
        for (int val : nums) {
            maxVal = Math.max(maxVal, val);
        }
        int[] sum = new int[maxVal + 1];
        for (int val : nums) {
            sum[val] += val;
        }
        return rob(sum);
    }

    public int rob(int[] nums) {
        int size = nums.length;
        int first = nums[0], second = Math.max(nums[0], nums[1]);
        for (int i = 2; i < size; i++) {
            int temp = second;
            second = Math.max(first + nums[i], second);
            first = temp;
        }
        return second;
    }

    /**
     * 未知 整数数组 arr 由 n 个非负整数组成。
     *
     * 经编码后变为长度为 n - 1 的另一个整数数组 encoded ，其中 encoded[i] = arr[i] XOR arr[i + 1]
     * 。例如，arr = [1,0,2,1] 经编码后得到 encoded = [1,2,3] 。
     *
     * 给你编码后的数组 encoded 和原数组 arr 的第一个元素 first（arr[0]）。
     *
     * 请解码返回原数组 arr 。可以证明答案存在并且是唯一的。
     *
     * @param encoded encoded = [1,2,3]
     * @param first   first = 1
     * @return [1,0,2,1]
     */
    public int[] decode(int[] encoded, int first) {
        // 利用异或运算的性质推导出 arr[i-1] ^ encoded[i-1] = arr[i]
        int n = encoded.length + 1;
        int[] arr = new int[n];
        arr[0] = first;
        for (int i = 1; i < n; i++) {
            arr[i] = arr[i - 1] ^ encoded[i - 1];
        }
        return arr;
    }

    /**
     * 判断是否为回文串
     * 
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        int n = s.length();
        int left = 0;
        int right = n - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    /**
     * 找到 s 中最长的回文子串
     * 
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        return "";
    }

    /**
     * 制作m朵花需要的最少天数
     * 
     * @param bloomDay
     * @param m
     * @param k
     * @return
     */
    public int minDays(int[] bloomDay, int m, int k) {
        // 如果需要的花朵数大于总的花朵数，不能完成制作
        if (m > bloomDay.length / k) {
            return -1;
        }
        int low = Integer.MAX_VALUE, high = 0;
        int length = bloomDay.length;
        // 遍历获得取得有花开需要的最小天数，所有花开需要的时间天数
        for (int i = 0; i < length; i++) {
            low = Math.min(low, bloomDay[i]);
            high = Math.max(high, bloomDay[i]);
        }
        // 进行二分查找
        while (low < high) {
            int days = (high - low) / 2 + low;
            if (canMake(bloomDay, days, m, k)) {
                high = days;
            } else {
                low = days + 1;
            }
        }
        return low;
    }

    /**
     * 是否可以得到需要的花
     * 
     * @param bloomDay
     * @param days
     * @param m
     * @param k
     * @return
     */
    public boolean canMake(int[] bloomDay, int days, int m, int k) {
        int bouquets = 0;
        int flowers = 0;
        int length = bloomDay.length;
        for (int i = 0; i < length && bouquets < m; i++) {
            if (bloomDay[i] <= days) {
                flowers++;
                if (flowers == k) {
                    bouquets++;
                    flowers = 0;
                }
            } else {
                flowers = 0;
            }
        }
        return bouquets >= m;
    }

    /**
     * 给你一个整数数组 perm ，它是前 n 个正整数的排列，且 n 是个 奇数 。
     * 
     * 它被加密成另一个长度为 n - 1 的整数数组 encoded ，满足 encoded[i] = perm[i] XOR perm[i +
     * 1] 。比方说，如果 perm = [1,3,2] ，那么 encoded = [2,1] 。
     * 
     * @param encoded
     * @return
     */
    public int[] decode(int[] encoded) {
        int n = encoded.length + 1;
        int total = 0;
        for (int i = 1; i <= n; i++) {
            total ^= i;
        }
        int odd = 0;
        for (int i = 1; i < n - 1; i += 2) {
            odd ^= encoded[i];
        }
        int[] perm = new int[n];
        perm[0] = total ^ odd;
        for (int i = 0; i < n - 1; i++) {
            perm[i + 1] = perm[i] ^ encoded[i];
        }
        return perm;
    }

    /**
     * 有一个正整数数组 arr，现给你一个对应的查询数组 queries，其中 queries[i] = [Li, Ri]。
     * 
     * 对于每个查询 i，请你计算从 Li 到 Ri 的 XOR 值（即 arr[Li] xor arr[Li+1] xor ... xor
     * arr[Ri]）作为本次查询的结果。
     * 
     * 并返回一个包含给定查询 queries 所有结果的数组。
     * 
     * 输入：arr = [1,3,4,8], queries = [[0,1],[1,2],[0,3],[3,3]] 输出：[2,7,14,8]
     * 
     * @param arr
     * @param queries
     * @return
     */
    public int[] xorQueries(int[] arr, int[][] queries) {
        int m = queries.length;
        int[] res = new int[m];
        for (int i = 0; i < m; i++) {
            xorCal(arr, queries[i][0], queries[i][1], res, i);
        }
        return res;
    }

    /**
     * 计算异或运算
     * 
     * @param arr 计算异或运算的数组
     * @param i   开始下标
     * @param j   结束下标
     * @param res 存储的结果数组
     * @param cur 当前的位置
     */
    private void xorCal(int[] arr, int i, int j, int[] res, int cur) {
        int xor = 0;
        for (int k = i; k <= j; k++) {
            xor = arr[k] ^ xor;
        }
        res[cur] = xor;
    }

    /**
     * 乘积最大子数组
     * 
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     * 
     * 1.分解子问题 dp[i]
     * 
     * 2.状态定义
     * 
     * 3.DP方程
     * 
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int n = nums.length;
        if (n == 1)
            return nums[0];
        int[] dpMax = new int[n + 1];
        int[] dpMin = new int[n + 1];
        dpMax[0] = nums[0];
        int max = dpMax[0];
        for (int i = 1; i < n; i++) {
            dpMax[i] = Math.max(Math.max(dpMax[i - 1] * nums[i], dpMin[i - 1] * nums[i]), nums[i]);
            dpMin[i] = Math.min(Math.min(dpMax[i - 1] * nums[i], dpMin[i - 1] * nums[i]), nums[i]);
            max = Math.max(Math.max(dpMin[i], dpMax[i]), max);
        }
        return max;
    }

    /**
     * 岛屿数量
     * 
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 
     * 此外，你可以假设该网格的四条边均被水包围。
     * 
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        int count = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0') {
                    continue;
                }
                count += isLand(grid, m, n, i, j);
            }
        }
        return count;
    }

    private int isLand(char[][] grid, int m, int n, int x, int y) {
        int[][] dir = { { 1, 0 }, { 0, -1 }, { -1, 0 }, { 0, 1 } };
        int count = 1;
        grid[x][y] = '0';
        for (int i = 0; i < dir.length; i++) {
            int dirX = x + dir[i][0], dirY = y + dir[i][1];
            if (dirX < m && dirX >= 0 && dirY >= 0 && dirY < n && grid[dirX][dirY] == '1') {
                grid[dirX][dirY] = '0';
                isLand(grid, m, n, dirX, dirY);
            }
        }
        return count;
    }

    /**
     * 
     * 有一个长度为 arrLen 的数组，开始有一个指针在索引 0 处。
     * 
     * 每一步操作中，你可以将指针向左或向右移动 1 步，或者停在原地（指针不能被移动到数组范围外）。
     * 
     * 给你两个整数 steps 和 arrLen ，请你计算并返回：在恰好执行 steps 次操作以后，指针仍然指向索引 0 处的方案数。
     * 
     * 由于答案可能会很大，请返回方案数 模 10^9 + 7 后的结果。
     * 
     * 
     * @param steps
     * @param arrLen
     * @return
     */
    public int numWays(int steps, int arrLen) {
        final int MODULO = 1000000007;
        int maxColumn = Math.min(arrLen - 1, steps);
        // dp[i][j]表示在 i 步操作之后，指针位于下标 j 的方案数
        int[][] dp = new int[steps + 1][maxColumn + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= steps; i++) {
            for (int j = 0; j <= maxColumn; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j - 1 >= 0) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MODULO;
                }
                if (j + 1 <= maxColumn) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % MODULO;
                }
            }
        }
        return dp[steps][0];
    }

    /**
     * 
     * 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
     * 
     * 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi]
     * ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
     * 
     * 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。 请你判断是否可能完成所有课程的学习？如果可以，返回 true
     * ；否则，返回 false 。
     * 
     * 
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < prerequisites.length; i++) {
            map.put(prerequisites[i][0], prerequisites[i][1]);
        }
        return false;
    }

    /**
     * 整数转罗马数字
     * 
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        // 把阿拉伯数字与罗马数字可能出现的所有情况和对应关系，放在两个数组中，并且按照阿拉伯数字的大小降序排列
        int[] nums = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
        String[] romans = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
        StringBuilder stringBuilder = new StringBuilder();
        int index = 0;
        while (index < 13) {
            // 特别注意：这里是等号
            while (num >= nums[index]) {
                stringBuilder.append(romans[index]);
                num -= nums[index];
            }
            index++;
        }
        return stringBuilder.toString();
    }

    /**
     * 罗马数字转整数
     * 
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        map.put('M', 1000);
        map.put('D', 500);
        map.put('C', 100);
        map.put('L', 50);
        map.put('X', 10);
        map.put('V', 5);
        map.put('I', 1);
        int res = 0;
        int preNum = map.get(s.charAt(0));
        for (int i = 1; i < s.length(); i++) {
            int num = map.get(s.charAt(i));
            if (preNum < num) {
                res -= preNum;
            } else {
                res += preNum;
            }
            preNum = num;
        }
        res += preNum;
        return res;
    }

    /**
     * 除自身以外数组的乘积
     * 
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;

        // L 和 R 分别表示左右两侧的乘积列表
        int[] L = new int[length];
        int[] R = new int[length];

        int[] answer = new int[length];

        // L[i] 为索引 i 左侧所有元素的乘积
        // 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
        L[0] = 1;
        for (int i = 1; i < length; i++) {
            L[i] = nums[i - 1] * L[i - 1];
        }

        // R[i] 为索引 i 右侧所有元素的乘积
        // 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
        R[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {
            R[i] = nums[i + 1] * R[i + 1];
        }

        // 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for (int i = 0; i < length; i++) {
            answer[i] = L[i] * R[i];
        }

        return answer;
    }

    /**
     * 数组交换方法
     * 
     * @param arr 要交换的数组
     * @param i   交换的下标
     * @param j   交换的下标
     * @return
     */
    private int[] swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        return arr;
    }

    /**
     * 数组打印方法
     * 
     * @param arr
     */
    private void print(int[] arr) {
        for (int num : arr) {
            System.out.print(num + "\t");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        HelloWorld hw = new HelloWorld();
        System.out.println("hello world!");
        char[][] grid = { { '1', '1', '1', '1', '0' }, { '1', '1', '0', '1', '0' }, { '1', '1', '0', '0', '0' },
                { '0', '0', '0', '0', '0' } };
        int[] arr = { 1, 2, 3, 4 };
        hw.print(hw.productExceptSelf(arr));
    }
}

// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;

    public Employee() {
    }

    public Employee(int id, int importance, List<Integer> subordinates) {
        this.id = id;
        this.importance = importance;
        this.subordinates = subordinates;
    }

    @Override
    public String toString() {
        return "Employee{" + "id=" + id + ", importance=" + importance + ", subordinates=" + subordinates + '}';
    }
}