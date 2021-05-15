package com.algorithms;

import java.util.*;

public class TreeNode {
    Integer sum = 0;
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {

    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    /**
     * 根据输入的字符串形式的数组构建二叉树，例如"[5,1,4,null,null,3,6]"
     *
     * @param target
     * @return
     */
    public TreeNode buildTree(String target) {
        // 去掉字符串中的中括号
        target = target.substring(1, target.length() - 1);
        // 把字符串按“,”分割成字符串数组
        String[] treeArr = target.split(",");
        int len = treeArr.length;
        int index = 0;
        TreeNode node = new TreeNode(string2Array(treeArr[index]));
        Deque<TreeNode> deque = new LinkedList<>();
        deque.addLast(node);
        index++;
        while (index < len) {
            TreeNode root = deque.pollFirst();
            if (index < len && !"null".equals(treeArr[index])) {
                root.left = new TreeNode(string2Array(treeArr[index]));
                deque.addLast(root.left);

            }
            index++;
            if (index < len && !"null".equals(treeArr[index])) {
                root.right = new TreeNode(string2Array(treeArr[index]));
                deque.addLast(root.right);
            }
            index++;
        }

        return node;
    }

    /**
     * 把给定字符串变成数字数组返回
     *
     * @param s
     * @return
     */
    private int string2Array(String s) {
        char[] arr = s.toCharArray();
        int res = 0;
        for (int i = 0; i < arr.length; i++) {
            res = res * 10 + (arr[i] - '0');
        }
        return res;
    }

    /**
     * 中序遍历
     *
     * @param root
     */
    public void inOrder(TreeNode root) {
        if (root == null)
            return;
        inOrder(root.left);
        System.out.println(root.val);
        inOrder(root.right);
    }

    /**
     * 判断是否为二叉搜索树
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        return isValid(root, null, null);
    }

    private boolean isValid(TreeNode root, Integer lower, Integer upper) {
        if (root == null)
            return true;

        int val = root.val;
        if (lower != null && val <= lower)
            return false;
        if (upper != null && val >= upper)
            return false;

        if (!isValid(root.right, val, upper))
            return false;
        if (!isValid(root.left, lower, val))
            return false;
        return true;
    }

    /**
     * 二叉树的最大深度
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * 判断是否为平衡二叉树
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        return height(root) >= 0;
    }

    public int height(TreeNode root) {
        if (root == null)
            return 0;
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);

        if (leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        } else {
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    /**
     * 二叉树的右视图
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        _rightSideDFS(root, 0, result);
        return result;
    }

    private void _rightSideDFS(TreeNode root, int dep, List<Integer> result) {
        if (root == null)
            return;
        if (dep == result.size()) {
            result.add(root.val);
        }
        dep++;
        _rightSideDFS(root.right, dep, result);
        _rightSideDFS(root.left, dep, result);
    }

    /**
     * 递增顺序搜索树
     *
     * @param root
     * @return
     */
    public TreeNode increasingBST(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfs(root, res);

        TreeNode dummyNode = new TreeNode(-1);
        TreeNode currNode = dummyNode;
        for (int value : res) {
            currNode.right = new TreeNode(value);
            currNode = currNode.right;
        }
        return dummyNode.right;
    }

    public void dfs(TreeNode node, List<Integer> res) {
        if (node == null) {
            return;
        }
        dfs(node.left, res);
        res.add(node.val);
        dfs(node.right, res);
    }

    /**
     * 对称二叉树
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    /**
     * 判断是否为镜像树
     *
     * @param node1
     * @param node2
     * @return
     */
    public boolean check(TreeNode node1, TreeNode node2) {
        // 如果均为空,则返回true
        if (node1 == null && node2 == null) {
            return true;
        }
        // 如果只有一个为空，返回false
        if (node1 == null || node2 == null) {
            return false;
        }
        // 如果两个都不为空，判断当前节点的值是否相等，并且继续递归
        return node1.val == node2.val && check(node1.left, node2.right) && check(node1.right, node2.left);
    }

    /**
     * 二叉搜索树的范围和
     *
     * @param root
     * @param low
     * @param high
     * @return
     */
    public int rangeSumBST(TreeNode root, int low, int high) {
        _sum(root, low, high);
        return sum;
    }

    private void _sum(TreeNode root, int low, int high) {
        if (root == null) {
            return;
        }
        _sum(root.left, low, high);
        if (root.val >= low && root.val <= high) {
            System.out.println(root.val);
            sum += root.val;
        }
        _sum(root.right, low, high);
    }

    /**
     * 给你二叉树的根结点 root ，请你将它展开为一个单链表
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        TreeNode curr = root;
        // 遍历二叉树
        while (curr != null) {
            // 如果左节点不为空
            if (curr.left != null) {
                // 找到左子树的最右节点，作为右子树的前驱节点
                TreeNode next = curr.left;
                TreeNode predecessor = next;
                while (predecessor.right != null) {
                    predecessor = predecessor.right;
                }
                // 最右节点的右节点指向右子树
                predecessor.right = curr.right;
                // 把左节点设为空
                curr.left = null;
                // 当前节点的右节点指向当前节点的左节点
                curr.right = next;
            }
            curr = curr.right;
        }
    }

    private void preOrder(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        preOrder(root.left, list);
        preOrder(root.right, list);
    }

    /**
     * 根据一棵树的前序遍历与中序遍历构造二叉树。 前序遍历 preorder = [3,9,20,15,7] 中序遍历 inorder =
     * [9,3,15,20,7]
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int preLen = preorder.length;
        int inLen = inorder.length;

        // 如果前序遍历和中序遍历的长度不相等，抛出异常
        if (preLen != inLen) {
            throw new RuntimeException("Incorrect input data!");
        }

        // 存储中序遍历的值以及对应的下标
        HashMap<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            indexMap.put(inorder[i], i);
        }

        return buildTree(preorder, 0, preLen - 1, indexMap, 0, inLen - 1);
    }

    /**
     * 构建二叉树
     *
     * @param preorder 前序遍历数组
     * @param preLeft  前序序列的左边界
     * @param preRight 前序序列的右边界
     * @param indexMap 中序遍历的值以及对应的下标Map
     * @param inLeft   中序序列的左边界
     * @param inRight  中序序列的右边界
     * @return
     */
    private TreeNode buildTree(int[] preorder, int preLeft, int preRight, HashMap<Integer, Integer> indexMap,
            int inLeft, int inRight) {
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        // 当前根节点的值
        int rootVal = preorder[preLeft];
        TreeNode root = new TreeNode(rootVal);
        // 根节点在中序遍历中的下标
        Integer rootIndex = indexMap.get(rootVal);
        root.left = buildTree(preorder, preLeft + 1, rootIndex - inLeft + preLeft, indexMap, inLeft, rootIndex - 1);
        root.right = buildTree(preorder, rootIndex - inLeft + preLeft + 1, preRight, indexMap, rootIndex + 1, inRight);
        return root;
    }

    /**
     * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    /**
     * 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
     * 
     * @param root
     * @return
     */
    int ans;

    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }

    private int depth(TreeNode root) {
        if (root == null)
            return 0;
        // 计算左子树的节点数
        int left = depth(root.left);
        // 计算右子树的节点数
        int right = depth(root.right);
        // 节点数为左右节点数之和加上根节点数
        ans = Math.max(ans, left + right + 1);
        return Math.max(left, right) + 1;
    }

    /**
     * 叶子节点相似的树
     * 
     * @param root1
     * @param root2
     * @return
     */
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        leafDFS(root1, list1);
        leafDFS(root2, list2);
        return list1.equals(list2);
    }

    private void leafDFS(TreeNode root, List<Integer> list) {
        if (root.left == null && root.right == null) {
            list.add(root.val);
            return;
        }
        if (root.left != null) {
            leafDFS(root.left, list);
        }
        if (root.right != null) {
            leafDFS(root.right, list);
        }
    }

    /**
     * 最近的公共祖先
     * 
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 如果root为空，说明已经遍历到叶节点了，如果root为p，q直接由此向上回溯
        if (root == null || root == p || root == q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null && right == null)
            return null; // 1.如果左右都为空，说明不存在要找的节点
        if (left == null)
            return right; // 3.如果左为空，右不为空，说明p，q不在左子树中，返回right
        if (right == null)
            return left; // 4.如果右为空，左不为空，说明p，q不在右子树中，返回left
        return root; // 2. left和right都不为空，说明当前节点就是最近公共祖先
    }

    public static void main(String[] args) {
        TreeNode root1 = new TreeNode();
        root1 = root1.buildTree("[3,5,1,6,2,0,8,null,null,7,4]");
        TreeNode p = new TreeNode(5);
        TreeNode q = new TreeNode(1);
        System.out.println(root1.lowestCommonAncestor(root1, p, q));
    }
}