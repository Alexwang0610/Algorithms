package com.algorithms;

import java.util.HashSet;

public class ListNode {
    int val;
    ListNode next;

    ListNode() {

    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }

    /**
     * 根据数组构造链表
     * 
     * @param nums 数组
     */
    ListNode(int[] nums) {
        ListNode target = new ListNode(nums[0]);
        ListNode head = target;
        for (int i = 1; i < nums.length; i++) {
            target.next = new ListNode(nums[i]);
            target = target.next;
        }
        this.val = head.val;
        this.next = head.next;
    }

    /**
     * 头插法插入节点
     * 
     * @param val 要插入节点的值
     */
    public void headInsert(int val) {
        ListNode node = new ListNode(val);
        node.next = this;
    }

    /**
     * 尾插法插入节点
     * 
     * @param val 要插入节点的值
     */
    public void lastInsert(int val) {

    }

    /**
     * 在中间插入节点
     * 
     * @param val   插入节点的值
     * @param index 插入节点的位置
     */
    public void insert(int val, int index) {

    }

    public static void delete(int val) {

    }

    /**
     * 获取链表的最后一个节点
     * 
     * @param head 目标链表
     * @return
     */
    public static ListNode getLastNode(ListNode head) {
        if (head == null || head.next == null)
            return head;
        while (head.next != null)
            head = head.next;
        return head;
    }

    /**
     * 链表打印函数
     * 
     * @param head 要打印的链表
     */
    public static void print(ListNode head) {
        while (head != null) {
            if (head.next != null)
                System.out.print(head.val + "->");
            else
                System.out.println(head.val);
            head = head.next;
        }
    }

    /**
     * 旋转链表
     * 
     * @param head 要旋转的链表
     * @param k    旋转的次数
     * @return
     */
    public static ListNode rotateRight(ListNode head, int k) {
        // 如果链表为空或者只有一个元素，或者旋转次数等于0
        if (head == null || head.next == null || k == 0)
            return head;

        ListNode nodeA = head;
        // 计算链表的长度
        int count = 0;
        while (nodeA.next.next != null) {
            nodeA = nodeA.next;
            count++;
        }
        count += 2;
        k %= count;
        if (k == 0)
            return head;
        ListNode node = nodeA.next;
        nodeA.next = null;
        node.next = head;
        node = rotateRight(node, k - 1);

        return node;
    }

    /**
     * 删除链表中重复的元素，如果有节点是重复的就把这个节点删除
     * 
     * @param head 链表
     * @return
     */
    public static ListNode deleteDuplicates(ListNode head) {
        // 1.迭代法，一次遍历
        // if (head == null || head.next == null)
        // return head;

        // // 因为链表的头结点可能会被删除，所以使用一个哑结点指向链表的头结点
        // ListNode dummyNode = new ListNode(0, head);
        // ListNode nodeA = dummyNode;

        // // 遍历链表的节点，判断当前节点的next以及当前节点next节点的next是否为空
        // while (nodeA.next != null && nodeA.next.next != null) {
        // // 如果当前节点的next节点的值和当前节点next节点的next节点的值相等
        // if (nodeA.next.val == nodeA.next.next.val) {
        // int same = nodeA.next.val;
        // // 遍历所有与当前节点next节点值相同的节点
        // while (nodeA.next != null && nodeA.next.val == same) {
        // nodeA.next = nodeA.next.next;
        // }
        // } else {
        // nodeA = nodeA.next;
        // }
        // }
        // return dummyNode.next;

        // 没有节点或者只有一个节点，必然没有重复元素
        if (head == null || head.next == null)
            return head;

        // 当前节点和下一个节点，值不同，则head的值是需要保留的，对head.next继续递归
        if (head.val != head.next.val) {
            head.next = deleteDuplicates(head.next);
            return head;
        } else {
            // 当前节点与下一个节点的值重复了，重复的值都不能要。
            // 一直往下找，找到不重复的节点。返回对不重复节点的递归结果
            ListNode notDup = head.next.next;
            while (notDup != null && notDup.val == head.val) {
                notDup = notDup.next;
            }
            return deleteDuplicates(notDup);
        }
    }

    /**
     * 分割链表，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前
     * 
     * @param head 要分割的链表
     * @param x    分割节点的值
     * @return
     */
    public static ListNode partition(ListNode head, int x) {
        ListNode smallListNode = new ListNode(0, head);
        ListNode smallNode = smallListNode;
        ListNode largeListNode = new ListNode(0, head);
        ListNode largeNode = largeListNode;

        while (head != null) {
            if (head.val < x) {
                smallListNode.next = head;
                smallListNode = smallListNode.next;
            } else {
                largeListNode.next = head;
                largeListNode = largeListNode.next;
            }
            head = head.next;
        }
        // 把large节点的next节点设为null
        largeListNode.next = null;
        smallListNode.next = largeNode.next;
        return smallNode.next;
    }

    /**
     * 相交链表
     * 
     * @param headA 链表A
     * @param headB 链表B
     * @return
     */
    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
            return null;
        ListNode nodeA = headA;
        ListNode nodeB = headB;

        while (nodeA != nodeB) {
            if (nodeA != null) {
                nodeA = nodeA.next;
            } else {
                nodeA = headB;
            }

            if (nodeB != null) {
                nodeB = nodeB.next;
            } else {
                nodeB = headA;
            }
        }
        return nodeA;
    }

    /**
     * 删除链表中与给定值相等的节点
     * 
     * @param head 链表
     * @param val  给定要删除节点的值
     * @return
     */
    public static ListNode removeElements(ListNode head, int val) {
        if (head == null)
            return null;
        // // 1.迭代
        // // 设置哨兵节点，因为可能会删除链表的头节点
        // ListNode sentinel = new ListNode(0, head);
        // ListNode curNode = head, prevNode = sentinel;
        // while (curNode != null) {
        // // 如果当前节点的值和要删除节点的值是相等的
        // if (curNode.val == val) {
        // prevNode.next = curNode.next;
        // } else {
        // // 否则前驱节点等于当前节点
        // prevNode = curNode;
        // }
        // curNode = curNode.next;
        // }
        // return sentinel.next;

        // 2.递归
        head.next = removeElements(head.next, val);
        return head.val == val ? head.next : head;
    }

    /**
     * 反转链表
     * 
     * @param head
     * @return
     */
    public static ListNode reverseList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode prevNode = null;
        ListNode curNode = head;

        // 链表:1->2->6->3->4->5->6
        while (curNode != null) {
            // 存储当前节点的next节点:node->1=node->2
            ListNode tempNode = curNode.next;
            // 当前节点的next节点变为前驱结点:node->2 = null
            curNode.next = prevNode;
            // 前驱节点变成当前节点:null = node->1
            prevNode = curNode;
            // 当前节点变成当前节点的下一个节点:node->1 = node->2
            curNode = tempNode;
        }
        return prevNode;
    }

    /**
     * 判断是否为回文链表
     * 
     * @param head
     * @return
     */
    public static boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        HashSet<Integer> set = new HashSet<>();
        boolean isPalindrome = false;
        while (head != null) {
            if (!set.add(head.val)) {
                isPalindrome = true;
            } else {
                isPalindrome = false;
            }
            head = head.next;
        }
        return isPalindrome;
    }

    /**
     * 删除链表中的节点
     * 
     * @param node
     */
    public static void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**
     * 环形链表，返回环形链表环的入口节点
     * 
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null)
            return null;
        ListNode cycle = isCycle(head);
        if (cycle == null)
            return null;
        ListNode slow = cycle;
        ListNode fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    private ListNode isCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next.next != null) {
            if (fast == slow) {
                return slow;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return null;
    }

    public static void main(String[] args) {
        int[] arr = { 1, 3, 2, 1 };
        ListNode head = new ListNode(arr);
        print(head);
        System.out.println(isPalindrome(head));
    }
}