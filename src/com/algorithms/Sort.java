package com.algorithms;

/**
 * @author: alex
 * @date: 2021/5/4 5:25 下午
 */

public class Sort {

    /**
     * 快速排序
     * @param arr
     * @param left  左边界
     * @param right 右边界
     */
    public void quickSort(int[] arr, int left, int right) {
        if (left >= right) return;
        // 首先获得轴的位置，然后再对左右数组进行排序
        int pivot = partition(arr, left, right);
        quickSort(arr, left, pivot - 1);
        quickSort(arr, pivot + 1, right);
    }

    /**
     * 分组
     * @param arr
     * @param leftBound 左边界
     * @param rightBound 右边界
     * @return
     */
    private int partition(int[] arr, int leftBound, int rightBound) {
        // 选右边界的值作为轴
        int pivot = arr[rightBound];
        int left = leftBound;
        int right = rightBound - 1;

        while (left <= right) {
//            System.out.println("left -> " + arr[left] + ";right -> " + arr[right]);
            // 如果左边的数比轴小，继续向右移动
            while (left <= right && arr[left] <= pivot) left++;
            // 如果右边的数比轴大，继续向左移动
            while (left <= right && arr[right] > pivot) right--;
            // 左边找到比轴大的数，右边找到比轴小的数，交换两个数
            if (left < right) {
                swap(arr, left, right);
//                print(arr);
            }
        }
        swap(arr, left, rightBound);
        return left;
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    private void print(int[] arr) {
        for (int num : arr) {
            System.out.print(num + "\t");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        Sort sort = new Sort();
        int[] arr = {1,3,2,5,4};
//        System.out.println("before sort:");
//        sort.print(arr);
//        sort.quickSort(arr, 0, arr.length - 1);
//        System.out.println("after sort:");
//        sort.print(arr);
    }
}
