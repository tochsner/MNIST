import random
import itertools

comparisons = 0

def quickSort(A):
   quickSortHelper(A, 0, len(A) - 1)

def quickSortHelper(A, l, r):
   if l < r:
       splitpoint = partition(A, l, r)

       print(splitpoint)

       quickSortHelper(A, l, splitpoint - 1)
       quickSortHelper(A, splitpoint + 1, r)

def partition(A,l,r):
   global comparisons

   i = l
   r_ = r - 1
   j = r_

   p = A[r]

   done = False
   while not done:
       while i < r and A[i] < p:
           i = i + 1
           comparisons += 1

       while j > l and A[j] > p:
           j = j - 1
           comparisons += 1

       if i < j:
           A[i], A[j] = A[j], A[i]
       else:
           done = True

   A[i], A[r] = A[r], A[i]

   return i


def try_all_permutations(n):
    global comparisons

    max_comparisons = 0

    combinations = itertools.permutations(range(n))

    for combination in combinations:
        quickSort(list(combination))

        if comparisons > max_comparisons:
            max_comparisons = comparisons
            print(max_comparisons)

        comparisons = 0

    print("---")
    print(n * (n-1) / 2)

def try_sorted(n):
    global comparisons

    sorted_list = list(range(n))
    quickSort(sorted_list)
    print(comparisons)
    comparisons = 0

n = 19
try_sorted(n)
print(n * (n-1) / 2)