def merge(a):
    if a <= 1:
        return 0
    return a + merge(a - 1) + merge(a - 1)

print(merge(16))
print(merge(4))