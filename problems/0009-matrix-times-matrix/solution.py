def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(b) != len(a[0]):
        return -1
    c = [
        [
            sum(i*j for i,j in zip(row,col))
            for col in zip(*b)
        ]
        for row in a
    ]
    return c