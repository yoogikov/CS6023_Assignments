import random
for tc in range(3, 11):
    in_file = open("testcases/input/input" + str(tc)+".txt", "w");
    out_file = open("testcases/output/output" + str(tc)+".txt", "w");
    p = random.randint(1, 1000)
    q = random.randint(1, 1000)
    r = random.randint(1, 1000)
    s = random.randint(1, 1000)
    A = [[random.randint(-10, 11) for i in range(q)]for j in range(p)]
    B = [[random.randint(-10, 11) for i in range(p)]for j in range(q)]
    C = [[random.randint(-10, 11) for i in range(r)]for j in range(q)]
    D = [[random.randint(-10, 11) for i in range(r)]for j in range(s)]
    print(p, q, r, s)
    E = [[A[i][j]+B[j][i] for j in range(q)] for i in range(p)]
    print("Added")
    F = [[0 for j in range(r)] for i in range(p)]
    for i in range(p):
        for k in range(q):
            for j in range(r):
                F[i][j] += E[i][k]*C[k][j];
    print("Mult1")
    X = [[0 for p in range(s)] for i in range(p)]
    for i in range(p):
        for j in range(s):
            for k in range(r):
                X[i][j] += F[i][k]*D[j][k];
    print("Mult2")
    in_file.write(f"{p} {q} {r} {s}\n")
    for i in range(p):
        st = ""
        for j in range(q):
            st += str(A[i][j]) + " "
        st += "\n"
        in_file.write(st)
    for i in range(q):
        st = ""
        for j in range(p):
            st += str(B[i][j]) + " "
        st += "\n"
        in_file.write(st)
    for i in range(q):
        st = ""
        for j in range(r):
            st += str(C[i][j]) + " "
        st += "\n"
        in_file.write(st)
    for i in range(s):
        st = ""
        for j in range(r):
            st += str(D[i][j]) + " "
        st += "\n"
        in_file.write(st)
    for i in range(p):
        st = ""
        for j in range(s):
            st += str(X[i][j]) + " "
        st += "\n"
        out_file.write(st)


