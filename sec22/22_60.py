#通常の関数
def hello_func(n):
    for x in range(1,n+1):
        print(f'hello func {x},',end="")
    print("end of hello_func")
    return 1

#ジェネレータ
def hello_gene(n):
    for x in range(1,n+1):
        yield(f'hello generator {x},') #ここでいったん、関数に再開可能な形で抜ける
    print("end of hello_gene")

#関数はreturnが実行されたら関数から抜けて戻ってこない。
print(hello_func(3)) #通常の関数

#next でyieldまで実行される。yieldが呼ばれるとジェネレータ関数を抜け、再度nextが呼ばれるまではジェネレータは実行さない。
#yieldで返す値がなくなったら 例外StopIterationが発生。
gen = hello_gene(3) # genはジェネレータオブジェクト
print(next(gen)) #yieldまでジェネレータ内の処理を進める
print(next(gen))
print(next(gen))

print("")
gen2 = hello_gene(5)
for g in gen2:
    print(g)
    pass