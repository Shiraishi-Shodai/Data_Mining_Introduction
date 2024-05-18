import numpy as np
import sympy
from concurrent.futures import ProcessPoolExecutor


def calc_pi(roop_count, pi):
    """素数かどうか判断するための整数を生成
    
    Args:
        roop_count (int ):  main関数のwhile文をループした回数
        pi        (float):  円周率...小数点以下の桁数は(17 + roop_count)
    
    Returns:
        int: 素数判定をするための整数
    """
    
    # 小数点以下を取得
    prime_candidate = str(pi).split('.')[1]
    # while文をループした回数分の先頭桁数と丸めさを避けるために取得した余分の1桁を無視
    prime_candidate = int(prime_candidate[roop_count:-1])
    return prime_candidate


if __name__ == '__main__':
    
    # 円周率の桁数(整数3の1桁と小数点以下の末尾の丸め誤差を避けるために小数点以下は17桁を取得する)
    PI_DIGIT = 18
    # 素数を取得したい回数
    MAX_PRIME_NUM = 10
    # 現在見つかった素数の個数
    prime_count = 0
    # while文をループした回数
    roop_count = 0
    # 最終的な答え(2024個目の素数)
    answer = 0
    # 並列処理をするプロセス数
    PROCESS_NUM = 5
    
    while prime_count != MAX_PRIME_NUM:
        
        #    素数判定をする値を一時的に格納するリスト
            tmp_candidate_list = []
            a = 0
            if roop_count == 0:
                a = 0
            else:
                a = 1
            # 5プロセスを並行処理する
            for i in range(a, PROCESS_NUM + a):
                pi = sympy.pi.evalf(PI_DIGIT + (roop_count + i))
                prime_candidate = calc_pi((roop_count + i), pi)
                tmp_candidate_list.append(prime_candidate)
                
            with ProcessPoolExecutor() as executor:
                for number, is_prime in zip(tmp_candidate_list, executor.map(sympy.ntheory.isprime, tmp_candidate_list)):
        
                    if is_prime:
                        prime_count += 1
                        print(prime_candidate)
                        
                        if prime_count == MAX_PRIME_NUM:
                            answer = prime_candidate
                            break
        
            roop_count += PROCESS_NUM
        
    print(f'{MAX_PRIME_NUM}個目の素数は{answer}')