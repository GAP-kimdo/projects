import numpy as np
import math
import itertools
from cvxopt import matrix, solvers
from numpy import transpose, sqrt, exp, log
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False


def order(num_t, list_t):  # element's order searching

    sorted_list = sorted(list_t, reverse=True)

    return list_t.index(sorted_list[num_t])


def efficient_frontier(mean, cov, ratio_min, ratio_max, portfolio_choice, risk_tolerance):  # 

    mean = mean + 1
    num = mean.shape[0]
    m = list(mean)
    q = matrix(cov)     # 공분산 매트릭스로 변환
    c = matrix(np.zeros(num))
    tm = []

    # tm 을 만드는 loop. 부등식 조건을 만드는 것

    for i in range(num):
        tmp = [float(j == i) for j in range(num)]
        for j in range(num):
            tmp.append(- float(j == i))
        tmp.append(m[i])  # 수정
        tm.append(tmp)

    a = - matrix(tm)
    b0 = np.zeros(2 * num + 1)

    for i in range(num):
        b0[i] = - ratio_min
        b0[num + i] = ratio_max

    # 양식   sol = solvers.qp(P, q, G, h, A, b)
    # objective function = 1/2*x'*P*x + q'*x
    # q' = 0으로 처리 (c)
    # G*x <= h   부등호 조건   ~ 부등호 조건 최대, 최소비중 넣고, 수익률 0 이상되게 하는것
    # A*x = b    등호 조건   ~ 1개밖에 없다. 비중 다 더하면 1

    a2 = matrix(np.ones(num), (1, num))
    b2 = matrix([[1.0]])
    b = matrix(transpose(b0))
    x = solvers.qp(q, c, a, b, a2, b2)['x']  # global minimum variance portfolio
    mu_min = float(transpose(m) @ x)

    # maximum return portfolio
    xm = matrix(np.zeros(num), (num, 1))   # max port의 비중

    for n in range(num):
        xm[n] = ratio_min   # 최소비중으로 깔고 시작

    for n in range(num):
        asset = order(n, list(m))     # 가장 수익률 좋은순으로
        extra = 1 - sum(xm)           # 여분의 비중
        if extra > ratio_max - ratio_min:
            xm[asset] = xm[asset] + ratio_max - ratio_min   # 가장 높은 수익률에 대해 비중 최대치를 넣는다
        else:
            xm[asset] = xm[asset] + extra                # 여분을 가지고 비중에 더한다.
            break
    mu_max = float(transpose(m) @ xm)
    mu_max = (mu_max - mu_min) * risk_tolerance + mu_min   # risk_tolerance 1이 최대치

    # portfolio set 구성

    strategy = [[] for l in range(portfolio_choice)]

    mu = [0 for l in range(portfolio_choice)]
    sigma = [0 for l in range(portfolio_choice)]

    for l in range(portfolio_choice):
        mu[l] = mu_min + (mu_max - mu_min) * l / (portfolio_choice - 1)
    for l in range(portfolio_choice):
        # print(l)
        b[2 * num] = - mu[l]
        x = solvers.qp(q, c, a, b, a2, b2)['x']
        for n in range(num):
            strategy[l].append(x[n])  # 포트폴리오 15개를 쌓아둔다.
            sigma[l] = sqrt(float(transpose(x) @ cov @ x))  # 변동성

    sigma_min = min(sigma)
    sigma_max = max(sigma)
    mu = list(np.array(mu) - 1)

    # 산출물은 EF 상에서 수익률 최소, 수익률 최대, 수익률, 변동성, 포트폴리오, 변동성 최소, 변동성 최대
    return mu, sigma, strategy


def portfolio_generate(mean, cov, time, ratio_min, ratio_max, portfolio_choice, risk_tolerance=1):  # 선택가능한 포트폴리오 pool 생성

    num = mean[0].shape[0]
    mu = np.zeros((time, portfolio_choice))
    sigma = np.zeros((time, portfolio_choice))
    portfolio = np.zeros((time, portfolio_choice, num))

    for t in range(time):
        mu[t], sigma[t], portfolio[t] = efficient_frontier(mean[t], cov[t], ratio_min, ratio_max, portfolio_choice, risk_tolerance)

    return mu, sigma, portfolio


def generate_grid(mu,sigma,W0,C,time,z,grid_gap):  # 목표도달 확률과 최적의 포트폴리오를 계산할 각 시점 및 재산들의 격자 생성

    mu_max, mu_min = mu.max(), mu.min()
    sigma_max, sigma_min = sigma.max(), sigma.min()

    grid_num_list = []  # 시점 별 grid갯수 저장

    wealth_min_list = []
    wealth_max_list = []

    minf = (mu_min-sigma_max**2/2)
    maxf = (mu_max-sigma_min**2/2)

    C_tmp = [C[i]+W0 if i==0 else C[i] for i in range(len(C))]

    for tau in range(time+1):

        t = np.arange(tau)
        
        # Das(2019)에 따라 각 시점별 재산 격자의 최소 최댓값 계산 
        wealth_min_tmp = (((np.array(C_tmp[:tau])>=0)*np.exp(minf*(tau-t)-z*sigma_max*(tau-t)**0.5)+(np.array(C_tmp[:tau])<0)*np.exp(maxf*(tau-t)+z*sigma_max*(tau-t)**0.5))*np.array(C_tmp[:tau])).sum()
        wealth_max_tmp = (((np.array(C_tmp[:tau])>=0)*np.exp(maxf*(tau-t)+z*sigma_max*(tau-t)**0.5)+(np.array(C_tmp[:tau])<0)*np.exp(minf*(tau-t)-z*sigma_max*(tau-t)**0.5))*np.array(C_tmp[:tau])).sum()
        
        # 재산 격자가 grid_gap의 배수가 되게함 & 재산 격자의 최솟값이 0이하가 되는 상황 방지
        wealth_min_tmp = max(grid_gap*(wealth_min_tmp//grid_gap),grid_gap-np.minimum(0,min(C_tmp[tau:])))        
        
        wealth_min_list.append(wealth_min_tmp)
        wealth_max_list.append(wealth_max_tmp)
    
        grid_num_list.append(int(np.ceil((wealth_max_tmp-wealth_min_tmp)/grid_gap))+1)

    # 각 시점 별 재산 격자 저장: 시점별 최솟값 부터 최댓값 까지 grid_gap간격으로 채움
    wealth_grid = np.zeros((time+1,max(grid_num_list)))    
    wealth_grid[0,0]=W0
    grid_num_list[0]=1
    for t in range(1,time+1):
        wealth_grid[t,:grid_num_list[t]] = wealth_min_list[t]+grid_gap*np.arange(grid_num_list[t])
    
    return grid_num_list, wealth_grid


def solve_gbi(time,number_of_grid,mu,sigma,C,wealth_grid,goal,portfolio_choice):
    
    act_num = len(mu[0])
    max_grid_num = max(number_of_grid)

    # 전이확률 p(t,w1,a,w2) : t 시점에 재산이 w1일때 a번째 포트폴리오를 선택했을때 t+1시점에 재산이 w2일 확률
    transition_prob = np.zeros((time,max_grid_num,act_num,max_grid_num))
    for t in range(time):    
        c = C[t]         
        grid_num_now = number_of_grid[t]
        grid_num_next = number_of_grid[t+1]
        w_now = wealth_grid[t,:grid_num_now].reshape(grid_num_now,1,1)
        w_next = wealth_grid[t+1,:grid_num_next].reshape(1,1,grid_num_next)
        P = norm.pdf((np.log(w_next/(w_now+c))-(mu[t].reshape(1,act_num,1)-sigma[t].reshape(1,act_num,1)**2/2))/sigma[t].reshape(1,act_num,1))
        transition_prob[t,:grid_num_now,:,:grid_num_next] = P/P.sum(axis=2).reshape(grid_num_now,act_num,1)

    # Q(t,w,a) : t시점에 재산이 w일때 a번째 포트폴리오를 선택했을때 최종시점에 목표에 도달할 확률(가치함수)
    Q = np.zeros((time+1,max_grid_num,act_num))
    for t in reversed(range(0,time+1)):  # Das(2019)에 따라 가치함수를 계산
        if t==time:
            Q[t,:number_of_grid[t],0] = (wealth_grid[t]+C[t]>=goal).astype(float)        
        else:            
            grid_num_now = number_of_grid[t]
            grid_num_next = number_of_grid[t+1]            
            Q[t,:grid_num_now,:] = np.matmul(transition_prob[t,:grid_num_now,:,:grid_num_next].reshape(grid_num_now,act_num,grid_num_next),Q[t+1,:grid_num_next,:].max(axis=1).reshape(1,grid_num_next,1)).reshape(grid_num_now,act_num)
            
    optimal_value = Q.max(axis=2)  # 각 시점과 상태별 최대의 가치(확률)
    optimal_strategy = Q.argmax(axis=2).astype(int)  # 각 시점과 상태별 가치(확률)을 최대로 하는 포트폴리오

    optimal_strategy[np.where(optimal_value>0.99999)]= 0  # 확률이 1에 가까운 상태에서 포트폴리오를 임의로 선택하는 상황 방지
    
    return transition_prob, optimal_value, optimal_strategy, Q


def state_probability(time,number_of_grid,optimal_strategy,transition_prob,wealth_grid):  # 각 시점 별로 각 상태가 실현될 확률(최적 action 선택시)
        
    sp2 = np.zeros((time+1,max(number_of_grid)))
    sp2[0,0]=1

    # 현재상태의 상태확률(state probability)와 전이확률(transition probability)를 활용하여 다음시점의 상태확률 계산
    for t in range(0,time):    
        for i in range(number_of_grid[t]):        
            a = optimal_strategy[t,i]  
            sp2[t+1,:number_of_grid[t+1]] += sp2[t,i]*transition_prob[t,i,a,:number_of_grid[t+1]]

    return sp2


def glide_path(time,num_of_assets,portfolio,optimal_strategy,number_of_grid,state_probability):  # 각 시점 별 자산들의 기대 비중

    gp = np.zeros((time, num_of_assets))
    # 각각의 상태의 상태확률을 활용하여 기대비중을 계산
    for t in range(time):
        for i in range(number_of_grid[t]):
            gp[t]+=state_probability[t,i]*portfolio[t,optimal_strategy[t,i],:]

    pd.DataFrame(gp).plot.bar(stacked=True)
    plt.legend(['K-stock', 'D-stock', 'N-stock', 'K-bond', 'D-bond', 'N-bond', 'F-C-bond', 'Commodity', 'R-estate', 'Liquidity'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.ylabel('weigh of asset')
    plt.xlabel('year')

    return gp


def state_probability_withdrawal(year,number_of_grid,optimal_strategy,transition_prob,wealth_grid,grid_gap,years_withdrawal,withdrawal,target_prob):  # 중간인출 시점과 금액이 주어졌을 때, 각 시점 별로 각 상태가 실현될 확률(최적 action 선택시)
        
    sp2 = np.zeros((year+1,max(number_of_grid)))
    sp2[0,0]=1

    for t in range(0,year):
        
        # 인출 시점의 state-prob array를 인출 금액만큼 아래로 평행이동
        if t in years_withdrawal:
            for j in range(1,number_of_grid[t]):
                sp2[t,max(j-int(round(withdrawal/grid_gap)),0)]+=sp2[t,j]
                sp2[t,j]=0

        for i in range(number_of_grid[t]):        
            a = optimal_strategy[t,i]  
            sp2[t+1,:number_of_grid[t+1]] += sp2[t,i]*transition_prob[t,i,a,:number_of_grid[t+1]]

    return sp2

