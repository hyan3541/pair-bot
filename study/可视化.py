import joblib
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data = joblib.load('coint720.pkl')
    print(data)
    coint = pd.Series
    for d in data:
        if d['pair'] == 'ZIL-USDT_SXP-USDT':
            coint = d['coint']
            break
    df = pd.DataFrame(coint)
    df.to_csv("coint.csv")
    plt.figure(figsize=(10, 6))

    # 将True/False转换为 1/0，直接绘制条形图
    plt.bar(coint.index, coint.values.astype(int), width=1, color=['blue' if x else 'gray' for x in coint.values], alpha=0.5)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ZIL-USDT_SXP-USDT coint')
    plt.xticks(rotation=45)
    plt.show()
