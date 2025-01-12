import os.path

pre_data_path = r'D:\coin-binance-spot-swap-preprocess-pkl-1h-2024-12-04'
swap_path = os.path.join(pre_data_path, 'swap_dict.pkl')
root_path = os.path.abspath(os.path.dirname(__file__))
min_qty_path = os.path.join(root_path, 'data', '最小下单量.csv')
plot_path = os.path.join(root_path, 'data', 'graph')
