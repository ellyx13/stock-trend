from vnstock import VnStock
from function import get_list_symbols_by_industry 

industry = "Công nghệ Thông tin"

list_symbols = get_list_symbols_by_industry(industry)

results = []
vn_stock = VnStock()
source = "VCI"
start_date='2014-01-01'
end_date='2024-12-01'
idx = 0
for symbol_raw in list_symbols:
    print(symbol_raw)
    symbol, company_name = symbol_raw.split(" - ", 1)
    idx += 1
    if idx == 4:
        break
    result = {}
    print("Analyzing: ", symbol)
    result["symbol"] = symbol
    result["company_name"] = company_name
    df, accuracy, cm = vn_stock.analyze(
            symbol=symbol,
            source=source,
            start_date=start_date,
            end_date=end_date,
            show=False,
        )
    result['TN'] = cm[0][0]
    result['FP'] = cm[0][1]
    result['FN'] = cm[1][0]
    result['TP'] = cm[1][1]
    result['accuracy'] = accuracy
    results.append(result)
    
    
print(results)