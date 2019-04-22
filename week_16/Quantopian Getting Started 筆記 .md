# Quantopian Getting Started 筆記 (Part2/2)

在上次教程中，我們介紹了notebook環境，並藉由繪製圖表的過程中開發了我們的第一個策略，接下來，我們要嘗試將我們開發出來的策略放到algorithm環境中進行回測。

## Section1: Algorithm API介紹

我們首先先簡單介紹algorithm環境所需要的一些知識，在這邊，algorithm裡面提供的函數可以處理包括訂單調度以及執行等功能，我們在這邊介紹一些核心函數。

### initialize(context)

這是一個初始化函數，以context作為一個input，所有參數的初始化與一次性邏輯都寫在這。這裡另一個重點為context為一python 字典，是用儲存整個過程的狀態，所有的變數應該存入context裡面，而非在宣告一個全局變數，存變數的方法(contex.some_attribute)

### before_trading_start(context, data)

這個函數會在每天交易開始前被呼叫，使用context(參數)與data(每天資料)作為input。這個函數可作為資料的前處理。

### schedule_function(func, day_rule, time_rule)

Quantopian預設在9:30AM-4PM Eastern Time 作交易，schedule_function可以設置客制的交易時間。

### 一個簡單的架構

```python
# Import Algorithm API
import quantopian.algorithm as algo


def initialize(context):
    # Initialize algorithm parameters
    context.day_count = 0
    context.daily_message = "Day {}."
    context.weekly_message = "Time to place some trades!"

    # Schedule rebalance function
    algo.schedule_function(
        rebalance,
        date_rule=algo.date_rules.week_start(),
        time_rule=algo.time_rules.market_open()
    )


def before_trading_start(context, data):
    # Execute any daily actions that need to happen
    # before the start of a trading session
    context.day_count += 1
    log.info(context.daily_message, context.day_count)


def rebalance(context, data):
    # Execute rebalance logic
    log.info(context.weekly_message)
```

## Section 2 Data Processing in Algorithms 

