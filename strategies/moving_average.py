import backtrader as bt

class MovingAverageStrategy(bt.Strategy):
    """
    A simple moving average crossover strategy.
    Buys when the short MA crosses above the long MA, sells when the short MA crosses below the long MA.
    """
    params = dict(
        short_window=20,
        long_window=50,
        printlog=False
    )

    def __init__(self):
        # Keep track of closing price
        self.data_close = self.datas[0].close
        
        # Create moving averages
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, 
            period=self.params.short_window
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, 
            period=self.params.long_window
        )
        
        # Create crossover indicator
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        
        # Initialize order tracking
        self.order = None

    def log(self, txt, dt=None, doprint=False):
        """Logging function"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def next(self):
        # Simply log the closing price of the series
        self.log(f'Close, {self.data_close[0]}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.crossover > 0:  # Buy signal
                self.log(f'BUY CREATE, {self.data_close[0]:.2f}')
                
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if self.crossover < 0:  # Sell signal
                self.log(f'SELL CREATE, {self.data_close[0]:.2f}')

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset the order tracking
        self.order = None

    def stop(self):
        self.log(f'Ending Value: {self.broker.getvalue():.2f}', doprint=True)


class RSIStrategy(bt.Strategy):
    """
    RSI-based strategy: buys when RSI is below 30 (oversold), sells when RSI is above 70 (overbought).
    """
    params = dict(
        rsi_period=14,
        rsi_buy_limit=30,
        rsi_sell_limit=70,
        printlog=False
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.rsi = bt.indicators.RSI_SMA(self.datas[0].close, period=self.params.rsi_period)
        
        # Initialize order tracking
        self.order = None

    def log(self, txt, dt=None, doprint=False):
        """Logging function"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            if self.rsi[0] < self.params.rsi_buy_limit:  # Buy signal
                self.log(f'BUY CREATE, {self.data_close[0]:.2f}')
                self.order = self.buy()
        else:
            if self.rsi[0] > self.params.rsi_sell_limit:  # Sell signal
                self.log(f'SELL CREATE, {self.data_close[0]:.2f}')
                self.order = self.sell()

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset the order tracking
        self.order = None

    def stop(self):
        self.log(f'Ending Value: {self.broker.getvalue():.2f}', doprint=True)