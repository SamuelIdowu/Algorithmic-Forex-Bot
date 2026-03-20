Here is the prompt to build that specific Hybrid CLI.

### The Prompt

> **Context:**
> I have a Python Algorithmic Trading project. I need a "Hybrid" Command Line Interface (CLI) in a new file named `cli.py`.
> **The Goal:**
> The CLI must support two modes of operation simultaneously for maximum flexibility:
> 1. **Automation Mode (Flags):** If I provide arguments (e.g., `--symbol AAPL`), it should execute immediately without asking questions.
> 2. **Interactive Mode (Prompts):** If I omit arguments (e.g., just running `python cli.py backtest`), it should detect the missing values and interactively prompt me for them using menus/text inputs.
> 
> 
> **Libraries:**
> * **Typer:** For the CLI structure and argument parsing.
> * **Questionary:** For the interactive prompts (menus and text input).
> * **Rich:** For colorful output, tables, and banners.
> 
> 
> **Required Commands:**
> **1. `backtest**`
> * **Arguments:** `symbol` (str), `start` (str), `end` (str).
> * **Logic:**
> * Check if `symbol` is provided. If `None`, use `questionary.text` to ask for it.
> * Check if dates are provided. If `None`, ask for them.
> * Once data is gathered, import and call `run_backtest(symbol, start, end)` from `backtest.backtest_engine`.
> * Use `Rich` to print a success message or a table of results after execution.
> 
> 
> 
> 
> **2. `trade**`
> * **Arguments:** `symbol` (str), `side` (str), `amount` (float).
> * **Logic:**
> * Check for missing arguments and prompt if necessary.
> * For `side`, use `questionary.select` to offer a ["buy", "sell"] menu.
> * **Safety Critical:** Regardless of whether flags were used or not, ask for a final **"Are you sure?"** confirmation using `questionary.confirm` before executing.
> * If confirmed, import and call `place_trade(symbol, side, amount)` from `live_trading.trader`.
> 
> 
> 
> 
> **Deliverables:**
> 1. The complete, runnable code for `cli.py`.
> 2. The implementation should correctly handle importing functions from my existing project structure (`backtest/` and `live_trading/`).
> 3. Add a `main` block so I can run it via `python cli.py`.
> 
> 
**