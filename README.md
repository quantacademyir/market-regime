# market-regime

کد و داده سه پست «مارکت رژیم» در [Quant Academy](https://quantacademy.ir/blog/market-regime-intro/).

| # | پست | اسکریپت |
|---|---|---|
| 1 | [مارکت رژیم چیست؟ نوسان بازار ثابت نیست](https://quantacademy.ir/blog/market-regime-intro/) | `scripts/01-regime-volatility.py` |
| 2 | [تشخیص مارکت رژیم بدون نگاه به آینده](https://quantacademy.ir/blog/market-regime-detection/) | `scripts/02-regime-detection.py` |
| 3 | [فیلتر مارکت رژیم: بهتر شدن یا Overfitting؟](https://quantacademy.ir/blog/market-regime-filter/) | `scripts/03-regime-filter.py` |

## اجرا

```bash
pip install -r requirements.txt
python3 scripts/01-regime-volatility.py
python3 scripts/02-regime-detection.py
python3 scripts/03-regime-filter.py
```

هر اسکریپت عددهای پست را چاپ می‌کند و نمودارها را در پوشه `figures/` می‌نویسد.

## داده

`data/btc_daily_5y.csv` پنج سال کندل روزانه **شبیه‌سازی‌شده** است (نه قیمت واقعی)، همان داده دوره «بک‌تست درست». نتیجه‌ها درباره بازار واقعی نیستند؛ فقط روش را نشان می‌دهند.

## سلب مسئولیت

آموزشی است. سیگنال، پیشنهاد خرید و فروش یا وعده سود نیست.
