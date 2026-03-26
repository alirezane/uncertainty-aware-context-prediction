import pandas as pd
import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options


options = Options()
options.headless = True
driver = webdriver.Firefox(executable_path="../drivers/geckodriver", options=options)

start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime(2019, 1, 10)
# end_date = datetime.datetime(2020, 1, 1)
delta = datetime.timedelta(days=1)
data_file_name = f"{datetime.datetime.today().strftime('%Y-%m-%d')}-" \
                 f"historical_weather_data_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"

weather_data = None
unsuccessful_dates = []
attempt = 0
while start_date < end_date:
    if attempt == 0:
        print(f"Crawling historical weather data for {start_date} - attempt: {attempt}")
    else:
        print(f"Re-attempting to crawl historical weather data for {start_date} - attempt: {attempt}")
    url = f'https://www.wunderground.com/history/daily/au/melbourne/YMML' \
          f'/date/{start_date.year}-{start_date.month}-{start_date.day}'
    try:
        driver.get(url)
        tables = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
        daily_data = pd.read_html(tables[1].get_attribute('outerHTML'))[0]
        daily_data.drop_duplicates(subset=['Time'], keep='first', inplace=True)
        daily_data['date'] = start_date.strftime('%Y-%m-%d')
        daily_data['DateTime'] = daily_data.date + ' ' + daily_data.Time
        daily_data['DateTime'] = pd.to_datetime(daily_data.DateTime)
        datetime_column = daily_data.pop('DateTime')
        daily_data.insert(0, 'DateTime', datetime_column)
        daily_data = daily_data[daily_data.DateTime.notna()]
        daily_data.drop(['Time', 'date'], axis=1, inplace=True)
        datetime_range = pd.Series(pd.date_range(start_date, periods=48, freq='30T'), name='DateTime')
        daily_data = pd.merge(left=datetime_range, right=daily_data, how='outer', on=['DateTime'])
        daily_data.sort_values(by=['DateTime'], inplace=True)
        daily_data.set_index(['DateTime'], inplace=True)
        daily_data = daily_data.resample('30min').nearest()
        daily_data.reset_index(inplace=True)
        print(f"Successfully crawled historical weather data for {start_date} -"
              f" attempt: {attempt} - shape: {daily_data.shape}")

        if weather_data is None:
            weather_data = daily_data
        else:
            weather_data = pd.concat([weather_data, daily_data[:48]])
        start_date += delta
        attempt = 0
    except Exception as e:
        print(e)
        print(f"Couldn't crawl data for {start_date} - attempt: {attempt}")
        attempt += 1

weather_data.to_csv(f"./data/{data_file_name}", index=False)
